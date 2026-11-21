"""E2 deep-learning baselines (Time-Series-Library implementations), trained per fold as global
models over sectors 1, 2, 4, 5. Multivariate-to-univariate (MS): soil moisture + past covariates.

Usage: python e2_deep.py [DLinear PatchTST iTransformer TimeXer TiDE TimeMixer TSMixer LSTM]
"""
import sys
import time
import types
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config as C
import e2_panel as P

TSL = C.ROOT / "external" / "Time-Series-Library"
sys.path.insert(0, str(TSL))
torch.manual_seed(C.SEED)
torch.set_num_threads(3)
np.random.seed(C.SEED)

SEQ, LABEL, H = 168, 48, P.H
FEATS = P.COVS_PAST + ["sm_f"]           # target last (TSLib MS convention)


def time_marks(idx):
    return np.stack([idx.hour / 23 - 0.5, idx.dayofweek / 6 - 0.5, (idx.day - 1) / 30 - 0.5,
                     (idx.dayofyear - 1) / 365 - 0.5], 1).astype(np.float32)


def prep(h):
    h = h.copy()
    h["sm_f"] = P.sm_filled(h)
    for c in P.COVS_PAST:
        h[c] = h[c].interpolate(limit_area="inside").ffill().bfill().fillna(0)
    return h


def windows(h, ends, mu, sd):
    X, Xm, Y, Ym = [], [], [], []
    vals = ((h[FEATS] - mu) / sd).values.astype(np.float32)
    marks = time_marks(h.index)
    pos = {t: i for i, t in enumerate(h.index)}
    for t in ends:
        i = pos.get(t)
        if i is None or i - SEQ + 1 < 0 or i + H >= len(h):
            continue
        X.append(vals[i - SEQ + 1:i + 1]); Xm.append(marks[i - SEQ + 1:i + 1])
        Y.append(vals[i - LABEL + 1:i + H + 1]); Ym.append(marks[i - LABEL + 1:i + H + 1])
    if not X:
        return None
    return [torch.tensor(np.stack(a)) for a in (X, Xm, Y, Ym)]


def cfg(model):
    return types.SimpleNamespace(
        task_name="long_term_forecast", seq_len=SEQ, label_len=LABEL, pred_len=H, enc_in=len(FEATS),
        dec_in=len(FEATS), c_out=len(FEATS), d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=128,
        factor=3, dropout=0.1, embed="timeF", freq="h", activation="gelu", moving_avg=25,
        features="MS", patch_len=24, use_norm=1, top_k=5, num_class=1, channel_independence=1,
        decomp_method="moving_avg", down_sampling_layers=2, down_sampling_method="avg",
        down_sampling_window=2, output_attention=False, num_kernels=6, distil=True)


class LSTMNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.rnn = nn.LSTM(c.enc_in, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(64, H)

    def forward(self, x, xm, xd, xdm):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).unsqueeze(-1)


def build(name):
    if name == "LSTM":
        return LSTMNet(cfg(name))
    import importlib
    return importlib.import_module(f"models.{name}").Model(cfg(name))


def forward(model, b):
    x, xm, y, ym = b
    dec = torch.cat([y[:, :LABEL], torch.zeros_like(y[:, LABEL:])], 1)
    out = model(x, xm, dec, ym)
    return out[:, -H:, -1]


def run_model(name):
    frames = []
    hs = {z: prep(P.hourly(z)) for z in C.PRIMARY_SECTORS}
    task = {}
    for z, f, h, orig in P.all_tasks():
        task.setdefault(f["fold"], []).append((z, f, orig))
    for k, lst in sorted(task.items()):
        # normalisation stats from training periods only
        tr_parts = [hs[z].loc[f["train_start"]:f["train_end"], FEATS] for z, f, _ in lst]
        mu, sd = pd.concat(tr_parts).mean(), pd.concat(tr_parts).std().replace(0, 1)
        tr, va = [], []
        for z, f, _ in lst:
            h = hs[z]
            ends = h.loc[f["train_start"] + pd.Timedelta(hours=SEQ):f["train_end"] - pd.Timedelta(hours=H)].index
            ends = ends[h["sm"].reindex(ends).notna().values]
            cut = int(len(ends) * 0.85)
            tr.append(windows(h, ends[:cut], mu, sd)); va.append(windows(h, ends[cut:], mu, sd))
        cat = lambda parts: [torch.cat([p[i] for p in parts if p is not None]) for i in range(4)]
        tr, va = cat(tr), cat(va)
        model = build(name)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state, bad = 1e9, None, 0
        for ep in range(12):
            model.train()
            perm = torch.randperm(len(tr[0]))
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                b = [t[idx] for t in tr]
                loss = nn.functional.mse_loss(forward(model, b), b[2][:, -H:, -1])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(forward(model, va), va[2][:, -H:, -1]).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        model.load_state_dict(best_state)
        model.eval()
        for z, f, orig in lst:
            if not orig:
                continue
            w = windows(hs[z], orig, mu, sd)
            with torch.no_grad():
                p = forward(model, w).numpy() * sd["sm_f"] + mu["sm_f"]
            valid = [t for t in orig if t in hs[z].index]
            for t, pp in zip(valid, p):
                frames.append(P.pred_frame(z, k, t, pp))
        print(f"  {name} fold {k}: train {len(tr[0])} val {len(va[0])} best val {best:.4f} epochs {ep + 1}", flush=True)
    return frames


if __name__ == "__main__":
    names = sys.argv[1:] or ["DLinear", "PatchTST", "iTransformer", "TimeXer", "TiDE", "TimeMixer", "TSMixer", "LSTM"]
    for n in names:
        t0 = time.time()
        try:
            P.save_preds(n, run_model(n))
            print(f"{n}: done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{n}: FAILED {e!r}", flush=True)
