"""E2 zero-shot time-series foundation models.

Models: Chronos-2 (univariate and with past+future covariates), TimesFM-2.5, Moirai-2.0, TiRex.
Context: up to CTX hours of gap-filled history before each origin.
Usage: python e2_foundation.py [chronos2 chronos2cov timesfm moirai2 tirex]
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
import config as C
import e2_panel as P

H, CTX = P.H, 512
torch.set_num_threads(max(1, torch.get_num_threads()))


def contexts():
    """List of (z, fold, t, ctx_frame) for all evaluation origins."""
    items = []
    for z, f, h, orig in P.all_tasks():
        h = h.copy()
        h["sm_f"] = P.sm_filled(h)
        for c in P.COVS_PAST:
            h[c] = h[c].interpolate(limit_area="inside").ffill().bfill()
        for t in orig:
            ctx = h.loc[t - pd.Timedelta(hours=CTX - 1): t]
            fut = h.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]
            items.append((z, f["fold"], t, ctx, fut))
    return items


def run_chronos2(items, covariates):
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    frames = []
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        ctx_rows, fut_rows = [], []
        for j, (z, k, t, ctx, fut) in enumerate(chunk):
            iid = f"{i + j}"
            c = pd.DataFrame({"item_id": iid, "timestamp": ctx.index, "target": ctx["sm_f"].values})
            if covariates:
                for cv in P.COVS_PAST:
                    c[cv] = ctx[cv].values
                fr = pd.DataFrame({"item_id": iid, "timestamp": fut.index})
                for cv in P.COVS_FUTURE:
                    fr[cv] = fut[cv].values
                fut_rows.append(fr)
            ctx_rows.append(c)
        df = pd.concat(ctx_rows)
        fdf = pd.concat(fut_rows) if covariates else None
        out = pipe.predict_df(df, future_df=fdf, prediction_length=H, quantile_levels=P.QUANTILES,
                              validate_inputs=False)
        for j, (z, k, t, ctx, fut) in enumerate(chunk):
            o = out[out["item_id"] == f"{i + j}"].sort_values("timestamp")
            q = np.sort(o[[str(x) for x in P.QUANTILES]].values, axis=1)
            frames.append(P.pred_frame(z, k, t, q[:, 4], q))
        print(f"  chronos2{'cov' if covariates else ''} {i + len(chunk)}/{len(items)}", flush=True)
    return frames


def run_timesfm(items):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=H * 2, normalize_inputs=True,
                                     use_continuous_quantile_head=True, fix_quantile_crossing=True))
    frames = []
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        point, quant = m.forecast(horizon=H, inputs=[c[3]["sm_f"].values.astype(np.float32) for c in chunk])
        # quant: [batch, H, 10] = mean + deciles 0.1..0.9
        for j, (z, k, t, ctx, fut) in enumerate(chunk):
            q = np.sort(quant[j, :H, 1:10], axis=1)
            frames.append(P.pred_frame(z, k, t, q[:, 4], q))
    return frames


def run_moirai2(items):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    model = Moirai2Forecast(module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                            prediction_length=H, context_length=CTX, target_dim=1,
                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    frames = []
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        x = np.stack([c[3]["sm_f"].values[-CTX:] for c in chunk]).astype(np.float32)
        out = np.asarray(model.predict(x))          # [batch, n_quantiles(9), H]
        for j, (z, k, t, ctx, fut) in enumerate(chunk):
            q = np.sort(out[j].T, axis=1)
            frames.append(P.pred_frame(z, k, t, q[:, 4], q))
    return frames


def run_tirex(items):
    from tirex import load_model
    m = load_model("NX-AI/TiRex", device="cpu", backend="torch")
    frames = []
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        x = torch.tensor(np.stack([c[3]["sm_f"].values[-CTX:] for c in chunk]), dtype=torch.float32)
        q, mean = m.forecast(context=x, prediction_length=H)   # q: [batch, H, 9]
        q = np.sort(np.asarray(q), axis=2)
        for j, (z, k, t, ctx, fut) in enumerate(chunk):
            frames.append(P.pred_frame(z, k, t, q[j, :, 4], q[j]))
    return frames


RUNNERS = {
    "chronos2": ("Chronos-2", lambda it: run_chronos2(it, False)),
    "chronos2cov": ("Chronos-2+cov", lambda it: run_chronos2(it, True)),
    "timesfm": ("TimesFM-2.5", run_timesfm),
    "moirai2": ("Moirai-2.0", run_moirai2),
    "tirex": ("TiRex", run_tirex),
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(RUNNERS)
    items = contexts()
    print(f"{len(items)} origins", flush=True)
    for w in which:
        name, fn = RUNNERS[w]
        t0 = time.time()
        try:
            frames = fn(items)
            P.save_preds(name, frames)
            print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{name}: FAILED {e!r}", flush=True)
