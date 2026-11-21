"""Build the corrected, true-time per-sector dataset from the published merged layer.

Corrections vs. the published preprocessed layer:
  * rain: merged layer repeats the hourly ERA5 value in every 10-min bin -> divide by 6
  * targets: computed on the true 10-min time grid (not on row offsets of a gappy series)
  * soil-moisture gaps: forward-filled up to DEADBAND_FILL_H (3 % deadband transmission),
    flagged as censored; longer gaps stay missing
  * stuck-sensor (flatline) segments and sector-3 post-failure readings masked and flagged
  * EC / pH error codes removed
  * sector-5 water volumes at 0.4 L/min (merged layer is already correct)
"""
import numpy as np
import pandas as pd
import config as C


def clean_sector(z: int) -> pd.DataFrame:
    d = pd.read_csv(C.MERGED / f"dataset_zone_{z}.csv", parse_dates=["ts"]).set_index("ts")
    d = d.asfreq("10min")
    out = pd.DataFrame(index=d.index)

    # ---- weather (outdoor ERA5) ----
    for c in ["weather_temp", "weather_humidity", "weather_pressure", "weather_wind_speed",
              "weather_radiation", "soil_temperature_0-7cm", "soil_temperature_7-18cm"]:
        out[c] = d[c]
    out["weather_rain"] = d["weather_rain"] / 6.0                     # fix 6x overcount
    for c in ["inside_temperature", "inside_humidity", "outside_temperature", "outside_humidity"]:
        out[c] = d[c]                                                  # on-site (partial coverage)

    # ---- soil moisture ----
    sm = d["soil_moisture"].copy()
    observed_raw = sm.notna()
    sm = sm.where(sm.between(0, 115)).clip(upper=100)
    obs = sm.dropna()
    med = obs.rolling(C.SPIKE_WIN, center=True, min_periods=3).median()
    spikes = (obs - med).abs() > C.SPIKE_THR
    obs = obs[~spikes]
    # flatline: consecutive identical observed readings spanning >= FLATLINE_H
    grp = (obs != obs.shift()).cumsum()
    span = obs.index.to_series().groupby(grp.values).transform(lambda t: (t.max() - t.min()))
    cnt = obs.groupby(grp.values).transform("size")
    flat = (span >= pd.Timedelta(hours=C.FLATLINE_H)) & (cnt >= 6)
    fault = pd.Series(False, index=d.index)
    fault.loc[obs.index[flat.values]] = True
    obs = obs[~flat.values]
    if z == 3:
        dead = obs.index >= pd.Timestamp(C.SECTOR3_PROBE_DEATH)
        fault.loc[obs.index[dead]] = True
        fault.loc[d.index >= pd.Timestamp(C.SECTOR3_PROBE_DEATH)] = True
        obs = obs[~dead]
    sm_obs = obs.reindex(d.index)
    out["sm_obs"] = sm_obs
    out["sm_observed"] = sm_obs.notna()
    out["sm_raw_present"] = observed_raw
    out["sm_fault"] = fault | (observed_raw & sm_obs.isna() & ~d["soil_moisture"].between(0, 115))
    out["sm_fault"] = out["sm_fault"] | (observed_raw & sm_obs.isna())   # any removed reading
    lim = C.DEADBAND_FILL_H * C.STEPS_PER_HOUR
    sm_fill = sm_obs.ffill(limit=lim)
    # do not carry a value across a fault segment
    sm_fill[fault] = np.nan
    out["sm"] = sm_fill
    out["sm_censored"] = out["sm"].notna() & ~out["sm_observed"]

    # ---- EC / pH ----
    ec = d["ec"].where(d["ec"].between(0, 1000))
    ph = d["ph"].where(d["ph"].between(3, 9))
    out["ec"] = ec.ffill(limit=lim)
    out["ph"] = ph.ffill(limit=lim)

    # ---- irrigation ----
    out["irr_min"] = d["irrigation_duration_minutes"].fillna(0)
    out["liters"] = d["liters_total"].fillna(0)
    H = C.STEPS_PER_HOUR
    out["water_past_4h"] = out["liters"].rolling(4 * H, min_periods=1).sum()
    fwd = out["liters"][::-1].rolling(24 * H, min_periods=1).sum()[::-1]
    out["water_to_24h"] = fwd.shift(-1).fillna(0)            # (t, t+24h]

    # ---- calendar (kept only for the reproduction / ablation) ----
    idx = out.index
    out["month"], out["day"], out["hour"] = idx.month, idx.day, idx.hour
    out["quarter"] = idx.quarter
    out["day_period"] = np.select([(idx.hour >= 5) & (idx.hour < 12), (idx.hour >= 12) & (idx.hour < 17),
                                   (idx.hour >= 17) & (idx.hour < 21)], [0, 1, 2], 3)

    # ---- true-time targets ----
    for h in C.HORIZONS_H:
        out[f"y_point_{h}h"] = out["sm"].shift(-h * H)
    win = 24 * H
    avail = out["sm"].notna().astype(float)
    s = out["sm"].fillna(0)
    fsum = s[::-1].rolling(win, min_periods=1).sum()[::-1].shift(-1)
    fcnt = avail[::-1].rolling(win, min_periods=1).sum()[::-1].shift(-1)
    mean24 = fsum / fcnt
    out["y_mean_24h"] = mean24.where(fcnt >= C.MIN_TARGET_COVERAGE * win)
    out["sector"] = z
    return out


def main():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in C.SECTORS:
        df = clean_sector(z)
        df.to_csv(C.PROCESSED / f"zone_{z}.csv")
        valid = df["sm"].notna()
        rows.append(dict(sector=z, grid_rows=len(df), sm_available=int(valid.sum()),
                         observed=int(df["sm_observed"].sum()), censored=int(df["sm_censored"].sum()),
                         fault_flags=int(df["sm_fault"].sum()),
                         rows_with_y_mean_24h=int((valid & df["y_mean_24h"].notna()).sum()),
                         rain_mm=round(df["weather_rain"].sum(), 1), water_L=round(df["liters"].sum(), 1)))
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    rep.to_csv(C.PROCESSED / "build_report.csv", index=False)


if __name__ == "__main__":
    main()
