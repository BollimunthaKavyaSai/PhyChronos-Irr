"""Fetch hourly ERA5 (Open-Meteo archive) variables missing from the dataset: global shortwave
radiation, vapour-pressure deficit and FAO-56 Penman-Monteith reference ET0, for the site."""
import requests
import pandas as pd
import config as C

URL = "https://archive-api.open-meteo.com/v1/archive"


def main():
    p = dict(latitude=C.SITE["lat"], longitude=C.SITE["lon"], start_date="2025-02-01", end_date="2025-09-30",
             hourly="et0_fao_evapotranspiration,shortwave_radiation,vapour_pressure_deficit,temperature_2m",
             timezone=C.SITE["tz"])
    r = requests.get(URL, params=p, timeout=60)
    r.raise_for_status()
    j = r.json()["hourly"]
    df = pd.DataFrame(j).rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"])
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.PROCESSED / "era5_hourly_extra.csv", index=False)
    print(df.describe().round(2).to_string())
    print("Season ET0 sum (mm):", round(df["et0_fao_evapotranspiration"].sum(), 1))


if __name__ == "__main__":
    main()
