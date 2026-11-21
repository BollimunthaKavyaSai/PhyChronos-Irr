"""Central configuration for the PhyChronos-Irr project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "primary_c837v6p8ph" / "extracted" / \
    "Soil Moisture, Irrigation Actuator and Weather Dat"
MERGED = RAW / "02_processed_data" / "merged"
PUBLISHED = RAW / "02_processed_data" / "preprocessed"
EVENTS = RAW / "02_processed_data" / "irrigation_events" / "final_irrigation_events.csv"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SITE = dict(lat=40.349306, lon=18.081556, tz="Europe/Rome")
SECTORS = [1, 2, 3, 4, 5]
PRIMARY_SECTORS = [1, 2, 4, 5]          # sector 3 = natural-fault case study

# Agronomist thresholds on the probe's 0-100 relative scale (README sec. 3)
THRESH = {
    1: dict(PWP=20, MADP=40, FC=60, SP=80, flow=1.1, start=40, stop=60),
    2: dict(PWP=20, MADP=40, FC=60, SP=80, flow=1.1, start=40, stop=60),
    3: dict(PWP=30, MADP=60, FC=80, SP=90, flow=1.1, start=60, stop=80),
    4: dict(PWP=40, MADP=80, FC=90, SP=95, flow=1.1, start=80, stop=90),
    5: dict(PWP=35, MADP=70, FC=90, SP=95, flow=0.4, start=50, stop=60),
}
SECTOR3_PROBE_DEATH = "2025-06-25 08:10"

STEP_MIN = 10
STEPS_PER_HOUR = 6
DEADBAND_FILL_H = 6        # max hours a reading is carried forward (3 % deadband transmission)
FLATLINE_H = 24            # identical repeated readings for >= 24 h -> stuck sensor
                           # (readings are integer-quantised: 6-12 h constant runs are normal)
SPIKE_WIN, SPIKE_THR = 9, 15.0
HORIZONS_H = [1, 6, 12, 24]
MIN_TARGET_COVERAGE = 0.8  # share of the 24 h window that must be observed/fillable

SEED = 42


def agronomic_class(x, s):
    t = THRESH[s]
    import numpy as np
    return np.select([x < t["PWP"], x < t["MADP"], x < t["FC"], x < t["SP"]], [0, 1, 2, 3], 4)
