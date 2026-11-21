"""Reference controllers from E8 (Fixed-interval, EDML-style) on the same plants/window as E8b."""
import sys
import pandas as pd
import config as C
import e8_control as E8
noise = "--noise" in sys.argv
rows = []
for z in C.PRIMARY_SECTORS:
    for plant in ["A", "B"]:
        for c in ["Fixed-interval", "EDML"]:
            r = E8.run(z, plant, c, noise)
            rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(C.RESULTS / "e8" / f"e8_refs{'_noise' if noise else ''}.csv", index=False)
