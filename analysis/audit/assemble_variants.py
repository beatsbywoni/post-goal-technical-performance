"""R1 audit · assemble reingest_variants.py chunks into one parquet per variant and mode, and verify that the v1 variant
reproduces the submitted parquets row for row.  python3 -m analysis.audit.assemble_variants assemble|verify [OUT_DIR] [VARIANT_DIR]
The v2 parquets are the ones in data/features/; the submitted ones are kept in data/features/_v1_submitted/."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "data/interim/audit_variants"; VAR = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data/interim/variants"
FN = {"score": "football_momentum_long.parquet", "concede": "football_long.parquet"}


def assemble():
    for v in ("v1", "v2", "v2_own", "v2_cont"):
        dest = VAR / v; dest.mkdir(parents=True, exist_ok=True)
        for mode in ("score", "concede"):
            files = sorted((OUT / f"{v}__{mode}").glob("*.pkl"))
            df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
            df.to_parquet(dest / FN[mode], index=False); print(v, mode, len(files), "matches", len(df), "rows", flush=True)


def verify_v1():
    for mode in ("score", "concede"):
        a = pd.read_parquet(REPO / "data/features/_v1_submitted" / FN[mode]); b = pd.read_parquet(VAR / "v1" / FN[mode])
        for df in (a, b):
            df["ctx"] = df.context.apply(lambda c: json.dumps({k: c.get(k) for k in sorted(c)}, sort_keys=True))
        cols = ["unit_id", "event_id", "window", "metric", "ctx"]
        m = a.merge(b, on=cols, how="outer", suffixes=("_a", "_b"), indicator=True)
        only_a = int((m._merge == "left_only").sum()); only_b = int((m._merge == "right_only").sum())
        both = m[m._merge == "both"]; bad = int((~np.isclose(both.value_a, both.value_b, rtol=1e-12, atol=1e-12)).sum())
        print(f"{mode}: submitted {len(a)} rows, v1 re-ingest {len(b)} rows; only-submitted {only_a}, only-reingest {only_b}, value mismatches {bad}")
        if only_a or only_b:
            print(m[m._merge != "both"].head(8)[cols + ["_merge"]].to_string())


if __name__ == "__main__":
    {"assemble": assemble, "verify": verify_v1}[sys.argv[1]]()
