#!/usr/bin/env python3
"""R1 audit · compare the reference adapter's full output (reingest_real.py chunks) with the vectorised adapter's
parquets (data/features/football_*_long.parquet, variant v2) row for row.

  python3 -m analysis.audit.compare_adapters [REAL_DIR] [FEATURES_DIR]

A record is keyed by (unit_id, event_id, window, metric); values are compared at 1e-9 and the context dicts
compared as sorted JSON. Writes docs/ijpas_r1_adapter_comparison.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/interim/audit_real"
FEAT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/features"
FN = {"score": "football_momentum_long.parquet", "concede": "football_long.parquet"}
KEY = ["unit_id", "event_id", "window", "metric"]


def _norm(v):
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)): return round(float(v), 6)   # parquet stores ints as floats in unified structs
    return str(v)


def ctx_json(c):
    return json.dumps({k: _norm(c[k]) for k in sorted(c) if c[k] is not None}, sort_keys=True)   # parquet struct columns add None for absent keys


def main(chunk=150):
    out = {"real_dir": str(REAL), "features_dir": str(FEAT)}
    for mode in ("score", "concede"):
        files = sorted((REAL / f"real__{mode}").glob("*.pkl"))
        fast = pd.read_parquet(FEAT / FN[mode]); fast["mid"] = fast.unit_id.str.rsplit("-", n=1).str[-1]
        r = {"matches_compared": 0, "rows_reference": 0, "rows_vectorised": 0, "only_in_reference": 0, "only_in_vectorised": 0,
             "value_mismatches": 0, "context_mismatches": 0, "metric_class_mismatches": 0}
        examples = []
        for i in range(0, len(files), chunk):          # chunked to stay within memory
            fs = files[i:i + chunk]; mids = {f.stem.split("_", 1)[1] for f in fs}
            real = pd.concat([pd.read_pickle(f) for f in fs], ignore_index=True)
            sub = fast[fast.mid.isin(mids)].copy()
            for df in (real, sub):
                df["ctx"] = df.context.apply(ctx_json)
            m = real[KEY + ["value", "metric_class", "ctx"]].merge(sub[KEY + ["value", "metric_class", "ctx"]], on=KEY, how="outer", suffixes=("_r", "_f"), indicator=True)
            both = m[m._merge == "both"]
            r["matches_compared"] += len(mids); r["rows_reference"] += int(len(real)); r["rows_vectorised"] += int(len(sub))
            r["only_in_reference"] += int((m._merge == "left_only").sum()); r["only_in_vectorised"] += int((m._merge == "right_only").sum())
            r["value_mismatches"] += int((~np.isclose(both.value_r, both.value_f, rtol=1e-9, atol=1e-9)).sum())
            r["context_mismatches"] += int((both.ctx_r != both.ctx_f).sum()); r["metric_class_mismatches"] += int((both.metric_class_r != both.metric_class_f).sum())
            if len(examples) < 5:
                bad = m[(m._merge != "both")]
                if len(bad): examples += bad.head(5 - len(examples))[KEY + ["_merge"]].astype(str).to_dict("records")
            print(mode, i + len(fs), "/", len(files), {k: v for k, v in r.items() if k.endswith("mismatches") or k.startswith("only")}, flush=True)
            del real, sub, m, both
        r["identical"] = bool(r["only_in_reference"] == r["only_in_vectorised"] == r["value_mismatches"] == r["context_mismatches"] == r["metric_class_mismatches"] == 0)
        if examples: r["examples"] = examples
        out[mode] = r; print(mode, r, flush=True)
        del fast
    (ROOT / "docs/ijpas_r1_adapter_comparison.json").write_text(json.dumps(out, indent=1))
    print("✓ docs/ijpas_r1_adapter_comparison.json")


if __name__ == "__main__":
    main()
