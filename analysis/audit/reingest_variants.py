"""R1 audit · full re-ingestion of the 2,404 matches with src/football/adapter_fast under four variants.

  python3 -m analysis.audit.reingest_variants SHARD NSHARDS [CACHE_DIR] [OUT_DIR]

Variants: v1 = the submitted pipeline (reproduces data/features/_v1_submitted exactly); v2 = corrected pipeline
(period-aware clock, shoot-outs excluded, dismissal flags, controls exclude 15 min after any goal); v2_own = v2 but
controls exclude only the analysed team's own post-goal windows (the submitted convention); v2_cont = v2 with a
continuous clock across periods. Match list: data/interim/match_list.json (competition, season, match_id).
Output: OUT_DIR/<variant>__<mode>/<order>_<match_id>.pkl, assembled by assemble_variants.py.
"""
from __future__ import annotations
import sys, time, warnings, json, traceback
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.football.adapter_fast import events_to_long_fast
VARIANTS = ["v1", "v2", "v2_own", "v2_cont"]; MODES = ["score", "concede"]
CACHE = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "data/raw/sb_events"; OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "data/interim/audit_variants"
CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True); (OUT / "_done").mkdir(exist_ok=True)
COLS = ["unit_id", "domain", "event_id", "window", "metric", "value", "metric_class", "context", "schema_ver"]


def main(shard, nshards):
    ml = json.loads((ROOT / "data/interim/match_list.json").read_text())
    from statsbombpy import sb
    log = open(OUT / f"log_shard{shard}.txt", "a"); t0 = time.time(); done = 0
    for order, (comp, season, mid) in enumerate(ml):
        if order % nshards != shard:
            continue
        marker = OUT / "_done" / f"{order:05d}_{mid}"
        if marker.exists():
            continue
        try:
            cp = CACHE / f"{mid}.pkl.gz"
            if cp.exists():
                ev = pd.read_pickle(cp)
            else:
                for attempt in range(4):
                    try:
                        ev = sb.events(match_id=mid); break
                    except Exception:
                        time.sleep(2 + 3 * attempt)
                else:
                    raise RuntimeError("download failed")
                ev.to_pickle(cp, compression="gzip")
            for v in VARIANTS:
                for mode in MODES:
                    recs = events_to_long_fast(ev, match_id=str(mid), competition=comp, event_mode=mode, variant=v)
                    d = OUT / f"{v}__{mode}"; d.mkdir(exist_ok=True)
                    pd.DataFrame(recs, columns=COLS).to_pickle(d / f"{order:05d}_{mid}.pkl")
            marker.write_text("ok"); done += 1
            if done % 50 == 0:
                log.write(f"{time.strftime('%H:%M:%S')} shard {shard}: {done} matches, {time.time()-t0:.0f}s\n"); log.flush()
        except Exception as e:  # noqa: BLE001
            log.write(f"ERROR {order} {mid}: {e}\n{traceback.format_exc()}\n"); log.flush()
    log.write(f"FINISHED shard {shard}: {done} in {time.time()-t0:.0f}s\n"); log.close()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
