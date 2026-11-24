"""R1 audit · full run of the reference adapter (src/football/adapter.events_to_long, v2) over the cached events, as an
independent reproduction of the vectorised adapter's output.  python3 -m analysis.audit.reingest_real SHARD NSHARDS [CACHE_DIR] [OUT_DIR]
Compare with compare_adapters.py."""
import sys, time, json, warnings, traceback
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.football.adapter import events_to_long
CACHE = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "data/raw/sb_events"; OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "data/interim/audit_real"
OUT.mkdir(parents=True, exist_ok=True); (OUT / "_done").mkdir(exist_ok=True)
COLS = ["unit_id", "domain", "event_id", "window", "metric", "value", "metric_class", "context", "schema_ver"]
ml = json.loads((ROOT / "data/interim/match_list.json").read_text()); shard, nsh = int(sys.argv[1]), int(sys.argv[2])
log = open(OUT / f"log_real{shard}.txt", "a"); t0 = time.time(); done = 0
for order, (comp, season, mid) in enumerate(ml):
    if order % nsh != shard: continue
    marker = OUT / "_done" / f"{order:05d}_{mid}"
    if marker.exists(): continue
    try:
        ev = pd.read_pickle(CACHE / f"{mid}.pkl.gz")
        for mode in ("score", "concede"):
            recs = events_to_long(ev, match_id=str(mid), competition=comp, event_mode=mode)
            d = OUT / f"real__{mode}"; d.mkdir(exist_ok=True)
            pd.DataFrame(recs, columns=COLS).to_pickle(d / f"{order:05d}_{mid}.pkl")
        marker.write_text("ok"); done += 1
        if done % 50 == 0: log.write(f"{time.strftime('%H:%M:%S')} {done} matches {time.time()-t0:.0f}s\n"); log.flush()
    except Exception as e:
        log.write(f"ERROR {order} {mid}: {e}\n{traceback.format_exc()}\n"); log.flush()
log.write(f"FINISHED shard {shard}: {done} in {time.time()-t0:.0f}s\n"); log.close()
