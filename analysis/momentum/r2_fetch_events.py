"""R2: StatsBomb 오픈데이터 경기별 이벤트를 sb_ingest 와 같은 캐시 형식(<cache>/<match_id>.pkl.gz)으로 내려받는다.

  python3 -m analysis.momentum.r2_fetch_events --cache-dir data/raw/sb_events [--workers 4] [--heldout]

main 표본은 data/interim/match_list.json(2,404경기), --heldout 은 configs/sb_heldout.yaml 의 대회에서 경기 목록을 받는다.
이미 캐시된 경기는 건너뛴다. 다운로드 실패는 stderr 에 적고 계속한다(완료 후 재실행하면 실패분만 다시 받는다).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore", module="statsbombpy")
ROOT = Path(__file__).resolve().parents[2]


def match_ids(heldout: bool) -> list[int]:
    from statsbombpy import sb
    if not heldout:
        return [int(r[2]) for r in json.load(open(ROOT / "data/interim/match_list.json"))]
    cfg = yaml.safe_load((ROOT / "configs/sb_heldout.yaml").read_text(encoding="utf-8"))
    out = []
    for c in cfg["competitions"]:
        m = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
        out.extend(int(x) for x in m["match_id"].tolist())
    return out


def fetch(mid: int, cache: Path) -> tuple[int, str]:
    from statsbombpy import sb
    cp = cache / f"{mid}.pkl.gz"
    if cp.exists():
        return mid, "cached"
    for attempt in range(3):
        try:
            ev = sb.events(match_id=mid)
            tmp = cache / f"{mid}.pkl.gz.part"
            ev.to_pickle(tmp, compression="gzip")
            tmp.replace(cp)
            return mid, "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return mid, f"fail: {err}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None, help="default data/raw/sb_events (main) or data/raw/sb_events_heldout (--heldout)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--heldout", action="store_true")
    a = ap.parse_args(argv)
    cache = ROOT / (a.cache_dir or ("data/raw/sb_events_heldout" if a.heldout else "data/raw/sb_events"))
    cache.mkdir(parents=True, exist_ok=True)
    mids = match_ids(a.heldout)
    todo = [m for m in mids if not (cache / f"{m}.pkl.gz").exists()]
    print(f"{len(mids)} matches, {len(todo)} to fetch → {cache}", flush=True)
    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fetch, m, cache) for m in todo]
        for i, f in enumerate(as_completed(futs), 1):
            mid, st = f.result()
            if st == "ok":
                n_ok += 1
            elif st.startswith("fail"):
                n_fail += 1
                print(f"! {mid}: {st}", file=sys.stderr, flush=True)
            if i % 100 == 0:
                print(f"  …{i}/{len(todo)} ok={n_ok} fail={n_fail} ({time.time() - t0:.0f}s)", flush=True)
    print(f"✓ done: ok={n_ok} fail={n_fail} cached_total={len(list(cache.glob('*.pkl.gz')))}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
