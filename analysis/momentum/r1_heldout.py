"""R1 · 표본 밖 재현(held-out replication): 2026-06-04 잠금 때 선택하지 않은 StatsBomb 오픈데이터의 완전한 대회.

  python3 -m analysis.momentum.r1_heldout ingest [cache_dir]   # 이벤트 적재 → data/features/heldout_{score,concede}.parquet
  python3 -m analysis.momentum.r1_heldout analyse              # 앵커 창 참조/주/풀링 추정 → docs/ijpas_r1_heldout.json

대회(configs/sb_heldout.yaml): Liga F 2023/24, Frauen-Bundesliga 2023/24, Serie A Women 2023/24, NWSL 2023, Indian Super League 2021/22,
UEFA Women's Euro 2022·2025. 단일 팀 중심의 부분 덤프(Bundesliga 2023/24 레버쿠젠, PL 2003/04 아스널 등)는 제외.
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.momentum.r1_lib import ANCHOR, EXEC, WINDOWS, Standardised, estimate, load, save_json  # noqa: E402
from src.football.adapter_fast import events_to_long_fast  # noqa: E402

CFG = ROOT / "configs/sb_heldout.yaml"
FEAT = {"score": ROOT / "data/features/heldout_football_momentum_long.parquet", "concede": ROOT / "data/features/heldout_football_long.parquet"}
OUT = ROOT / "docs/ijpas_r2_heldout.json"
COLS = ["unit_id", "domain", "event_id", "window", "metric", "value", "metric_class", "context", "schema_ver"]


def ingest(cache_dir: str | None):
    from statsbombpy import sb
    warnings.filterwarnings("ignore")
    comps = yaml.safe_load(CFG.read_text(encoding="utf-8"))["competitions"]
    recs = {"score": [], "concede": []}; t0 = time.time(); n = 0
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for c in comps:
        m = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
        print(f"▶ {c['competition']} {c['season']}: {len(m)} matches", flush=True)
        for mid in m.match_id.tolist():
            cp = Path(cache_dir) / f"{mid}.pkl.gz" if cache_dir else None
            if cp is not None and cp.exists():
                ev = pd.read_pickle(cp)
            else:
                for attempt in range(4):
                    try:
                        ev = sb.events(match_id=mid); break
                    except Exception:  # noqa: BLE001
                        time.sleep(2 + 3 * attempt)
                else:
                    print("  ! failed", mid); continue
                if cp is not None:
                    ev.to_pickle(cp, compression="gzip")
            for mode in ("score", "concede"):
                recs[mode] += events_to_long_fast(ev, match_id=str(mid), competition=f"{c['competition']} {c['season']}", event_mode=mode, variant="v2")
            n += 1
            if n % 100 == 0:
                print(f"  {n} matches {time.time()-t0:.0f}s", flush=True)
    for mode in ("score", "concede"):
        pd.DataFrame(recs[mode], columns=COLS).to_parquet(FEAT[mode], index=False); print("✓", FEAT[mode], len(recs[mode]))


def analyse():
    res = {}
    for side in ("score", "concede"):
        df = _load_heldout(side)
        res[side] = {}
        groups = {"all_heldout": sorted(df.competition.unique())}
        groups.update({c: [c] for c in sorted(df.competition.unique())})
        masks = {f"stage_{st}": (df.stage == st) for st in ("league", "league_phase2", "group", "knockout")}
        for name, comps in list(groups.items()) + list(masks.items()):
            sub = df[df.competition.isin(comps)] if not isinstance(comps, pd.Series) else df[comps]
            r = {}
            for key in ("pre", "state"):
                S = Standardised(sub, key); ev = S.frame("event", ANCHOR)
                pl = S.frame("control", cells=ev.reset_index()[["unit_id", "ctx"]].drop_duplicates())
                r[key] = {"unit": estimate(ev.z.to_numpy(), pl.z.to_numpy(), n=2000), "pooled": estimate(ev.zp.to_numpy(), pl.zp.to_numpy(), n=2000)} if len(ev) >= 20 else {"unit": {"d": float("nan"), "n_event": int(len(ev))}, "pooled": {"d": float("nan")}}
            r["matches"] = int(sub.match_id.nunique()); r["units"] = int(sub.unit_id.nunique())
            res[side][name] = r
            print(side, name, "matches", r["matches"], "ref", r["pre"]["unit"]["d"], "prim", r["state"]["unit"]["d"], r["state"]["unit"].get("ci95"), "pooled", r["state"]["pooled"]["d"], "n", r["state"]["unit"].get("n_event"), flush=True)
    save_json(res, OUT); print("✓", OUT)


def _load_heldout(side):
    """r1_lib.load 와 같은 전처리를 held-out parquet 에 적용."""
    return load(side, features_dir=FEAT[side].parent, exclusion="dismissal_window", ledger=ROOT / "data/interim/goal_ledger_heldout.parquet", files={side: FEAT[side].name})


if __name__ == "__main__":
    if sys.argv[1] == "ingest":
        ingest(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        analyse()
