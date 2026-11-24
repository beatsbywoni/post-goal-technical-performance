#!/usr/bin/env python3
"""R1 · 골·퇴장·연장·승부차기 집계 (Figure 1 의 출발 상자, Table 1 의 골 수) → docs/ijpas_r1_goals.json

  python3 -m analysis.momentum.r1_goals <events_dir_or_"sb">

events_dir: 경기별 statsbombpy 이벤트 프레임을 pickle(gzip) 로 저장한 폴더(<match_id>.pkl.gz); "sb" 면 statsbombpy 로 받는다.
집계는 src/football/adapter 의 extract_goals (승부차기 제외) 와 extract_dismissals 를 그대로 쓴다.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.football.adapter import SHOOTOUT_PERIOD, extract_dismissals, extract_goals  # noqa: E402

OUT = ROOT / "docs/ijpas_r1_goals.json"


def main(events_dir: str):
    warnings.filterwarnings("ignore")
    ml = json.loads((ROOT / "data/interim/match_list.json").read_text())
    tot = {"goals": 0, "own_goals": 0, "pen_goals": 0, "goals_p1_stoppage": 0, "goals_ET": 0, "dismissals": 0, "matches_with_dismissal": 0, "matches_ET": 0,
           "shootout_matches": 0, "shootout_kicks": 0, "shootout_converted": 0}
    per = {}
    if events_dir == "sb":
        from statsbombpy import sb
    for comp, season, mid in ml:
        ev = pd.read_pickle(Path(events_dir) / f"{mid}.pkl.gz") if events_dir != "sb" else sb.events(match_id=mid)
        goals = extract_goals(ev); dis = extract_dismissals(ev)
        p = per.setdefault(comp, {"goals": 0, "matches": 0}); p["goals"] += len(goals); p["matches"] += 1
        tot["goals"] += len(goals); tot["dismissals"] += len(dis); tot["matches_with_dismissal"] += bool(dis)
        og = ((ev.type == "Own Goal For")).sum(); tot["own_goals"] += int(og)
        shots = ev[(ev.type == "Shot") & (ev.shot_outcome == "Goal")]
        if "shot_type" in ev.columns:
            tot["pen_goals"] += int(((shots.shot_type == "Penalty") & (shots.period != SHOOTOUT_PERIOD)).sum())
        tot["goals_p1_stoppage"] += sum(1 for g in goals if g["period"] == 1 and g["t"] >= 45 * 60)
        tot["goals_ET"] += sum(1 for g in goals if g["period"] in (3, 4))
        if (ev.period >= 3).any():
            tot["matches_ET"] += 1
        so = ev[ev.period == SHOOTOUT_PERIOD]
        if len(so):
            tot["shootout_matches"] += 1; k = so[so.type == "Shot"]; tot["shootout_kicks"] += int(len(k)); tot["shootout_converted"] += int((k.shot_outcome == "Goal").sum())
    out = {"totals": tot, "per_comp": per}
    OUT.write_text(json.dumps(out, indent=1)); print(json.dumps(tot)); print("✓", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
