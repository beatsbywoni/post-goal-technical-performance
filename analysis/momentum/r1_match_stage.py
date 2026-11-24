#!/usr/bin/env python3
"""R1 · 경기 단계(competition_stage) 메타데이터: StatsBomb matches 의 competition_stage 를 받아 data/interim/match_stage.json 에 저장.
  python3 -m analysis.momentum.r1_match_stage [main|heldout]
리그 = 'Regular Season'; 국제대회는 'Group Stage' 와 그 밖의 녹아웃 단계('Round of 16', 'Quarter-finals', 'Semi-finals', 'Final', '3rd Place Final' 등).
"""
import json, sys, warnings
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore")
from statsbombpy import sb
which = sys.argv[1] if len(sys.argv) > 1 else "main"
cfg = yaml.safe_load(open(ROOT / ("configs/sb_competitions.yaml" if which == "main" else "configs/sb_heldout.yaml"), encoding="utf-8"))
comps = [c for tier in cfg["tiers"].values() for c in tier] if which == "main" else cfg["competitions"]
out = {}
for c in comps:
    m = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
    for r in m.itertuples(index=False):
        out[str(r.match_id)] = {"competition": c["competition"], "season": str(c.get("season", "")), "stage": str(r.competition_stage), "home": r.home_team, "away": r.away_team,
                                "home_score": int(r.home_score), "away_score": int(r.away_score), "match_week": int(r.match_week) if r.match_week == r.match_week else None}
    print(c["competition"], c.get("season"), len(m), sorted(set(m.competition_stage)), flush=True)
p = ROOT / ("data/interim/match_stage.json" if which == "main" else "data/interim/match_stage_heldout.json"); p.write_text(json.dumps(out, indent=0)); print("✓", p, len(out))
