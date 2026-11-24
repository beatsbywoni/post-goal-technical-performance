"""StatsBomb 경기 메타(홈/어웨이) 수집 → data/interim/football_venue.json  (Memmert 2025 사설: match location 맥락화).

python3 -m src.football.sb_venue      # 16개 대회-시즌 match list 만 요청(가벼움, ~1분)
출력: {match_id: {"home", "away", "competition", "season", "home_score", "away_score"}}  (상대 수준 층화용 스코어 포함)
"""
from __future__ import annotations

import json
from pathlib import Path

from statsbombpy import sb

from .sb_ingest import selected_comps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "interim" / "football_venue.json"


def main() -> int:
    out = {}
    for c in selected_comps():
        m = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
        for _, r in m.iterrows():
            out[str(r["match_id"])] = {"home": r["home_team"], "away": r["away_team"],
                                       "competition": c["competition"], "season": c["season"],
                                       "home_score": int(r["home_score"]), "away_score": int(r["away_score"])}
        print(f"▶ {c['competition']} {c['season']}: {len(m)} matches", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"✓ {OUT} — {len(out)} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
