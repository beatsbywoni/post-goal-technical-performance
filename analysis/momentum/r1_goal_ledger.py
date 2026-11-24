#!/usr/bin/env python3
"""R1 · 골 원장(goal ledger): 모든 골의 실제 직전·직후 점수 상태, 다음 골까지의 시간, 퇴장·연장 정보.

  python3 -m analysis.momentum.r1_goal_ledger <events_dir_or_"sb"> [main|heldout]

사건 창의 "기준 골이 만든 상태"(state_after)와 참조 분석의 "직전 상태"(state_before)는 이 원장에서 가져오고,
창 시작 점수에서 역산하지 않는다. goal_index·시간축·정렬은 src/football/adapter_fast 와 동일하게 만들어
파생자료(context.goal_index)와 1:1 로 결합된다. 승부차기(period 5) 제외.

출력: data/interim/goal_ledger.parquet (main) 또는 data/interim/goal_ledger_heldout.parquet
열: match_id, competition, season, goal_index, team(득점/자책골 수혜 팀), opponent, period, t_axis, t_in_period, elapsed, minute,
    state_before, state_after (득점 팀 관점 gf-ga), next_goal_dt, prev_goal_dt (초; 같은 피리어드 안의 다음/직전 골까지, 없으면 inf),
    own_goal, penalty, dismissal_before(골 이전 퇴장 수), n_goals_match, extra_time(골이 연장전), period_end_dt(피리어드 종료까지 초)
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.football.adapter import DISMISSAL_CARDS, PERIOD_START_MIN, SHOOTOUT_PERIOD  # noqa: E402


def ledger_for_match(df: pd.DataFrame, match_id: str, competition: str, season: str) -> list[dict]:
    df = df.copy()
    df["_p"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int) if "period" in df.columns else 1
    df = df[df["_p"] != SHOOTOUT_PERIOD].copy()
    if df.empty:
        return []
    teams = [t for t in df["team"].dropna().unique()]
    df["_t"] = df["minute"].astype(float).to_numpy() * 60 + pd.to_numeric(df["second"], errors="coerce").fillna(0).astype(float).to_numpy()
    df["_tin"] = df["_t"] - df["_p"].map(PERIOD_START_MIN).fillna(0) * 60
    df["_ta"] = df["_p"] * 1e5 + df["_tin"]
    p_end = df.groupby("_p")["_tin"].max().to_dict(); offs, acc = {}, 0.0
    for p in sorted(p_end):
        offs[p] = acc; acc += float(p_end[p]) + 1.0
    typ = df["type"].to_numpy(dtype=object); tm = df["team"].to_numpy(dtype=object); ta = df["_ta"].to_numpy(float); pp = df["_p"].to_numpy()
    so = df["shot_outcome"].to_numpy(dtype=object) if "shot_outcome" in df.columns else np.full(len(df), None, dtype=object)
    st = df["shot_type"].to_numpy(dtype=object) if "shot_type" in df.columns else np.full(len(df), None, dtype=object)
    mins = df["minute"].to_numpy(float)
    goals = [{"t": float(ta[i]), "team": tm[i], "period": int(pp[i]), "own": typ[i] == "Own Goal For", "pen": st[i] == "Penalty", "minute": float(mins[i])}
             for i in range(len(df)) if (typ[i] == "Shot" and so[i] == "Goal") or typ[i] == "Own Goal For"]
    goals = sorted(goals, key=lambda g: (g["period"], g["t"]))          # adapter_fast 와 동일한 정렬
    dism = []
    for col in ("foul_committed_card", "bad_behaviour_card"):
        if col in df.columns:
            mask = df[col].isin(DISMISSAL_CARDS).to_numpy()
            dism += [float(ta[i]) for i in np.where(mask)[0]]
    rows = []
    el = [offs[g["period"]] + g["t"] - g["period"] * 1e5 for g in goals]          # 연속 시계(피리어드 간 휴식 제외)
    for i, g in enumerate(goals):
        team = g["team"]; opp = next((t for t in teams if t != team), None)
        gf_before = sum(1 for h in goals[:i] if h["team"] == team); ga_before = i - gf_before
        same_p = [h for h in goals if h["period"] == g["period"]]
        nxt = [h["t"] for h in goals[i + 1:] if h["period"] == g["period"]]
        prv = [h["t"] for h in goals[:i] if h["period"] == g["period"]]
        rows.append({"match_id": str(match_id), "competition": competition, "season": season, "goal_index": i, "team": team, "opponent": opp,
                     "period": g["period"], "t_axis": g["t"], "t_in_period": g["t"] - g["period"] * 1e5, "elapsed": round(offs[g["period"]] + g["t"] - g["period"] * 1e5, 1),
                     "minute": g["minute"], "state_before": f"{gf_before}-{ga_before}", "state_after": f"{gf_before + 1}-{ga_before}",
                     "next_goal_dt": (nxt[0] - g["t"]) if nxt else np.inf, "prev_goal_dt": (g["t"] - prv[-1]) if prv else np.inf,
                     "next_goal_dt_any": (el[i + 1] - el[i]) if i + 1 < len(goals) else np.inf,
                     "own_goal": bool(g["own"]), "penalty": bool(g["pen"]), "dismissal_before": int(sum(1 for d in dism if d < g["t"])),
                     "dismissal_within_15min": bool(any(g["t"] <= d < g["t"] + 900 for d in dism)),
                     "n_goals_match": len(goals), "extra_time": g["period"] >= 3, "period_end_dt": float(p_end[g["period"]] - (g["t"] - g["period"] * 1e5)),
                     "n_goals_same_period": len(same_p)})
    return rows


def main(events_dir: str, which: str = "main"):
    warnings.filterwarnings("ignore")
    if which == "main":
        ml = json.loads((ROOT / "data/interim/match_list.json").read_text()); out = ROOT / "data/interim/goal_ledger.parquet"
    else:
        cfg = yaml.safe_load(open(ROOT / "configs/sb_heldout.yaml", encoding="utf-8"))
        from statsbombpy import sb
        ml = []
        for c in cfg["competitions"]:
            m = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
            ml += [[c["competition"], c.get("season", str(c["season_id"])), int(mid)] for mid in m.match_id]
        out = ROOT / "data/interim/goal_ledger_heldout.parquet"
    if events_dir == "sb":
        from statsbombpy import sb
    from statsbombpy import sb
    rows = []
    for comp, season, mid in ml:
        p = Path(events_dir) / f"{mid}.pkl.gz"
        ev = pd.read_pickle(p) if events_dir != "sb" and p.exists() else sb.events(match_id=mid)
        rows += ledger_for_match(ev, str(mid), comp, str(season))
    led = pd.DataFrame(rows); led.to_parquet(out, index=False)
    print(f"✓ {out}: {len(led):,} goals in {led.match_id.nunique():,} matches; own {int(led.own_goal.sum())}, "
          f"next goal within 15 min {int((led.next_goal_dt < 900).sum())}, within 10 min {int((led.next_goal_dt < 600).sum())}, within 5 min {int((led.next_goal_dt < 300).sum())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "main")
