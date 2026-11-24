"""R1 · 골 이후 시간 경과에 따른 net z (R3 요청): 골 후 0,1,…,10 분에서 시작하는 5분 슬라이딩 창과 2.5분 비중첩 구간.
2차(R1b): 단일 골 노출(창 종료 전 다른 골 제외), 사건 셀 통제, 모든 1분 간격 창(s0..s10)에서 분석 가능한 공통 골 표본(common) 병행.

  python3 -m analysis.momentum.r1_timecourse <events_dir_or_"sb">   → docs/ijpas_r1_timecourse.json

표준화 기저선(같은 점수상태 통제 셀)과 통제 창 net z 는 v2 적재 데이터(data/features)에서 가져오고,
사건 측 창만 이벤트 자료에서 다시 집계한다(피리어드 안에서 자름; 퇴장 포함 창 제외; 승부차기 제외).
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.momentum.r1_lib import DIR, EXEC, MIN_N, Standardised, estimate, load, save_json  # noqa: E402
from src.football.adapter import DISMISSAL_CARDS, PERIOD_START_MIN, SHOOTOUT_PERIOD, score_state  # noqa: E402
from src.football.adapter_fast import _TeamArrays, _coords  # noqa: E402

OUT = ROOT / "docs/ijpas_r2_timecourse.json"
STARTS = list(range(0, 11))             # 5분 슬라이딩 창 시작(분)
BINS = [(0, 2.5), (2.5, 5), (5, 7.5), (7.5, 10), (10, 12.5), (12.5, 15)]


def event_windows(ev: pd.DataFrame, match_id: str, mode: str) -> pd.DataFrame:
    """한 경기의 사건(득점/실점) × 창 정의(start, dur) → 지표 원값 행."""
    if ev.empty or "team" not in ev.columns:
        return pd.DataFrame()
    df = ev.copy(); df["_p"] = pd.to_numeric(df.period, errors="coerce").fillna(1).astype(int) if "period" in df.columns else 1
    df = df[df._p != SHOOTOUT_PERIOD]
    teams = [t for t in df.team.dropna().unique()]
    df["_lx"], df["_ly"], df["_ex"] = _coords(df)
    df["_t"] = df.minute.astype(float) * 60 + pd.to_numeric(df.second, errors="coerce").fillna(0)
    df["_tin"] = df._t - df._p.map(PERIOD_START_MIN).fillna(0) * 60; df["_ta"] = df._p * 1e5 + df._tin
    p_end = df.groupby("_p")._tin.max().to_dict()
    typ = df.type.to_numpy(dtype=object); tm = df.team.to_numpy(dtype=object); ta = df._ta.to_numpy(float); pp = df._p.to_numpy()
    so = df.shot_outcome.to_numpy(dtype=object) if "shot_outcome" in df.columns else np.full(len(df), None, dtype=object)
    goals = sorted([{"t": float(ta[i]), "team": tm[i], "period": int(pp[i])} for i in range(len(df)) if (typ[i] == "Shot" and so[i] == "Goal") or typ[i] == "Own Goal For"], key=lambda g: (g["period"], g["t"]))
    dism = []
    for col in ("foul_committed_card", "bad_behaviour_card"):
        if col in df.columns:
            dism += [float(x) for x in ta[df[col].isin(DISMISSAL_CARDS).to_numpy()]]
    TA = {t: _TeamArrays(df[df.team == t], "_ta") for t in teams}
    rows = []
    for i, g in enumerate(goals):
        focal = [g["team"]] if mode == "score" else [t for t in teams if t != g["team"]]
        for team in focal:
            eid = f"{'score' if mode == 'score' else 'concede'}_{i}_{team}"; end_p = g["period"] * 1e5 + p_end[g["period"]] + 1e-6
            defs = [(f"s{s}", s * 60, s * 60 + 300) for s in STARTS] + [(f"b{a}-{b}", a * 60, b * 60) for a, b in BINS]
            nxt = [h["t"] for h in goals[i + 1:] if h["period"] == g["period"]]; next_dt = (nxt[0] - g["t"]) if nxt else np.inf
            for lab, lo, hi in defs:
                a, b = g["t"] + lo, min(g["t"] + hi, end_p)
                if b <= a or any(a <= d < b for d in dism):
                    continue
                if next_dt < hi:                      # 단일 골 노출: 기준 골과 창 종료 사이에 다른 골이 있으면 제외
                    continue
                i0, i1 = TA[team].rng(a, b); vals = TA[team].metrics(i0, i1, EXEC, None)
                if vals is None:
                    continue
                st = score_state(goals, team, a + 1e-6)
                for m in EXEC:
                    if vals[m] is not None and not np.isnan(vals[m]):
                        rows.append({"unit_id": f"{team}-{match_id}", "event_id": eid, "window": lab, "metric": m, "value": vals[m], "score_state": st, "common15": next_dt >= 900})
    return pd.DataFrame(rows)


def main(events_dir: str):
    ml = json.loads((ROOT / "data/interim/match_list.json").read_text()) if events_dir != "sb" else None
    if events_dir == "sb":
        import yaml
        from statsbombpy import sb
        warnings.filterwarnings("ignore")
        cfg = yaml.safe_load(open(ROOT / "configs/sb_competitions.yaml", encoding="utf-8"))
        ml = [(c["competition"], c["season"], int(mid)) for tier in cfg["tiers"].values() for c in tier for mid in sb.matches(competition_id=c["competition_id"], season_id=c["season_id"]).match_id]
        getter = lambda mid: sb.events(match_id=mid)
    else:
        getter = lambda mid: pd.read_pickle(Path(events_dir) / f"{mid}.pkl.gz")
    t0 = time.time(); res = {}
    frames = {"score": [], "concede": []}
    for k, (comp, season, mid) in enumerate(ml):
        ev = getter(mid)
        for mode in ("score", "concede"):
            frames[mode].append(event_windows(ev, str(mid), mode))
        if (k + 1) % 300 == 0:
            print(f"{k+1}/{len(ml)} {time.time()-t0:.0f}s", flush=True)
    for side in ("score", "concede"):
        E = pd.concat(frames[side], ignore_index=True)
        df = load(side); S = Standardised(df, "state"); pl = S.frame("control")
        cells = S.cells.rename(columns={"ctx": "score_state"})
        m = E.merge(cells, on=["unit_id", "metric", "score_state"], how="inner")
        m["z"] = m.metric.map(DIR) * (m.value - m.bm) / m.bs.replace(0, np.nan); m["zp"] = m.metric.map(DIR) * (m.value - m.bm) / m.metric.map(S.pooled)
        m = m[m.z.notna()]
        g = m.groupby(["event_id", "unit_id", "window"]).agg(z=("z", "mean"), zp=("zp", "mean"), score_state=("score_state", "first"), common15=("common15", "first")).reset_index()
        out = {}
        # common goal sample: events analysable (z defined, cell present) in every one-minute-step window s0..s10, so that the
        # time course is not driven by changing composition. (Under same-state event-cell controls the 'no further goal within
        # 15 min' flag is redundant: a state that ends within 15 min has no control cell unless it recurs, so it is not used.)
        s_labs = [f"s{s}" for s in STARTS]
        g["eu"] = g.event_id + "|" + g.unit_id          # event_id is unique only within a match; unit_id carries the match
        ids_by_w = {lab: set(g.loc[g.window == lab, "eu"]) for lab in s_labs}
        common_ids = set.intersection(*ids_by_w.values())
        for lab in s_labs + [f"b{a}-{b}" for a, b in BINS]:
            e = g[g.window == lab]
            cells_e = e[["unit_id", "score_state"]].drop_duplicates().rename(columns={"score_state": "ctx"}); pl_e = S.frame("control", cells=cells_e)
            out[lab] = {"unit": estimate(e.z.to_numpy(), pl_e.z.to_numpy(), n=1000), "pooled": estimate(e.zp.to_numpy(), pl_e.zp.to_numpy(), n=1000),
                        "n_common15_flag": int(e.common15.astype(bool).sum())}
            if lab in s_labs:
                ec = e[e.eu.isin(common_ids)]; cells_c = ec[["unit_id", "score_state"]].drop_duplicates().rename(columns={"score_state": "ctx"}); pl_c = S.frame("control", cells=cells_c)
                out[lab]["common"] = {"unit": estimate(ec.z.to_numpy(), pl_c.z.to_numpy(), n=1000), "pooled": estimate(ec.zp.to_numpy(), pl_c.zp.to_numpy(), n=1000)}
                print(side, lab, out[lab]["unit"]["d"], out[lab]["unit"]["ci95"], out[lab]["pooled"]["d"], "n", out[lab]["unit"]["n_event"], "common", out[lab]["common"]["unit"]["d"], out[lab]["common"]["unit"]["n_event"], flush=True)
            else:
                print(side, lab, out[lab]["unit"]["d"], out[lab]["unit"]["ci95"], out[lab]["pooled"]["d"], "n", out[lab]["unit"]["n_event"], flush=True)
        out["_common"] = {"n_goals": len(common_ids), "definition": "events analysable in every window s0..s10"}
        res[side] = out
    save_json(res, OUT); print("✓", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
