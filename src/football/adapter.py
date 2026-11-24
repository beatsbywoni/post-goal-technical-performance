"""StatsBomb 이벤트 → 롱포맷 계약 변환 (§5.2–5.6). v2 (2026-10, IJPAS 1차 수정판).

도메인 지식(실점 사건 정의·윈도우·지표 매핑)은 *여기 + configs/football.yaml* 에만 산다.
출력은 src/core/contract.LongRecord 와 동일한 롱포맷 → 같은 src/core/ 코드가
무수정으로 처리한다(이식 주장, §5.6).

입력: statsbombpy `sb.events(match_id)` 형태의 DataFrame (flatten 컬럼).
핵심 컬럼(없으면 안전 처리): type, team, period, minute, second, pass_length, pass_outcome,
under_pressure, shot_statsbomb_xg, shot_outcome, foul_committed_card, bad_behaviour_card, location,
pass_end_location.

규약: baseline 윈도우명은 LoL 과 동일하게 "baseline"(스코어 상태 매칭 평시 분 표본),
사건후는 post5/post10/post15. → core.run_pipeline(domain="football") 로 그대로 통과.

v2 에서 바뀐 점 (v1 = 2026-09-15 IJPAS 제출본; 변경 사유는 docs/adapter_v2_changes.md):
  1. 경기 시계가 피리어드를 안다. v1 은 minute·second 만 썼기 때문에 전반 추가시간(45+)과 후반 시작(45:00)
     이 같은 시각으로 겹치고, 연장전도 같은 문제가 있었다. v2 는 (period, 피리어드 내 경과 시간)으로 정렬·집계하며,
     사건 창과 통제 창은 그 피리어드 끝에서 잘린다(최종 휘슬에서 잘리던 것과 같은 규칙). 하프타임을 건너뛰는 창은 없다.
  2. 승부차기(period 5)는 제외한다. v1 은 승부차기 킥을 득점·실점 사건으로 세었다(29경기 198개).
  3. 퇴장(Red Card·Second Yellow; Foul Committed 또는 Bad Behaviour)을 맥락에 기록한다:
     dismissal_in_window(창 안에 퇴장 발생), after_dismissal(창 시작 전에 이미 퇴장이 있어 수적 불균형 상태).
     제외는 분석 단계에서 한다(사전 계획: 퇴장이 포함된 창 제외; 민감도: 퇴장 이후의 모든 창 제외).
  4. 통제 창은 어느 팀의 골이든 그 뒤 15분과 겹치면 제외한다. v1 은 분석 대상 사건(득점 또는 실점 한쪽)의 사후
     창만 제외했으므로 Methods 의 "did not overlap any post-goal window" 와 맞지 않았다.
  5. 맥락에 period, half, t_start(피리어드 내 시작 초), elapsed(경기 누적 플레이 시간 초), window_dur(실제 창 길이 초),
     goal_index, event_elapsed(사건 골의 누적 플레이 시간)를 추가한다 → 시간 경과 분석·흐름도·포함/제외 비교용.
  6. 기술 지표 3 + 선택 지표 4 외에, 원 단위 모형(이항·포아송)용 카운트 지표와 전술 기술 지표(descriptor 클래스)를
     함께 낸다: sp_attempts, sp_completed, touches, miscontrols, def_actions, fouls, pass_volume, possession_share,
     long_pass_share, action_x_mean, def_action_x_mean. net z 계산에는 들어가지 않는다(direction=context).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# §5.2 윈도우 (초 단위 경계). pre 는 보조(baseline 은 평시 분 분포로 별도 구성).
POST_WINDOWS = {"post5": (0, 300), "post10": (300, 600), "post15": (600, 900)}
EXCLUDE_AFTER_GOAL = 900          # 통제 창 제외 범위: 어느 골이든 그 뒤 15분
TOUCH_TYPES = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Ball Receipt"}
DEF_ACTIONS = {"Pressure", "Duel", "Interception", "Block", "Clearance", "Foul Committed"}
DISMISSAL_CARDS = {"Red Card", "Second Yellow"}
PERIOD_START_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}   # StatsBomb 분 표기의 피리어드 시작 분
SHOOTOUT_PERIOD = 5

EXEC_METRICS = ["short_pass_acc", "miscontrol_rate", "foul_per_def_action"]
CHOICE_METRICS = ["shot_xg_mean", "shot_distance", "pass_progression", "pressure_freq"]
DESCRIPTOR_METRICS = ["sp_attempts", "sp_completed", "touches", "miscontrols", "def_actions", "fouls",
                      "pass_volume", "possession_share", "long_pass_share", "action_x_mean", "def_action_x_mean"]
ALL_METRICS = EXEC_METRICS + CHOICE_METRICS + DESCRIPTOR_METRICS


def _t(row) -> float:
    """경기 시계(초) = minute*60+second. v1 과 동일한 보조 함수(테스트·외부 호출 호환)."""
    return float(row.get("minute", 0)) * 60 + float(row.get("second", 0) or 0)


def _mband(t_sec: float) -> str:
    """경기 분 밴드(15분 단위)."""
    m = t_sec / 60
    for hi in (15, 30, 45, 60, 75):
        if m < hi:
            return f"{hi-15}-{hi}"
    return "75+"


def _mband_period(period: int, t_in: float) -> str:
    """피리어드를 아는 분 밴드: 전반 추가시간은 30-45 에, 후반은 45 부터, 연장은 75+ 에 둔다."""
    if period <= 1:
        return _mband(min(t_in, 45 * 60 - 1))
    if period == 2:
        return _mband(45 * 60 + t_in)
    return "75+"


def extract_goals(df: pd.DataFrame) -> list[dict]:
    """득점 이벤트 추출 → [{t, team}] (Shot/Goal + Own Goal For). 승부차기 킥은 제외.

    t 는 경기 시계(초; v1 과 호환). 피리어드 컬럼이 있으면 period 도 담는다.
    """
    goals = []
    has_p = "period" in df.columns
    for _, r in df.iterrows():
        typ = r.get("type")
        if has_p and int(r.get("period") or 1) == SHOOTOUT_PERIOD:
            continue
        if (typ == "Shot" and r.get("shot_outcome") == "Goal") or typ == "Own Goal For":
            goals.append({"t": _t(r), "team": r.get("team"), "period": int(r.get("period") or 1) if has_p else 1})
    return sorted(goals, key=lambda g: (g["period"], g["t"]))


def score_state(goals: list[dict], team: str, t: float) -> str:
    """team 관점 스코어라인 'gf-ga' (시각 t '이전' 골 기준; t 는 goals 와 같은 시간축)."""
    gf = sum(1 for g in goals if g["team"] == team and g["t"] < t)
    ga = sum(1 for g in goals if g["team"] != team and g["t"] < t)
    return f"{gf}-{ga}"


def score_transition(goals: list[dict], team: str, t: float) -> str:
    """실점 직전→직후 상태 전이 라벨 (§5.4 장치 ③)."""
    gf = sum(1 for g in goals if g["team"] == team and g["t"] < t)
    ga = sum(1 for g in goals if g["team"] != team and g["t"] < t)  # 이 실점 제외
    if gf == ga:
        return "draw_to_behind"
    if gf < ga:
        return "behind_to_gap"
    if gf == ga + 1:
        return "ahead_to_draw"
    return "ahead_reduced"


def concede_events(goals: list[dict], teams: list[str]) -> list[dict]:
    """각 팀의 실점 사건 = 상대 득점. [{event_id, team, t, score_transition, goal_index}]."""
    out = []
    for i, g in enumerate(goals):
        for team in teams:
            if team != g["team"]:
                out.append({"event_id": f"concede_{i}_{team}", "team": team, "t": g["t"], "goal_index": i,
                            "period": g.get("period", 1), "score_transition": score_transition(goals, team, g["t"])})
    return out


def score_transition_scorer(goals: list[dict], team: str, t: float) -> str:
    """득점 직전→직후 상태 전이(득점팀 관점)."""
    gf = sum(1 for g in goals if g["team"] == team and g["t"] < t)
    ga = sum(1 for g in goals if g["team"] != team and g["t"] < t)
    if gf == ga:
        return "draw_to_ahead"
    if gf < ga:
        return "behind_to_draw" if gf + 1 == ga else "behind_reduced"
    return "ahead_extended"


def score_events(goals: list[dict], teams: list[str]) -> list[dict]:
    """각 팀의 득점 사건. [{event_id, team, t, score_transition, goal_index}]."""
    out = []
    for i, g in enumerate(goals):
        if g["team"] in teams:
            out.append({"event_id": f"score_{i}_{g['team']}", "team": g["team"], "t": g["t"], "goal_index": i,
                        "period": g.get("period", 1), "score_transition": score_transition_scorer(goals, g["team"], g["t"])})
    return out


def extract_dismissals(df: pd.DataFrame) -> list[dict]:
    """퇴장 [{t, team, period}] — Foul Committed(foul_committed_card) 또는 Bad Behaviour(bad_behaviour_card) 의 Red Card·Second Yellow."""
    out = []
    has_p = "period" in df.columns
    for col in ("foul_committed_card", "bad_behaviour_card"):
        if col not in df.columns:
            continue
        sub = df[df[col].isin(DISMISSAL_CARDS)]
        for _, r in sub.iterrows():
            out.append({"t": _t(r), "team": r.get("team"), "period": int(r.get("period") or 1) if has_p else 1})
    return sorted(out, key=lambda d: (d["period"], d["t"]))


# ── 지표 계산 (지표 = configs/football.yaml metrics 키) ────────────
def _metric(sub: pd.DataFrame, metric: str, opp_passes: int | None = None) -> float:
    """팀의 이벤트 부분집합에서 단일 지표값. 표본/컬럼 없으면 NaN(누락에 견고)."""
    if sub.empty or "type" not in sub.columns:
        return math.nan
    typ = sub["type"]
    if metric in ("short_pass_acc", "sp_attempts", "sp_completed"):   # 실행: 압박 없는 단거리 패스
        p = sub[typ == "Pass"]
        if "pass_length" in p.columns:
            p = p[pd.to_numeric(p["pass_length"], errors="coerce") < 20]
        if "under_pressure" in p.columns:
            p = p[p["under_pressure"] != True]  # noqa: E712 (NaN/False 통과)
        if metric == "sp_attempts":
            return float(len(p))
        if p.empty:
            return math.nan if metric == "short_pass_acc" else 0.0
        complete = p["pass_outcome"].isna() if "pass_outcome" in p.columns else pd.Series([True] * len(p))
        return float(complete.sum()) if metric == "sp_completed" else float(complete.mean())
    if metric in ("miscontrol_rate", "touches", "miscontrols"):       # 실행: 터치당 실수
        touches = int(typ.isin(TOUCH_TYPES).sum())
        miss = int(typ.isin(["Miscontrol", "Dispossessed"]).sum())
        if metric == "touches":
            return float(touches)
        if metric == "miscontrols":
            return float(miss)
        return float(miss / touches) if touches else math.nan
    if metric in ("foul_per_def_action", "def_actions", "fouls"):     # 실행/경계
        defa = int(typ.isin(DEF_ACTIONS).sum())
        fouls = int((typ == "Foul Committed").sum())
        if metric == "def_actions":
            return float(defa)
        if metric == "fouls":
            return float(fouls)
        return float(fouls / defa) if defa else math.nan
    if metric == "shot_xg_mean":              # 선택: 슛당 xG
        if "shot_statsbomb_xg" not in sub.columns:
            return math.nan
        s = pd.to_numeric(sub[typ == "Shot"]["shot_statsbomb_xg"], errors="coerce").dropna()
        return float(s.mean()) if len(s) else math.nan
    if metric == "shot_distance":             # 선택: 슛 거리(골대 120,40 기준)
        if "_lx" not in sub.columns:
            return math.nan
        s = sub[typ == "Shot"]
        d = np.sqrt((120 - s["_lx"]) ** 2 + (40 - s["_ly"]) ** 2).dropna()
        return float(d.mean()) if len(d) else math.nan
    if metric == "pass_progression":          # 선택: 패스 전진성(end_x − start_x)
        if "_lx" not in sub.columns or "_ex" not in sub.columns:
            return math.nan
        p = sub[typ == "Pass"]
        adv = (p["_ex"] - p["_lx"]).dropna()
        return float(adv.mean()) if len(adv) else math.nan
    if metric == "pressure_freq":             # 선택: 윈도우당 pressure 빈도
        return float((typ == "Pressure").sum())
    if metric == "pass_volume":               # 기술(전술): 창당 패스 시도 수
        return float((typ == "Pass").sum())
    if metric == "possession_share":          # 기술(전술): 팀 패스 / (팀 패스 + 상대 패스)
        own = int((typ == "Pass").sum())
        if opp_passes is None or own + opp_passes == 0:
            return math.nan
        return float(own / (own + opp_passes))
    if metric == "long_pass_share":           # 기술(전술): 30야드 이상 패스 비중
        p = sub[typ == "Pass"]
        if p.empty or "pass_length" not in p.columns:
            return math.nan
        pl = pd.to_numeric(p["pass_length"], errors="coerce").dropna()
        return float((pl >= 30).mean()) if len(pl) else math.nan
    if metric == "action_x_mean":             # 기술(전술): 온볼 행동의 평균 x (필드 틸트 대리)
        if "_lx" not in sub.columns:
            return math.nan
        x = sub[typ.isin(["Pass", "Carry", "Shot", "Dribble"])]["_lx"].dropna()
        return float(x.mean()) if len(x) else math.nan
    if metric == "def_action_x_mean":         # 기술(전술): 수비 행동의 평균 x (수비 라인 높이)
        if "_lx" not in sub.columns:
            return math.nan
        x = sub[typ.isin(DEF_ACTIONS)]["_lx"].dropna()
        return float(x.mean()) if len(x) else math.nan
    return math.nan


def _period_axis(df: pd.DataFrame) -> pd.DataFrame:
    """피리어드 인지 시간축을 붙인다: _p(피리어드), _tin(피리어드 내 경과 초), _ta(분석 축 = _p*1e5 + _tin)."""
    df = df.copy()
    df["_t"] = df.apply(_t, axis=1) if len(df) else []
    if "period" in df.columns:
        df["_p"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    else:
        df["_p"] = 1
    df["_tin"] = df["_t"] - df["_p"].map(PERIOD_START_MIN).fillna(0) * 60
    df["_ta"] = df["_p"] * 1e5 + df["_tin"]
    return df


def events_to_long(
    df: pd.DataFrame,
    match_id: str,
    metrics: list[str] | None = None,
    minute_bin: int = 1,
    competition: str | None = None,
    event_mode: str = "concede",
) -> list[dict]:
    """한 경기 이벤트 → 롱포맷 레코드 리스트 (v2, 피리어드 인지).

    · baseline: post 와 동일한 5분 슬라이딩 윈도우(위약; 2분 스텝) — 스코어 상태 매칭. 피리어드 안에서만 슬라이딩.
      어느 팀의 골이든 그 뒤 15분과 겹치는 창은 제외.
    · post5/10/15: 각 사건 윈도우의 팀 이벤트 집계. 피리어드 끝에서 잘림(window_dur 에 실제 길이).
    metrics 미지정 시 ALL_METRICS. competition 은 context 에 기록(LOCO용).
    """
    from ..core.config import load_metric_meta
    metrics = metrics or ALL_METRICS
    mclass = {m: d["class"] for m, d in load_metric_meta("football").items()}

    if df.empty or "team" not in df.columns:
        return []
    if "period" in df.columns:
        df = df[pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int) != SHOOTOUT_PERIOD]
        if df.empty:
            return []
    teams = [t for t in df["team"].dropna().unique()]
    df = _period_axis(df)
    goals = extract_goals(df)
    # goals 의 t 를 분석 축으로 바꾼다 (피리어드 인지 정렬·상태 계산)
    for g in goals:
        g["t"] = g["period"] * 1e5 + (g["t"] - PERIOD_START_MIN.get(g["period"], 0) * 60)
    goals = sorted(goals, key=lambda g: g["t"])
    dismissals = extract_dismissals(df)
    for d in dismissals:
        d["t"] = d["period"] * 1e5 + (d["t"] - PERIOD_START_MIN.get(d["period"], 0) * 60)
    events = score_events(goals, teams) if event_mode == "score" else concede_events(goals, teams)
    # 위치 좌표를 경기당 1회만 파싱
    def _coord(col, idx):
        if col not in df.columns:
            return np.nan
        return df[col].apply(lambda l: l[idx] if isinstance(l, (list, tuple, np.ndarray)) and len(l) > idx else np.nan)
    df["_lx"] = _coord("location", 0)
    df["_ly"] = _coord("location", 1)
    df["_ex"] = _coord("pass_end_location", 0)
    # 피리어드 경계와 누적 플레이 시간 오프셋
    p_end = df.groupby("_p")["_tin"].max().to_dict()
    offs, acc = {}, 0.0
    for p in sorted(p_end):
        offs[p] = acc
        acc += float(p_end[p]) + 1.0
    n_goals = len(goals)

    def where(ta: float) -> tuple[int, float]:
        p = int(ta // 1e5)
        return p, ta - p * 1e5

    def elapsed(ta: float) -> float:
        p, tin = where(ta)
        return offs.get(p, 0.0) + tin

    def dismissal_flags(a: float, b: float) -> tuple[bool, bool]:
        in_w = any(a <= d["t"] < b for d in dismissals)
        before = any(d["t"] < a for d in dismissals)
        return in_w, before

    records: list[dict] = []

    def emit(unit, event_id, window, metric, value, ctx):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        records.append(dict(
            unit_id=unit, domain="football", event_id=event_id, window=window,
            metric=metric, value=float(value), metric_class=mclass[metric],
            context=ctx, schema_ver="v1",
        ))

    def window_records(team, unit, event_id, window, a, b, extra_ctx):
        """[a, b) 창(분석 축) 의 팀 지표 레코드. b 는 호출 전에 피리어드 끝으로 잘려 있어야 한다."""
        if b <= a:
            return
        team_ev = df[(df["team"] == team) & (df["_ta"] >= a) & (df["_ta"] < b)]
        opp_passes = int(((df["team"] != team) & (df["type"] == "Pass") & (df["_ta"] >= a) & (df["_ta"] < b)).sum())
        p, tin = where(a)
        in_w, before = dismissal_flags(a, b)
        ctx = {"score_state": score_state(goals, team, a + 1e-6 if window != "baseline" else a),
               "match_id": match_id, "competition": competition, "minute_band": _mband_period(p, tin),
               "period": int(p), "half": "H1" if p == 1 else ("H2" if p == 2 else "ET"),
               "t_start": round(float(tin), 1), "elapsed": round(float(elapsed(a)), 1), "window_dur": round(float(b - a), 1),
               "dismissal_in_window": bool(in_w), "after_dismissal": bool(before), "n_goals_match": n_goals}
        ctx.update(extra_ctx)
        for m in metrics:
            emit(unit, event_id, window, m, _metric(team_ev, m, opp_passes), ctx)

    # 사건 윈도우 집계 (창은 사건 골의 피리어드 끝에서 잘림)
    for ev in events:
        unit = f"{ev['team']}-{match_id}"
        pg = ev["period"]; end_p = pg * 1e5 + p_end[pg] + 1e-6
        for win, (lo, hi) in POST_WINDOWS.items():
            a, b = ev["t"] + lo, min(ev["t"] + hi, end_p)
            window_records(ev["team"], unit, ev["event_id"], win, a, b,
                           {"score_transition": ev["score_transition"], "goal_index": int(ev["goal_index"]),
                            "event_elapsed": round(float(elapsed(ev["t"])), 1)})

    # baseline: 피리어드 안에서 5분 창 2분 스텝; 어느 골이든 그 뒤 15분과 겹치면 제외
    wlen, step = 300, 120
    excl = [(g["t"], g["t"] + EXCLUDE_AFTER_GOAL) for g in goals]
    for team in teams:
        unit = f"{team}-{match_id}"
        for p in sorted(p_end):
            t0 = p * 1e5
            seg_max = p * 1e5 + p_end[p]
            while t0 + wlen <= seg_max:
                t1 = t0 + wlen
                if any(t0 < hi and t1 > lo for lo, hi in excl):
                    t0 += step
                    continue
                eid = f"bl_{team}_{int(round(elapsed(t0)))}"
                window_records(team, unit, eid, "baseline", t0, t1, {})
                t0 += step
    return records
