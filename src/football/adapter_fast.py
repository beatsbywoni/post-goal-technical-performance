"""src/football/adapter.events_to_long 의 벡터화 구현 (v2 와 레코드 단위로 동일한 출력; 약 40배 빠름).

재적재 전체(2,404경기 × 2모드)를 수 분에 끝내기 위한 것으로, 정확성의 기준은 adapter.events_to_long 이다.
tests/test_adapter_fast_equivalence.py 와 scripts 의 표본 검증이 두 구현의 출력이 레코드 단위로 같음을 확인한다.

variant:
  "v2"      = adapter.events_to_long 과 동일(기본). 통제 창은 어느 골이든 그 뒤 15분과 겹치면 제외.
  "v2_own"  = 통제 창 제외를 분석 대상 사건(득점 또는 실점 한쪽)의 사후 창에만 적용(v1 의 제외 규칙; 민감도).
  "v2_cont" = 피리어드 끝에서 창을 자르지 않고 누적 플레이 시간 축에서 연속으로 창을 잡음(하프타임을 건너뜀; 민감도).
  "v1"      = 2026-09-15 제출본 어댑터 재현(피리어드 무시, 승부차기 포함, 기술 지표 없음).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .adapter import (ALL_METRICS, DEF_ACTIONS, DISMISSAL_CARDS, EXCLUDE_AFTER_GOAL, PERIOD_START_MIN, POST_WINDOWS,
                      SHOOTOUT_PERIOD, TOUCH_TYPES, _mband, _mband_period, score_state, score_transition,
                      score_transition_scorer)

V1_METRICS = ["short_pass_acc", "miscontrol_rate", "foul_per_def_action", "shot_xg_mean", "shot_distance", "pass_progression", "pressure_freq"]


def _coords(df):
    def _c(col, idx):
        if col not in df.columns:
            return np.full(len(df), np.nan)
        return df[col].apply(lambda l: l[idx] if isinstance(l, (list, tuple, np.ndarray)) and len(l) > idx else np.nan).to_numpy(dtype=float)
    return _c("location", 0), _c("location", 1), _c("pass_end_location", 0)


class _TeamArrays:
    """팀 이벤트를 시간순(안정 정렬)으로 두고 누적합으로 창 지표를 O(1)에 계산."""

    def __init__(self, s: pd.DataFrame, tcol: str):
        order = np.argsort(s[tcol].to_numpy(dtype=float), kind="stable")
        s = s.iloc[order]
        self.t = s[tcol].to_numpy(dtype=float)
        typ = s["type"]
        is_pass = (typ == "Pass").to_numpy(dtype=bool)
        plen = pd.to_numeric(s["pass_length"], errors="coerce").to_numpy(dtype=float) if "pass_length" in s.columns else np.full(len(s), np.nan)
        elig = is_pass.copy()
        if "pass_length" in s.columns:
            elig &= (plen < 20)
        if "under_pressure" in s.columns:
            elig &= (s["under_pressure"] != True).to_numpy(dtype=bool)  # noqa: E712
        comp = elig & (s["pass_outcome"].isna().to_numpy(dtype=bool) if "pass_outcome" in s.columns else np.ones(len(s), bool))
        cs = lambda a: np.concatenate([[0], np.cumsum(a.astype(np.float64))])
        self.c_pass = cs(is_pass); self.c_elig = cs(elig); self.c_comp = cs(comp)
        self.c_touch = cs(typ.isin(TOUCH_TYPES).to_numpy()); self.c_miss = cs(typ.isin(["Miscontrol", "Dispossessed"]).to_numpy())
        self.c_defa = cs(typ.isin(DEF_ACTIONS).to_numpy()); self.c_foul = cs((typ == "Foul Committed").to_numpy()); self.c_press = cs((typ == "Pressure").to_numpy())
        has_len = is_pass & ~np.isnan(plen)
        self.c_len = cs(has_len); self.c_long = cs(has_len & (plen >= 30))
        lx, ly, ex = s["_lx"].to_numpy(dtype=float), s["_ly"].to_numpy(dtype=float), s["_ex"].to_numpy(dtype=float)
        onball = typ.isin(["Pass", "Carry", "Shot", "Dribble"]).to_numpy() & ~np.isnan(lx)
        self.c_onb = cs(onball); self.c_onb_x = cs(np.where(onball, lx, 0.0))
        defx = typ.isin(DEF_ACTIONS).to_numpy() & ~np.isnan(lx)
        self.c_defx = cs(defx); self.c_defx_x = cs(np.where(defx, lx, 0.0))
        self.is_shot = (typ == "Shot").to_numpy(dtype=bool); self.is_pass = is_pass
        self.has_xg = "shot_statsbomb_xg" in s.columns
        self.xg = pd.to_numeric(s["shot_statsbomb_xg"], errors="coerce").to_numpy(dtype=float) if self.has_xg else None
        self.lx, self.ly, self.ex = lx, ly, ex

    def rng(self, a, b):
        return int(np.searchsorted(self.t, a, "left")), int(np.searchsorted(self.t, b, "left"))

    def passes(self, a, b):
        i0, i1 = self.rng(a, b); return int(self.c_pass[i1] - self.c_pass[i0])

    def metrics(self, i0, i1, metrics, opp_passes):
        if i1 <= i0:
            return None
        out = {}
        for m in metrics:
            if m == "short_pass_acc":
                ne = self.c_elig[i1] - self.c_elig[i0]; out[m] = float((self.c_comp[i1] - self.c_comp[i0]) / ne) if ne else math.nan
            elif m == "sp_attempts": out[m] = float(self.c_elig[i1] - self.c_elig[i0])
            elif m == "sp_completed": out[m] = float(self.c_comp[i1] - self.c_comp[i0]) if (self.c_elig[i1] - self.c_elig[i0]) else 0.0
            elif m == "miscontrol_rate":
                nt = self.c_touch[i1] - self.c_touch[i0]; out[m] = float((self.c_miss[i1] - self.c_miss[i0]) / nt) if nt else math.nan
            elif m == "touches": out[m] = float(self.c_touch[i1] - self.c_touch[i0])
            elif m == "miscontrols": out[m] = float(self.c_miss[i1] - self.c_miss[i0])
            elif m == "foul_per_def_action":
                nd = self.c_defa[i1] - self.c_defa[i0]; out[m] = float((self.c_foul[i1] - self.c_foul[i0]) / nd) if nd else math.nan
            elif m == "def_actions": out[m] = float(self.c_defa[i1] - self.c_defa[i0])
            elif m == "fouls": out[m] = float(self.c_foul[i1] - self.c_foul[i0])
            elif m == "shot_xg_mean":
                if self.has_xg:
                    v = self.xg[i0:i1][self.is_shot[i0:i1]]; v = v[~np.isnan(v)]; out[m] = float(pd.Series(v).mean()) if len(v) else math.nan
                else: out[m] = math.nan
            elif m == "shot_distance":
                sh = self.is_shot[i0:i1]; d = np.sqrt((120 - self.lx[i0:i1][sh]) ** 2 + (40 - self.ly[i0:i1][sh]) ** 2); d = d[~np.isnan(d)]
                out[m] = float(pd.Series(d).mean()) if len(d) else math.nan
            elif m == "pass_progression":
                ps = self.is_pass[i0:i1]; adv = self.ex[i0:i1][ps] - self.lx[i0:i1][ps]; adv = adv[~np.isnan(adv)]
                out[m] = float(pd.Series(adv).mean()) if len(adv) else math.nan
            elif m == "pressure_freq": out[m] = float(self.c_press[i1] - self.c_press[i0])
            elif m == "pass_volume": out[m] = float(self.c_pass[i1] - self.c_pass[i0])
            elif m == "possession_share":
                own = self.c_pass[i1] - self.c_pass[i0]; out[m] = float(own / (own + opp_passes)) if (opp_passes is not None and own + opp_passes > 0) else math.nan
            elif m == "long_pass_share":
                nl = self.c_len[i1] - self.c_len[i0]; out[m] = float((self.c_long[i1] - self.c_long[i0]) / nl) if nl else math.nan
            elif m == "action_x_mean":
                n = self.c_onb[i1] - self.c_onb[i0]; out[m] = float((self.c_onb_x[i1] - self.c_onb_x[i0]) / n) if n else math.nan
            elif m == "def_action_x_mean":
                n = self.c_defx[i1] - self.c_defx[i0]; out[m] = float((self.c_defx_x[i1] - self.c_defx_x[i0]) / n) if n else math.nan
            else: out[m] = math.nan
        return out


def events_to_long_fast(df: pd.DataFrame, match_id: str, competition: str | None = None,
                        event_mode: str = "concede", variant: str = "v2", restart_exclusion: str | None = None) -> list[dict]:
    """restart_exclusion (R2 민감도; 기본 None 이면 출력 불변):
      "kickoff_possession" = StatsBomb play_pattern == "From Kick Off" 인 이벤트(킥오프로 시작한 소유권 전체, 양 팀)를
                             지표 계산에서 제외한다. 골·퇴장·피리어드 경계·창 격자는 전체 이벤트로 그대로 정한다.
      "first60s"           = 어느 골이든 그 뒤 60초 이내와 각 피리어드 첫 60초의 이벤트를 지표 계산에서 제외한다(창 격자 불변).
    """
    from ..core.config import load_metric_meta
    mclass = {m: d["class"] for m, d in load_metric_meta("football").items()}
    if df.empty or "team" not in df.columns:
        return []
    v1 = variant == "v1"
    metrics = V1_METRICS if v1 else ALL_METRICS
    df = df.copy()
    if "period" in df.columns:
        df["_p"] = pd.to_numeric(df["period"], errors="coerce").fillna(1).astype(int)
    else:
        df["_p"] = 1
    if not v1:
        df = df[df["_p"] != SHOOTOUT_PERIOD].copy()
        if df.empty:
            return []
    teams = [t for t in df["team"].dropna().unique()]
    df["_lx"], df["_ly"], df["_ex"] = _coords(df)
    df["_t"] = df["minute"].astype(float).to_numpy() * 60 + pd.to_numeric(df["second"], errors="coerce").fillna(0).astype(float).to_numpy()
    df["_tin"] = df["_t"] - df["_p"].map(PERIOD_START_MIN).fillna(0) * 60
    p_end = df.groupby("_p")["_tin"].max().to_dict(); offs, acc = {}, 0.0
    for p in sorted(p_end):
        offs[p] = acc; acc += float(p_end[p]) + 1.0
    if v1:
        df["_ta"] = df["_t"]
    elif variant == "v2_cont":
        df["_ta"] = df["_tin"] + df["_p"].map(offs)
    else:
        df["_ta"] = df["_p"] * 1e5 + df["_tin"]
    tcol = "_ta"

    def where(ta):
        if v1:
            return 1, ta
        if variant == "v2_cont":
            p = max((q for q in offs if offs[q] <= ta + 1e-9), default=1); return p, ta - offs[p]
        p = int(ta // 1e5); return p, ta - p * 1e5

    def elapsed(ta):
        p, tin = where(ta); return offs.get(p, 0.0) + tin

    # 골·퇴장 (프레임 순서로 모아 시간축 안정 정렬 — adapter 와 동일)
    typ = df["type"].to_numpy(dtype=object); tm = df["team"].to_numpy(dtype=object); ta_arr = df[tcol].to_numpy(dtype=float); pp = df["_p"].to_numpy()
    so = df["shot_outcome"].to_numpy(dtype=object) if "shot_outcome" in df.columns else np.full(len(df), None, dtype=object)
    goals = [{"t": float(ta_arr[i]), "team": tm[i], "period": int(pp[i])} for i in range(len(df)) if (typ[i] == "Shot" and so[i] == "Goal") or typ[i] == "Own Goal For"]
    goals = sorted(goals, key=lambda g: g["t"]) if v1 else sorted(goals, key=lambda g: (g["period"], g["t"]))
    dism = []
    if not v1:
        for col in ("foul_committed_card", "bad_behaviour_card"):
            if col in df.columns:
                mask = df[col].isin(DISMISSAL_CARDS).to_numpy()
                dism += [{"t": float(ta_arr[i]), "team": tm[i]} for i in np.where(mask)[0]]
        dism.sort(key=lambda d: d["t"])
    if event_mode == "score":
        events = [{"event_id": f"score_{i}_{g['team']}", "team": g["team"], "t": g["t"], "goal_index": i, "period": g["period"],
                   "score_transition": score_transition_scorer(goals, g["team"], g["t"])} for i, g in enumerate(goals) if g["team"] in teams]
    else:
        events = [{"event_id": f"concede_{i}_{team}", "team": team, "t": g["t"], "goal_index": i, "period": g["period"],
                   "score_transition": score_transition(goals, team, g["t"])} for i, g in enumerate(goals) for team in teams if team != g["team"]]
    if restart_exclusion is None:
        keep = np.ones(len(df), dtype=bool)
    elif restart_exclusion == "kickoff_possession":
        keep = ~(df["play_pattern"] == "From Kick Off").to_numpy(dtype=bool) if "play_pattern" in df.columns else np.ones(len(df), dtype=bool)
    elif restart_exclusion == "first60s":
        tin = df["_tin"].to_numpy(dtype=float)
        keep = tin >= 60.0
        for g in goals:
            keep &= ~((ta_arr >= g["t"]) & (ta_arr < g["t"] + 60.0))
    else:
        raise ValueError(restart_exclusion)
    dfm = df[keep]
    TA = {team: _TeamArrays(dfm[dfm["team"] == team], tcol) for team in teams}
    n_goals = len(goals)
    records = []

    def emit(unit, eid, window, vals, ctx):
        if vals is None:
            return
        for m in metrics:
            v = vals[m]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            records.append(dict(unit_id=unit, domain="football", event_id=eid, window=window, metric=m, value=float(v),
                                metric_class=mclass[m], context=ctx, schema_ver="v1"))

    def ctx_for(team, window, a, b, extra):
        if v1:
            return {"score_state": score_state(goals, team, a + 1e-6 if window != "baseline" else a), **extra,
                    "match_id": match_id, "competition": competition, "minute_band": _mband(a)}
        p, tin = where(a)
        in_w = any(a <= d["t"] < b for d in dism); before = any(d["t"] < a for d in dism)
        ctx = {"score_state": score_state(goals, team, a + 1e-6 if window != "baseline" else a),
               "match_id": match_id, "competition": competition, "minute_band": _mband_period(p, tin),
               "period": int(p), "half": "H1" if p == 1 else ("H2" if p == 2 else "ET"),
               "t_start": round(float(tin), 1), "elapsed": round(float(elapsed(a)), 1), "window_dur": round(float(b - a), 1),
               "dismissal_in_window": bool(in_w), "after_dismissal": bool(before), "n_goals_match": n_goals}
        ctx.update(extra)
        return ctx

    def window_records(team, unit, eid, window, a, b, extra):
        if b <= a:
            return
        ta_ = TA[team]; i0, i1 = ta_.rng(a, b)
        opp = sum(TA[o].passes(a, b) for o in teams if o != team) if not v1 else None
        emit(unit, eid, window, ta_.metrics(i0, i1, metrics, opp), ctx_for(team, window, a, b, extra))

    post_spans = []
    for ev in events:
        unit = f"{ev['team']}-{match_id}"
        for win, (lo, hi) in POST_WINDOWS.items():
            a, b = ev["t"] + lo, ev["t"] + hi
            if not v1 and variant != "v2_cont":
                b = min(b, ev["period"] * 1e5 + p_end[ev["period"]] + 1e-6)
            extra = {"score_transition": ev["score_transition"]} if v1 else {"score_transition": ev["score_transition"], "goal_index": int(ev["goal_index"]), "event_elapsed": round(float(elapsed(ev["t"])), 1)}
            window_records(ev["team"], unit, ev["event_id"], win, a, b, extra)
            post_spans.append((ev["team"], ev["t"], ev["t"] + hi))
    wlen, step = 300, 120
    if v1:
        segments = [(0.0, float(int(df["_t"].max())))]
    elif variant == "v2_cont":
        segments = [(0.0, float(df[tcol].max()))]
    else:
        segments = [(p * 1e5, p * 1e5 + p_end[p]) for p in sorted(p_end)]
    for team in teams:
        unit = f"{team}-{match_id}"
        if v1 or variant == "v2_own":
            excl = [(lo, hi) for t_, lo, hi in post_spans if t_ == team]
        else:
            excl = [(g["t"], g["t"] + EXCLUDE_AFTER_GOAL) for g in goals]
        for seg_start, seg_max in segments:
            t0 = seg_start
            while t0 + wlen <= seg_max:
                t1 = t0 + wlen
                if any(t0 < hi and t1 > lo for lo, hi in excl):
                    t0 += step; continue
                eid = f"bl_{team}_{t0}" if v1 else f"bl_{team}_{int(round(elapsed(t0)))}"
                if v1:
                    ta_ = TA[team]; i0, i1 = ta_.rng(t0, t1)
                    if i1 <= i0:
                        t0 += step; continue
                    emit(unit, eid, "baseline", ta_.metrics(i0, i1, metrics, None), ctx_for(team, "baseline", t0, t1, {}))
                else:
                    window_records(team, unit, eid, "baseline", t0, t1, {})
                t0 += step
    return records
