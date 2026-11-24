"""IJPAS 1차 수정(R1) 공용 함수 — 적재(v2 parquet), 제외 규칙, 표준화(net z), d·부트스트랩·순열.

규약(원고 Methods 와 1:1):
  · 통제 셀 = (unit, metric, ctx) 의 통제 창 ≥ MIN_N(3) 일 때만 기저선 추정(평균·SD).
  · z = d_k (x − b_ik) / s_ik (유닛 분모) 또는 / pooled_k (지표별 유닛 SD 의 중앙값; 풀링 분모).
  · net z = 창 안의 표준화된 지표 z 의 평균(K ≤ 3; K=3 제한은 need_k).
  · 윈저화: 사건 창 net z 와 통제 창 net z 를 **각각** 1/99 백분위에서 자른다(사전 계획 "post-event values winsorised";
    제출본과 동일 규약). 'joint' = 합동 백분위(민감도).
  · d = (mean_event − mean_control) / 합동 SD(두 집단 분산의 자유도 가중 평균의 제곱근).
  · 부트스트랩 CI: 창 단위 재표집(기본) 또는 클러스터(유닛/경기/팀) 재표집.
  · 순열 귀무: 유닛 안에서 사건/통제 라벨을 뒤섞음(within-unit).
R1 (2026-10, 2차 수정) 추가 규약:
  · 골 원장(data/interim/goal_ledger.parquet)으로 사건의 직전·직후 상태를 지정한다(창 시작 점수에서 역산하지 않음).
  · 단일 골 노출(exposure="single_goal", 기본): 기준 골과 창 종료 사이에 어느 팀이든 다른 골이 있으면 그 사건 창을 제외한다.
    exposure="all" 은 R1 1차 규약(창 시작 점수 기준, 중간 골 유지)의 민감도용.
  · 주 추정의 통제 = 사건이 있는 셀(unit × state)의 통제 창(control_scope="event_cells", 기본); 적격 셀 전체는 "all_cells"(민감도).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXEC = ["short_pass_acc", "miscontrol_rate", "foul_per_def_action"]
DIR = {"short_pass_acc": 1.0, "miscontrol_rate": -1.0, "foul_per_def_action": -1.0}
FILES = {"score": "football_momentum_long.parquet", "concede": "football_long.parquet"}
MIN_N = 3
WINDOWS = ["post5", "post10", "post15"]
ANCHOR = "post10"
LEAGUE_M = {"La Liga", "Premier League", "Serie A", "Ligue 1"}
TOURN_M = {"FIFA World Cup", "UEFA Euro", "Copa America", "African Cup of Nations"}
LEAGUE_W = {"FA Women's Super League"}
TOURN_W = {"Women's World Cup"}
STRATA = {"men_league": LEAGUE_M, "men_tournament": TOURN_M, "women_league": LEAGUE_W, "women_tournament": TOURN_W}
COMP_ORDER = ["La Liga", "Premier League", "Serie A", "Ligue 1", "FIFA World Cup", "UEFA Euro", "Copa America",
              "African Cup of Nations", "FA Women's Super League", "Women's World Cup"]


def pre_state(st: str, side: str) -> str:
    gf, ga = map(int, st.split("-"))
    return f"{max(gf-1,0)}-{ga}" if side == "score" else f"{gf}-{max(ga-1,0)}"


HI = {"post5": 300.0, "post10": 600.0, "post15": 900.0}
# StatsBomb competition_stage → analysis stage. Explicit mapping: every stage name in the main and held-out samples is listed, and an
# unlisted name raises. "Championship Round" and "Relegation Play-offs" (Serie A Women 2023/24) are second-phase round-robin leagues
# (five teams, home and away; FIGC 2023/24 regulations), not knockout ties, and form their own category.
STAGE_MAP = {"Regular Season": "league",
             "Championship Round": "league_phase2", "Relegation Play-offs": "league_phase2",
             "Group Stage": "group",
             "Round of 16": "knockout", "Quarter-finals": "knockout", "Semi-finals": "knockout", "3rd Place Final": "knockout", "Final": "knockout",
             "1st Round ": "knockout", "1st Round": "knockout", "Play-offs - Semi-Finals": "knockout", "Championship - Final": "knockout"}


def _swap(st: str) -> str:
    a, b = st.split("-"); return f"{b}-{a}"


def load(side: str, features_dir: Path | None = None, exclusion: str = "dismissal_window", exposure: str = "single_goal",
         ledger: Path | None = None, files: dict | None = None, continuous_clock: bool = False) -> pd.DataFrame:
    """v2 롱포맷(실행 지표 3개)을 읽어 평탄한 열로 바꾼다.

    exclusion: "none" | "dismissal_window"(사전 계획: 퇴장이 포함된 창 제외) | "after_dismissal"(퇴장 이후의 모든 창도 제외).
    exposure:  "single_goal"(기본; 기준 골과 창 종료 사이에 다른 골이 있는 사건 창 제외) | "all"(유지; 민감도).
    골 원장을 결합해 사건 창의 score_state 를 원장의 state_after(팀 관점)로, state_pre 를 원장의 state_before 로 지정한다.
    단일 골 노출 창에서는 창 시작 점수와 state_after 가 항상 같다(검증됨); exposure="all" 에서는 중간 골이 있는 창에 한해
    창 시작 점수(R1 1차 규약)를 유지하고 other_goal_before_end 로 표시한다.
    """
    fd = features_dir or (ROOT / "data/features")
    df = pd.read_parquet(fd / (files or FILES)[side])
    df = df[df.metric.isin(EXEC)].copy()
    ctx = pd.DataFrame(df.context.tolist(), index=df.index)
    for c in ("score_state", "competition", "minute_band", "half", "period", "t_start", "elapsed", "window_dur",
              "dismissal_in_window", "after_dismissal", "n_goals_match", "goal_index", "event_elapsed", "score_transition"):
        if c in ctx.columns:
            df[c] = ctx[c].values
    df = df.drop(columns=["context", "domain", "schema_ver"])   # 메모리: dict 열 제거
    df["match_id"] = df.unit_id.str.rsplit("-", n=1).str[-1]
    df["team"] = df.unit_id.str.rsplit("-", n=1).str[0]
    df["is_post"] = df.window.str.startswith("post")
    # 골 원장 결합(사건 창만): 직전·직후 상태, 다음 골까지의 시간
    led = pd.read_parquet(ledger or (ROOT / "data/interim/goal_ledger.parquet"))
    if continuous_clock and "next_goal_dt_any" in led.columns:                 # v2_cont 변형: 창이 피리어드를 넘으므로 다음 골을 연속 시계로 잰다
        led = led.assign(next_goal_dt=led.next_goal_dt_any)
    led = led[["match_id", "goal_index", "state_before", "state_after", "next_goal_dt", "minute", "own_goal", "penalty"]].rename(columns={"minute": "goal_minute"})
    df["goal_index"] = pd.to_numeric(df.get("goal_index"), errors="coerce")
    post = df.is_post.to_numpy()
    m = df.loc[post, ["match_id", "goal_index"]].astype({"goal_index": int}).merge(led.astype({"goal_index": int}), on=["match_id", "goal_index"], how="left")
    assert m.state_after.notna().all(), "ledger: unmatched post-goal windows"
    flip = side == "concede"
    s_after = m.state_after.map(_swap) if flip else m.state_after
    s_before = m.state_before.map(_swap) if flip else m.state_before
    hi = df.loc[post, "window"].map(HI).to_numpy()
    other = (m.next_goal_dt.to_numpy() < hi)
    df["other_goal_before_end"] = False; df.loc[post, "other_goal_before_end"] = other
    df["next_goal_dt"] = np.inf; df.loc[post, "next_goal_dt"] = m.next_goal_dt.to_numpy()
    df["goal_minute"] = np.nan; df.loc[post, "goal_minute"] = m.goal_minute.to_numpy()
    df["state_pre"] = df.score_state
    df.loc[post, "state_pre"] = s_before.to_numpy()
    # 단일 골 창에서는 창 시작 점수 == state_after (검증); 중간 골 창은 exposure 규칙에 따름
    agree = (df.loc[post, "score_state"].to_numpy() == s_after.to_numpy())
    assert bool(np.all(agree[~other])), "ledger: window-start state differs from state_after in a single-goal window"
    if exposure == "single_goal":
        df = df[~df.other_goal_before_end]
    elif exposure != "all":
        raise ValueError(exposure)
    if "half" not in df.columns:
        df["half"] = "?"
    df["format"] = np.where(df.competition.isin(LEAGUE_M | LEAGUE_W), "league", "tournament")
    # 경기 단계(StatsBomb competition_stage): league(Regular Season) / group(Group Stage) / knockout(그 밖의 단계)
    stage_path = ROOT / "data/interim" / ("match_stage_heldout.json" if "heldout" in (files or FILES)[side] else "match_stage.json")
    if stage_path.exists():
        st = json.loads(stage_path.read_text())
        def _stage(mid):
            v = st.get(str(mid)); s_ = v["stage"] if v else ""
            if not s_: return "unknown"
            if s_ not in STAGE_MAP: raise KeyError(f"unmapped competition_stage: {s_!r} (match {mid}); add it to STAGE_MAP")
            return STAGE_MAP[s_]
        df["stage"] = df.match_id.map(_stage)
    else:
        df["stage"] = "unknown"
    df["sex"] = np.where(df.competition.isin(LEAGUE_W | TOURN_W), "women", "men")
    if exclusion == "dismissal_window" and "dismissal_in_window" in df.columns:
        df = df[~df.dismissal_in_window.astype(bool)]
    elif exclusion == "after_dismissal" and "after_dismissal" in df.columns:
        df = df[~df.dismissal_in_window.astype(bool) & ~df.after_dismissal.astype(bool)]
    return df.reset_index(drop=True)


def ctx_key(df: pd.DataFrame, key: str) -> pd.Series:
    """매칭 키: 'state'(주 분석), 'pre'(참조), 'state_half', 'state_band', 'pre_half'."""
    if key == "state":
        return df.score_state
    if key == "pre":
        return df.state_pre
    if key == "state_half":
        return df.score_state + "|" + df.half.astype(str)
    if key == "pre_half":
        return df.state_pre + "|" + df.half.astype(str)
    if key == "state_band":
        return df.score_state + "|" + df.minute_band.astype(str)
    raise ValueError(key)


class Standardised:
    """한 (side, key) 에 대한 표준화 결과. ev/pl = (unit_id, event_id) 색인의 DataFrame: z(유닛 분모), zp(풀링 분모), k, bn, 메타."""

    def __init__(self, df: pd.DataFrame, key: str, min_n: int = MIN_N, control_filter=None):
        d = df.copy()
        d["ctx"] = ctx_key(d, key)
        base = d[~d.is_post]
        if control_filter is not None:
            base = base[control_filter(base)]
        b = base.groupby(["unit_id", "metric", "ctx"])["value"].agg(bm="mean", bs="std", bn="count").reset_index()
        self.cells = b[b.bn >= min_n].copy()
        self.pooled = self.cells.groupby("metric")["bs"].median().to_dict()
        self.base = base
        self.post = d[d.is_post]
        self.key = key

    def event_cells(self, window: str, need_k: int = 1) -> pd.DataFrame:
        """window 의 분석 사건이 속한 (unit_id, ctx) 셀 목록."""
        ev = self.frame("event", window, need_k=need_k).reset_index()
        return ev[["unit_id", "ctx"]].drop_duplicates()

    def frame(self, which: str, window: str | None = None, need_k: int = 1, loo: bool = False, cells: pd.DataFrame | None = None) -> pd.DataFrame:
        """which='event' → window 의 사건 창; 'control' → 통제 창(loo=True 면 leave-one-out 기저선).
        cells: (unit_id, ctx) 프레임을 주면 통제 창을 그 셀로 한정한다(주 추정: 사건이 있는 셀의 통제만; event_cells(window))."""
        if which == "event":
            fr = self.post[self.post.window == window]
            m = fr.merge(self.cells, on=["unit_id", "metric", "ctx"], how="inner")
            m["z"] = m.metric.map(DIR) * (m.value - m.bm) / m.bs.replace(0, np.nan)
        else:
            fr = self.base
            if cells is not None:
                fr = fr.merge(cells[["unit_id", "ctx"]].drop_duplicates(), on=["unit_id", "ctx"], how="inner")
            m = fr.merge(self.cells, on=["unit_id", "metric", "ctx"], how="inner")
            if loo:
                g = m.groupby(["unit_id", "metric", "ctx"])["value"]
                n = g.transform("count"); S = g.transform("sum"); SS = (m.value ** 2).groupby([m.unit_id, m.metric, m.ctx]).transform("sum")
                m1 = (S - m.value) / (n - 1); var = ((SS - m.value ** 2) - (n - 1) * m1 ** 2) / (n - 2)
                m["z"] = m.metric.map(DIR) * (m.value - m1) / np.sqrt(var.clip(lower=0)).replace(0, np.nan)
                m["zp"] = m.metric.map(DIR) * (m.value - m1) / m.metric.map(self.pooled)
            else:
                m["z"] = m.metric.map(DIR) * (m.value - m.bm) / m.bs.replace(0, np.nan)
        if "zp" not in m.columns:
            m["zp"] = m.metric.map(DIR) * (m.value - m.bm) / m.metric.map(self.pooled)
        m["rel"] = m.bs / m.metric.map(self.pooled)
        m = m[m.z.notna()]
        # 지표별 z 도 보존(상관·지표별 d 용)
        piv = m.pivot_table(index=["unit_id", "event_id"], columns="metric", values="z", aggfunc="first")
        g = m.groupby(["unit_id", "event_id"]).agg(z=("z", "mean"), zp=("zp", "mean"), k=("z", "size"), bn=("bn", "min"), rel=("rel", "mean"),
                                                   ctx=("ctx", "first"), match_id=("match_id", "first"), team=("team", "first"),
                                                   competition=("competition", "first"), format=("format", "first"), sex=("sex", "first"),
                                                   score_state=("score_state", "first"), elapsed=("elapsed", "first"),
                                                   window_dur=("window_dur", "first"), half=("half", "first"), period=("period", "first"), stage=("stage", "first"))
        g = g.join(piv.reindex(g.index).add_prefix("z_"))
        if "goal_index" in m.columns and which == "event":
            g = g.join(m.groupby(["unit_id", "event_id"]).agg(goal_index=("goal_index", "first"), event_elapsed=("event_elapsed", "first"),
                                                               n_goals_match=("n_goals_match", "first"), score_transition=("score_transition", "first"),
                                                               next_goal_dt=("next_goal_dt", "first"), goal_minute=("goal_minute", "first"),
                                                               other_goal_before_end=("other_goal_before_end", "first")))
        return g[g.k >= need_k]


# ── 효과크기 ──────────────────────────────────────────────────────
def wins(a: np.ndarray, lo=1, hi=99) -> np.ndarray:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return a if len(a) < 10 else np.clip(a, np.percentile(a, lo), np.percentile(a, hi))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp else float("nan")


def d_sep(ev, pl, lo=1, hi=99):
    """사건·통제 각각 윈저화(기본 규약)."""
    return cohen_d(wins(ev, lo, hi), wins(pl, lo, hi))


def d_joint(ev, pl, lo=1, hi=99):
    """합동 백분위 윈저화(민감도)."""
    ev, pl = np.asarray(ev, float), np.asarray(pl, float); q = np.percentile(np.concatenate([ev, pl]), [lo, hi])
    return cohen_d(np.clip(ev, *q), np.clip(pl, *q))


def d_raw(ev, pl):
    return cohen_d(np.asarray(ev, float), np.asarray(pl, float))


def boot_d(ev, pl, n=5000, seed=42, q=(2.5, 97.5), fn=d_sep):
    rng = np.random.default_rng(seed); ev, pl = np.asarray(ev, float), np.asarray(pl, float)
    ds = np.array([fn(rng.choice(ev, len(ev)), rng.choice(pl, len(pl))) for _ in range(n)])
    return [float(np.nanpercentile(ds, q[0])), float(np.nanpercentile(ds, q[1]))]


def estimate(ev, pl, n=5000, seed=42) -> dict:
    """기본 보고 묶음: d(각각 윈저), 95% CI, 90% 와 99% CI(TOST), n, 보조 규약."""
    ev, pl = np.asarray(ev, float), np.asarray(pl, float)
    rng = np.random.default_rng(seed)
    ds = np.array([d_sep(rng.choice(ev, len(ev)), rng.choice(pl, len(pl))) for _ in range(n)])
    out = {"d": round(d_sep(ev, pl), 4), "ci95": [round(float(np.nanpercentile(ds, 2.5)), 4), round(float(np.nanpercentile(ds, 97.5)), 4)],
           "ci90": [round(float(np.nanpercentile(ds, 5)), 4), round(float(np.nanpercentile(ds, 95)), 4)],
           "ci99": [round(float(np.nanpercentile(ds, 0.5)), 4), round(float(np.nanpercentile(ds, 99.5)), 4)],
           "p_one_sided_lt0": round(float(np.mean(ds >= 0)), 5), "n_event": int(len(ev)), "n_control": int(len(pl)),
           "d_joint": round(d_joint(ev, pl), 4), "d_raw": round(d_raw(ev, pl), 4), "d_trim5": round(d_sep(ev, pl, 5, 95), 4),
           "mean_event_w": round(float(wins(ev).mean()), 4), "mean_control_w": round(float(wins(pl).mean()), 4),
           "median_event": round(float(np.median(ev)), 4), "median_control": round(float(np.median(pl)), 4),
           "sd_event_w": round(float(wins(ev).std(ddof=1)), 4), "sd_control_w": round(float(wins(pl).std(ddof=1)), 4),
           "n_boot": n}
    # 중앙값 차이(강건 추정) 부트스트랩
    md = np.array([np.median(rng.choice(ev, len(ev))) - np.median(rng.choice(pl, len(pl))) for _ in range(min(n, 2000))])
    out["median_diff"] = round(float(np.median(ev) - np.median(pl)), 4)
    out["median_diff_ci95"] = [round(float(np.percentile(md, 2.5)), 4), round(float(np.percentile(md, 97.5)), 4)]
    return out


def cluster_boot_d(ev_df: pd.DataFrame, pl_df: pd.DataFrame, cluster: str, n=1000, seed=7, col="z", keep=False) -> dict:
    """클러스터(유닛/경기/팀) 재표집 부트스트랩 CI. 두 집단에서 같은 클러스터 집합을 재표집한다."""
    rng = np.random.default_rng(seed)
    ev = ev_df.reset_index(); pl = pl_df.reset_index()
    clusters = np.array(sorted(set(ev[cluster]) | set(pl[cluster])))
    ie = {c: ev.index[ev[cluster] == c].to_numpy() for c in clusters}; ip = {c: pl.index[pl[cluster] == c].to_numpy() for c in clusters}
    e = ev[col].to_numpy(); p = pl[col].to_numpy(); ds = []
    for _ in range(n):
        pick = rng.choice(clusters, len(clusters))
        ei = np.concatenate([ie[c] for c in pick]); pi = np.concatenate([ip[c] for c in pick])
        if len(ei) < 10 or len(pi) < 10:
            continue
        ds.append(d_sep(e[ei], p[pi]))
    ds = np.array(ds)
    out = {"d": round(d_sep(e, p), 4), "ci95": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
           "se": round(float(ds.std(ddof=1)), 4), "n_clusters": int(len(clusters)), "cluster": cluster, "n_boot": int(len(ds)),
           "n_event": int(len(e)), "n_control": int(len(p))}
    if keep:
        out["_ds"] = ds
    return out


def cluster_inference(ev_df: pd.DataFrame, pl_df: pd.DataFrame, cluster: str = "unit_id", n=5000, seed=7, col="z", sesoi=0.15) -> dict:
    """군집(기본: 팀-경기 유닛) 재표집 부트스트랩으로 d 의 95%·99% CI, 단측 p(악화), 등가 판정과 Monte Carlo 오차를 계산한다.
    통제 창은 cells 로 한정된 상태로 넘겨야 한다(주 추정과 같은 집합)."""
    r = cluster_boot_d(ev_df, pl_df, cluster, n=n, seed=seed, col=col, keep=True)
    ds = np.asarray(r.pop("_ds")); p = float(np.mean(ds >= 0)); nb = len(ds)
    r.update({"ci99": [round(float(np.percentile(ds, 0.5)), 4), round(float(np.percentile(ds, 99.5)), 4)],
              "ci90": [round(float(np.percentile(ds, 5)), 4), round(float(np.percentile(ds, 95)), 4)],
              "p_one_sided_lt0": round(p, 5), "p_mc_se": round(float(np.sqrt(max(p, 1 / nb) * (1 - max(p, 1 / nb)) / nb)), 5),
              "p_one_sided_lt0_upper95": round(float(np.percentile(np.random.default_rng(1).binomial(nb, max(p, 0.5 / nb), 2000) / nb, 97.5)), 5),
              "n_boot": nb, "sesoi": sesoi})
    r["tost"] = {"equivalent_alpha005": bool(r["ci99"][0] > -sesoi and r["ci99"][1] < sesoi), "improvement_excluded_alpha005": bool(r["ci99"][1] < sesoi),
                 "deterioration_excluded_alpha005": bool(r["ci99"][0] > -sesoi), "deterioration_criterion_met_alpha005": bool(p < 0.005)}
    return r


def perm_within_unit(ev_df: pd.DataFrame, pl_df: pd.DataFrame, n=2000, seed=11, col="z") -> dict:
    """유닛 안에서 사건/통제 라벨을 뒤섞는 순열 귀무분포(d 의 중앙 95% 구간)와 관측 d 의 양측 p."""
    rng = np.random.default_rng(seed)
    ev = ev_df.reset_index()[["unit_id", col]].assign(is_ev=1); pl = pl_df.reset_index()[["unit_id", col]].assign(is_ev=0)
    allw = pd.concat([ev, pl], ignore_index=True)
    allw = allw[allw.unit_id.isin(set(ev.unit_id))]           # 사건이 있는 유닛만 뒤섞음 (사건 없는 유닛의 통제는 항상 통제)
    others = pl[~pl.unit_id.isin(set(ev.unit_id))][col].to_numpy()
    grp = allw.groupby("unit_id").indices; z = allw[col].to_numpy(); lab = allw.is_ev.to_numpy().copy()
    d_obs = d_sep(z[lab == 1], np.concatenate([z[lab == 0], others]))
    ds = []
    for _ in range(n):
        l2 = lab.copy()
        for idx in grp.values():
            l2[idx] = rng.permutation(lab[idx])
        ds.append(d_sep(z[l2 == 1], np.concatenate([z[l2 == 0], others])))
    ds = np.array(ds)
    return {"d_obs": round(d_obs, 4), "null95": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
            "p_two_sided": round(float(np.mean(np.abs(ds) >= abs(d_obs))), 5), "n_perm": n}


def tost(est: dict, sesoi=0.15) -> dict:
    """등가 판정: (1−2α) CI 가 ±SESOI 안에 들면 등가. α=0.005 → 99% CI(일관 α), α=0.05 → 90% CI(관행)."""
    return {"sesoi": sesoi,
            "equivalent_alpha005": bool(est["ci99"][0] > -sesoi and est["ci99"][1] < sesoi),
            "equivalent_alpha05": bool(est["ci90"][0] > -sesoi and est["ci90"][1] < sesoi),
            "improvement_excluded_alpha005": bool(est["ci99"][1] < sesoi), "deterioration_excluded_alpha005": bool(est["ci99"][0] > -sesoi)}


def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))
