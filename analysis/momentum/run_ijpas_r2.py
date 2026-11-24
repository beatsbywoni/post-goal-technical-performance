"""IJPAS 1차 수정 · 2차 분석(R1b, 2026-10) — 골 원장·단일 골 노출·사건 셀 통제 규약으로 원고의 모든 수치를 다시 계산한다.

  python3 -m analysis.momentum.run_ijpas_r2 all                      # 전체 → docs/ijpas_r2.json (메모리 때문에 파트별 실행 권장)
  R2_OUT=docs/ijpas_r2_A.json python3 -m analysis.momentum.run_ijpas_r2 main cells flow
  python3 -m analysis.momentum.run_ijpas_r2 merge docs/ijpas_r2_A.json ...
  python3 -m analysis.momentum.run_ijpas_r2 variant v2_own <features_dir>

파트: main cells flow strata loco oppq paired depend crossed sens standard indicators nearest binom descriptors mdes clusterw (all = 이 전부);
      R2 추가: clipgrid (clipping 격자에 계획서 규칙 추가 + 다음 피리어드 통제 비중), indpooled (지표별 within-unit·pooled d) — all 에 포함되지 않으므로 따로 실행
규약(analysis/momentum/r1_lib.py docstring 과 1:1):
  · 사건 창의 점수 상태 = 골 원장의 state_after(기준 골이 만든 상태); 참조 분석의 상태 = state_before(기준 골 직전 상태).
  · 단일 골 노출: 기준 골과 창 종료 사이에 다른 골(어느 팀이든)이 있는 사건 창은 제외(exposure="single_goal").
  · 주 추정의 통제 = 사건이 있는 셀(unit × state)의 통제 창; 적격 셀 전체의 통제는 민감도(all_cells).
  · d = (mean_event − mean_control)/pooled SD; 사건·통제 각각 1/99 윈저화; 추론은 유닛 클러스터 부트스트랩(5,000) 95%·99% CI 와 단측 p.
"""
from __future__ import annotations

import json, os, sys, time, warnings, platform
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.momentum.r1_lib import (ANCHOR, COMP_ORDER, DIR, EXEC, FILES, HI, MIN_N, STRATA, WINDOWS, Standardised, boot_d,  # noqa: E402
                                      cluster_boot_d, cluster_inference, cohen_d, ctx_key, d_joint, d_raw, d_sep, estimate, load,
                                      perm_within_unit, save_json, tost, wins)

OUT = Path(os.environ.get("R2_OUT", str(ROOT / "docs/ijpas_r2.json")))
RES: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
T0 = time.time()
SCOPES = ("event_cells", "all_cells")


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def put(key, value):
    RES[key] = value; save_json(RES, OUT)


def frames(S: Standardised, window: str, scope: str = "event_cells", need_k: int = 1, loo: bool = False):
    """(사건 창, 통제 창) — scope='event_cells' 면 통제를 사건이 있는 셀로 한정."""
    ev = S.frame("event", window, need_k=need_k)
    cells = ev.reset_index()[["unit_id", "ctx"]].drop_duplicates() if scope == "event_cells" else None
    pl = S.frame("control", need_k=need_k, loo=loo, cells=cells)
    return ev, pl


def both(ev, pl, n=5000):
    return {"unit": estimate(ev.z.to_numpy(), pl.z.to_numpy(), n=n), "pooled": estimate(ev.zp.to_numpy(), pl.zp.to_numpy(), n=n)}


def lmm(ev, pl, col="z", groups="unit_id", extra_fixed=None, vc=None, methods=("lbfgs", "cg", "powell")):
    """선형혼합모형 net z ~ event (+extra), 랜덤 절편 groups (+vc). 사건·통제 각각 1/99 윈저화. 수렴 진단을 기록한다."""
    import statsmodels.formula.api as smf
    e = ev.reset_index().copy(); p = pl.reset_index().copy()
    for fr in (e, p):
        lo, hi = np.percentile(fr[col], [1, 99]); fr["y"] = fr[col].clip(lo, hi)
    e["event"] = 1; p["event"] = 0; df = pd.concat([e, p], ignore_index=True)
    formula = "y ~ event" + (" + " + extra_fixed if extra_fixed else "")
    out = None
    for meth in methods:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            md = smf.mixedlm(formula, df, groups=df[groups], vc_formula=vc, re_formula="1")
            try:
                r = md.fit(reml=True, method=[meth], maxiter=1000)
            except Exception as ex:  # noqa: BLE001
                out = {"error": str(ex), "method": meth}; continue
        fixed = [k for k in r.params.index if not k.endswith(" Var") and k != "Group Var"]
        llf = float(r.llf); ses = {k: float(r.bse[k]) for k in fixed}
        out = {"params": {k: round(float(r.params[k]), 4) for k in fixed}, "se": {k: round(v, 4) for k, v in ses.items()},
               "p": {k: float(r.pvalues[k]) for k in fixed}, "n": int(len(df)), "groups": int(df[groups].nunique()),
               "converged": bool(r.converged), "method": meth, "llf": (round(llf, 2) if np.isfinite(llf) else str(llf)),
               "warnings": sorted({str(w.message)[:160] for w in wlist})[:6]}
        vc_names = list(getattr(md.exog_vc, "names", []) or [])
        vcomp = np.atleast_1d(np.asarray(r.vcomp, float)) if hasattr(r, "vcomp") else np.array([])
        out["var_components"] = {n: round(float(v), 5) for n, v in zip(vc_names, vcomp)} if len(vcomp) else {}
        out["group_var"] = round(float(np.asarray(r.cov_re)[0, 0]), 5); out["resid_var"] = round(float(r.scale), 5)
        out["boundary_variance"] = bool(out["group_var"] < 1e-6 or any(v < 1e-6 for v in out["var_components"].values()))
        out["max_fixed_se"] = round(max(ses.values()), 4)
        # a fit is usable as evidence only if it converged with a finite log-likelihood and finite, sensible standard errors
        out["valid_fit"] = bool(out["converged"] and np.isfinite(llf) and all(np.isfinite(v) and v < 10 for v in ses.values()))
        # numerically fitted (valid_fit) is distinguished from usable as inferential evidence: no boundary variance, no Hessian warning
        out["usable_as_evidence"] = bool(out["valid_fit"] and not out["boundary_variance"] and not any("Hessian" in w or "boundary" in w for w in out["warnings"]))
        if out["valid_fit"]:
            break
    return out


def twoway_interactions(ev, pl, col, factors):
    """사건 효과의 층 상호작용: 유닛 내 평균 제거(within-unit demeaning) OLS, 팀 × 경기 두 방향 군집 강건 SE.
    y_dm = b·event_dm + Σ b_k·(event × 1[level k])_dm. factors = {name: (reference, level, ...)}. 사건·통제 각각 1/99 윈저화."""
    import statsmodels.api as sm
    e = ev.reset_index().copy(); p = pl.reset_index().copy()
    for fr in (e, p):
        lo, hi = np.percentile(fr[col], [1, 99]); fr["y"] = fr[col].clip(lo, hi)
    e["event"] = 1.0; p["event"] = 0.0; d = pd.concat([e, p], ignore_index=True)
    cols = {"event": d.event.to_numpy()}
    for name, levels in factors.items():
        for lv in levels[1:]:
            cols[f"event:{name}[{lv}]"] = (d.event * (d[name] == lv)).to_numpy()
    X = pd.DataFrame(cols); g = d.groupby("unit_id")
    y_dm = d.y - g.y.transform("mean"); X_dm = X - X.groupby(d.unit_id.to_numpy()).transform("mean")
    groups = np.column_stack([pd.factorize(d.team)[0], pd.factorize(d.match_id)[0]])
    fit = sm.OLS(y_dm, X_dm).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int()
    terms = {k: {"coef": round(float(fit.params[k]), 4), "se": round(float(fit.bse[k]), 4), "ci95": [round(float(ci.loc[k, 0]), 4), round(float(ci.loc[k, 1]), 4)],
                 "p": float(fit.pvalues[k])} for k in X.columns}
    n_lv = {name: {lv: int((d[name] == lv).sum()) for lv in levels} for name, levels in factors.items()}
    return {"terms": terms, "reference": {name: levels[0] for name, levels in factors.items()}, "n_rows": int(len(d)), "n_by_level": n_lv,
            "n_teams": int(d.team.nunique()), "n_matches": int(d.match_id.nunique()), "outcome": col,
            "definition": "within-unit demeaned OLS of winsorised net z on the event indicator and event × stratum terms; two-way cluster-robust SE (team, match); the event coefficient is the effect in the reference stratum"}


def raw_units(side_df, key, window):
    """지표별 원 단위 차이(사건 − 사건 셀의 통제 평균; 비율은 %p). 창 부트스트랩과 유닛 클러스터 부트스트랩."""
    out = {}; rng = np.random.default_rng(3)
    for m in EXEC:
        d = side_df[side_df.metric == m].copy(); d["ctx"] = ctx_key(d, key)
        base = d[~d.is_post]; b = base.groupby(["unit_id", "ctx"])["value"].agg(bm="mean", bn="count").reset_index(); b = b[b.bn >= MIN_N]
        e = d[(d.is_post) & (d.window == window)].merge(b, on=["unit_id", "ctx"])
        de = (e.value - e.bm).to_numpy() * 100
        bs = [rng.choice(de, len(de)).mean() for _ in range(2000)]
        units = e.unit_id.to_numpy(); uu = np.unique(units); idx = {u: np.where(units == u)[0] for u in uu}
        cb = [de[np.concatenate([idx[u] for u in rng.choice(uu, len(uu))])].mean() for _ in range(2000)]
        out[m] = {"diff_pp": round(float(de.mean()), 3), "ci95_window": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                  "ci95_unit_cluster": [round(float(np.percentile(cb, 2.5)), 3), round(float(np.percentile(cb, 97.5)), 3)],
                  "control_mean_pct": round(float(e.bm.mean() * 100), 2), "event_mean_pct": round(float(e.value.mean() * 100), 2), "n": int(len(de)),
                  "note": "event window minus the mean of its own cell's control windows (in sample); equal weight per event"}
    return out


# ── parts ──────────────────────────────────────────────────────────────────────
def part_main(data):
    """표 3: 사건 × 창 × 매칭(pre/state/state_half) × 분모(유닛/풀링) × 통제 범위(사건 셀/전체) + 원 단위 + 앵커의 군집 추론·LMM."""
    for side, df in data.items():
        for key in ("pre", "state", "state_half"):
            S = Standardised(df, key)
            for w in WINDOWS:
                ev, pl = frames(S, w); _, pla = frames(S, w, "all_cells")
                r = {**both(ev, pl), "all_cells": both(ev, pla, n=2000),
                     "k3_unit": estimate(ev[ev.k == 3].z.to_numpy(), pl[pl.k == 3].z.to_numpy(), n=2000),
                     "n_cells_event": int(ev.reset_index()[["unit_id", "ctx"]].drop_duplicates().shape[0]), "n_units_event": int(ev.reset_index().unit_id.nunique()),
                     "n_cells_control": int(pl.reset_index().groupby(["unit_id", "ctx"]).ngroups), "n_cells_control_all": int(pla.reset_index().groupby(["unit_id", "ctx"]).ngroups),
                     "k_dist": ev.k.value_counts().to_dict(), "k3_share_pct": round(float((ev.k == 3).mean() * 100), 1)}
                if key in ("pre", "state"):
                    r["raw"] = raw_units(df, key, w)
                if w == ANCHOR:
                    r["cluster_unit"] = cluster_inference(ev, pl, "unit_id", n=5000); r["cluster_unit_pooled"] = cluster_inference(ev, pl, "unit_id", n=5000, col="zp")
                    r["unit"]["tost_window_boot"] = tost(r["unit"]); r["pooled"]["tost_window_boot"] = tost(r["pooled"])
                    r["lmm_unit"] = lmm(ev, pl, "z"); r["lmm_pooled_unit"] = lmm(ev, pl, "zp")
                put(f"{side}|{key}|{w}", r)
                log(side, key, w, "unit", r["unit"]["d"], r["unit"]["ci95"], "all", r["all_cells"]["unit"]["d"], "pooled", r["pooled"]["d"], "n", r["unit"]["n_event"], r["unit"]["n_control"])
        S = Standardised(df, "state")
        for w in WINDOWS:
            ev, pl = frames(S, w)
            put(f"{side}|state|{w}|per_indicator", {m: {"d": round(d_sep(ev[f"z_{m}"].dropna().to_numpy(), pl[f"z_{m}"].dropna().to_numpy()), 4),
                                                        "ci95": boot_d(ev[f"z_{m}"].dropna().to_numpy(), pl[f"z_{m}"].dropna().to_numpy(), n=1000),
                                                        "n": int(ev[f"z_{m}"].notna().sum())} for m in EXEC})
        # 중간 골 유지(R1 1차 규약) 민감도
        d_all = load(side, exposure="all"); S2 = Standardised(d_all, "state"); ev2, pl2 = frames(S2, ANCHOR)
        put(f"{side}|exposure_all|{ANCHOR}", {**both(ev2, pl2, n=2000), "n_multi_goal_windows": int(ev2.other_goal_before_end.astype(bool).sum())})
        log(side, "exposure=all", RES[f"{side}|exposure_all|{ANCHOR}"]["unit"]["d"])


def part_cells(data):
    """셀 통계(사건 셀의 통제 창 수 분포), K=3 항등식(지표 z 평균 = net z, 윈저화 전) 검증, 가중치 서술용 수치."""
    meta = RES.get("_meta", {})
    for side, df in data.items():
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR)
        k3 = ev[ev.k == 3]; ident = np.nanmax(np.abs(k3[[f"z_{m}" for m in EXEC]].mean(axis=1) - k3.z)) if len(k3) else float("nan")
        meta[f"cell_windows_{side}"] = {"median": float(np.median(ev.bn)), "p10": float(np.percentile(ev.bn, 10)), "p90": float(np.percentile(ev.bn, 90)), "n_event": int(len(ev)),
                                        "control_windows_in_event_cells": int(len(pl)), "mean_controls_per_event_cell": round(float(len(pl) / len(ev)), 2),
                                        "k3_identity_max_abs_dev": float(ident), "k3_share_pct": round(float((ev.k == 3).mean() * 100), 1)}
    meta["median_cell_windows"] = meta["cell_windows_score"]["median"]; meta["p10_cell_windows"] = meta["cell_windows_score"]["p10"]
    put("_meta", meta); log("cells", {k: v for k, v in meta.items() if k.startswith("cell_windows")})


def part_flow(data):
    """흐름도(골 원장 기준, 상호배타적 순차 제외): 6,528 골 → 창 데이터 없음 → 다른 골이 창 종료 전 → 창에 퇴장 → 같은 상태 통제 셀 < 3
    (퇴장 창 제외 탓인지 구분) → z 정의 불가 → 분석. 참조 분석(직전 상태)도 같은 방식. 포함/제외 골 비교는 전체 6,528 골 분모."""
    led = pd.read_parquet(ROOT / "data/interim/goal_ledger.parquet"); led["gid"] = led.match_id + "#" + led.goal_index.astype(str)
    stage = json.loads((ROOT / "data/interim/match_stage.json").read_text())
    for side in ("score", "concede"):
        full = load(side, exclusion="none", exposure="all"); df = data[side]
        post = full[full.is_post].copy(); post["gid"] = post.match_id + "#" + post.goal_index.astype(int).astype(str)
        # 통제 셀 크기: 퇴장 창 포함/제외
        flow = {"goals_total": int(len(led)), "events_total": int(len(led))}
        for key in ("state", "pre"):
            S_none = Standardised(full, key); S = Standardised(df, key)
            for w in WINDOWS:
                hi = HI[w]
                has = post[post.window == w].groupby("gid").first()
                step1 = set(led.gid) - set(has.index)                                   # 창 데이터 없음
                multi = set(has[has.other_goal_before_end.astype(bool)].index)           # 다른 골이 창 종료 전
                rem = set(has.index) - multi
                dism = set(has[has.dismissal_in_window.astype(bool)].index) & rem           # 창에 퇴장
                rem = rem - dism
                # 셀 조건: 단일 골·퇴장 제외 후의 사건이 df(주 규약) 에 있고 셀 ≥3 인지
                an = S.frame("event", w).reset_index(); an["gid"] = an.match_id + "#" + an.goal_index.astype(int).astype(str)
                analysed = set(an.gid) & rem
                an_none = S_none.frame("event", w).reset_index(); an_none["gid"] = an_none.match_id + "#" + an_none.goal_index.astype(int).astype(str)
                cell_ok_with_dism_controls = set(an_none.gid) & rem                        # 퇴장 창을 통제에 넣으면 셀 ≥3
                short_cell = rem - analysed
                short_because_dismissal_controls = short_cell & cell_ok_with_dism_controls
                short_other = short_cell - short_because_dismissal_controls
                # z 정의 불가(SD=0 등): 사건 행은 있고 셀도 ≥3 인데 z 가 NaN 인 경우 — frame 이 이미 제거; 셀 ≥3 인 사건 중 frame 에 없는 것
                flow[f"{key}|{w}"] = {"goals_total": int(len(led)), "no_window_data": int(len(step1)), "other_goal_before_window_end": int(len(multi)),
                                     "dismissal_in_window": int(len(dism)), "cell_short_due_to_dismissal_control_exclusion": int(len(short_because_dismissal_controls)),
                                     "cell_short_or_z_undefined": int(len(short_other)), "analysed": int(len(analysed)),
                                     "check_sum": int(len(step1) + len(multi) + len(dism) + len(short_because_dismissal_controls) + len(short_other) + len(analysed))}
                assert flow[f"{key}|{w}"]["check_sum"] == len(led), flow[f"{key}|{w}"]
        # 포함/제외 비교(앵커, 주 분석): 전체 골 분모와 창 데이터가 있는 골 분모
        an = Standardised(df, "state").frame("event", ANCHOR).reset_index()
        an["gid"] = an.match_id + "#" + an.goal_index.astype(int).astype(str); inc = set(an.gid)
        L = led.copy(); L["included"] = L.gid.isin(inc); L["has_data"] = L.gid.isin(set(post[post.window == ANCHOR].gid))
        L["stage"] = L.match_id.map(lambda m: {"Regular Season": "league", "Group Stage": "group"}.get(stage.get(m, {}).get("stage", ""), "knockout"))
        L["women"] = L.competition.isin({"FA Women's Super League", "Women's World Cup"})
        L["state_after_team"] = L.state_after if side == "score" else L.state_after.map(lambda s: "-".join(s.split("-")[::-1]))
        def summ(g):
            return {"n": int(len(g)), "minute_mean": round(float(g.minute.mean()), 1), "minute_median": round(float(g.minute.median()), 1),
                    "goals_in_match_mean": round(float(g.n_goals_match.mean()), 2), "share_knockout_pct": round(float((g.stage == "knockout").mean() * 100), 1),
                    "share_group_pct": round(float((g.stage == "group").mean() * 100), 1), "share_women_pct": round(float(g.women.mean() * 100), 1),
                    "share_own_goal_pct": round(float(g.own_goal.mean() * 100), 1), "share_penalty_pct": round(float(g.penalty.mean() * 100), 1),
                    "next_goal_within_15min_pct": round(float((g.next_goal_dt < 900).mean() * 100), 1),
                    "top_states": g.state_after_team.value_counts(normalize=True).head(5).round(3).to_dict()}
        flow["included_vs_excluded_anchor"] = {"included": summ(L[L.included]), "excluded_all": summ(L[~L.included]), "excluded_with_window_data": summ(L[(~L.included) & L.has_data]),
                                               "excluded_no_window_data": summ(L[(~L.included) & (~L.has_data)])}
        put(f"{side}|flow", flow); log(side, "flow", flow[f"state|{ANCHOR}"], flow["included_vs_excluded_anchor"]["included"]["n"])


def _stratum(df, comps=None, mask=None, n=2000):
    sub = df[df.competition.isin(comps)] if comps is not None else df[mask]
    r = {}
    for key in ("pre", "state"):
        S = Standardised(sub, key); ev, pl = frames(S, ANCHOR)
        r[key] = both(ev, pl, n=n) if len(ev) >= 20 else {"unit": {"d": float("nan"), "n_event": int(len(ev))}, "pooled": {"d": float("nan")}}
    return r


def part_strata(data):
    """형식×성별 층, 대회별, 경기 단계(league/group/knockout), 상호작용 LMM(앵커)."""
    for side, df in data.items():
        res = {}
        groups = dict(STRATA); groups.update({"league": STRATA["men_league"] | STRATA["women_league"], "tournament": STRATA["men_tournament"] | STRATA["women_tournament"],
                                              "men": STRATA["men_league"] | STRATA["men_tournament"], "women": STRATA["women_league"] | STRATA["women_tournament"]})
        groups.update({c: {c} for c in COMP_ORDER})
        for name, comps in groups.items():
            res[name] = _stratum(df, comps=comps); log(side, "stratum", name, res[name]["pre"]["unit"]["d"], res[name]["state"]["unit"]["d"], res[name]["state"]["unit"].get("n_event"))
        for st in ("league", "group", "knockout"):
            res[f"stage_{st}"] = _stratum(df, mask=(df.stage == st)); log(side, "stage", st, res[f"stage_{st}"]["state"]["unit"]["d"], res[f"stage_{st}"]["state"]["unit"].get("n_event"))
        for st in ("group", "knockout"):
            for sx in ("men", "women"):
                res[f"stage_{st}_{sx}"] = _stratum(df, mask=(df.stage == st) & (df.sex == sx), n=1000)
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR)
        res["interaction_lmm_unit"] = lmm(ev, pl, "z", extra_fixed="event:C(format, Treatment('league')) + event:C(sex, Treatment('men'))")
        res["interaction_lmm_pooled"] = lmm(ev, pl, "zp", extra_fixed="event:C(format, Treatment('league')) + event:C(sex, Treatment('men'))")
        res["interaction_lmm_stage_unit"] = lmm(ev, pl, "z", extra_fixed="event:C(stage, Treatment('league')) + event:C(sex, Treatment('men'))")
        # interactions with cluster-robust inference: within-unit demeaned OLS, two-way cluster (team × match); the unit-intercept LMMs
        # above sit on a variance boundary (control z average zero within a cell by construction) and are kept only as diagnostics
        for col in ("z", "zp"):
            res[f"interaction_twoway_{col}"] = twoway_interactions(ev, pl, col, {"format": ("league", "tournament"), "sex": ("men", "women")})
        res["interaction_twoway_stage_z"] = twoway_interactions(ev, pl, "z", {"stage": ("league", "group", "knockout"), "sex": ("men", "women")})
        log(side, "interaction twoway", {k: (v["coef"], v["ci95"], round(v["p"], 3)) for k, v in res["interaction_twoway_z"]["terms"].items()})
        put(f"{side}|strata", res)


def part_loco(data):
    """대회 하나씩 제외(leave-one-competition-out) 재추정 — 10회, 주 분석과 참조 분석, 유닛/풀링."""
    for side, df in data.items():
        res = {}
        for c in COMP_ORDER:
            sub = df[df.competition != c]; r = {"excluded": c}
            for key in ("pre", "state"):
                S = Standardised(sub, key); ev, pl = frames(S, ANCHOR); r[key] = both(ev, pl, n=2000)
            res[c] = r; log(side, "loco without", c, r["state"]["unit"]["d"], r["state"]["unit"]["ci95"], r["state"]["unit"]["n_event"])
        ds = [res[c]["state"]["unit"]["d"] for c in COMP_ORDER]; dp = [res[c]["pre"]["unit"]["d"] for c in COMP_ORDER]
        res["range_state_unit"] = [round(min(ds), 4), round(max(ds), 4)]; res["range_pre_unit"] = [round(min(dp), 4), round(max(dp), 4)]
        put(f"{side}|loco", res)


def part_oppq(data):
    """상대 팀 수준(사후): 대회-시즌 내 상대의 승점/경기 3분위."""
    venue = json.loads((ROOT / "data/interim/football_venue.json").read_text())
    pts = {}
    for mid, v in venue.items():
        key = (v["competition"], v.get("season", "")); hs, as_ = v["home_score"], v["away_score"]
        hp, ap = (3, 0) if hs > as_ else (0, 3) if hs < as_ else (1, 1)
        for team, pnt in ((v["home"], hp), (v["away"], ap)):
            a = pts.setdefault((key, team), [0.0, 0]); a[0] += pnt; a[1] += 1
    ppg = {k: v[0] / v[1] for k, v in pts.items()}; terc = {}
    for key in {k[0] for k in ppg}:
        vals = np.array([v for k, v in ppg.items() if k[0] == key]); terc[key] = (np.percentile(vals, 100 / 3), np.percentile(vals, 200 / 3))
    def tier(mid, team):
        v = venue.get(str(mid))
        if not v: return "unknown"
        opp = v["away"] if v["home"] == team else v["home"] if v["away"] == team else None
        q = ppg.get(((v["competition"], v.get("season", "")), opp))
        if q is None: return "unknown"
        lo, hi = terc[(v["competition"], v.get("season", ""))]; return "low" if q <= lo else "high" if q > hi else "mid"
    for side, df in data.items():
        units = df[["unit_id", "match_id", "team"]].drop_duplicates(); units["opp_tier"] = [tier(m, t) for m, t in zip(units.match_id, units.team)]
        tmap = dict(zip(units.unit_id, units.opp_tier)); d2 = df.assign(opp_tier=df.unit_id.map(tmap))
        res = {"n_units_by_tier": units.opp_tier.value_counts().to_dict(), "definition": "opponent's points per match over the competition-season (3/1/0), terciles within competition-season"}
        for t in ("low", "mid", "high"):
            res[t] = _stratum(d2, mask=(d2.opp_tier == t)); log(side, "opponent", t, res[t]["state"]["unit"]["d"], res[t]["state"]["unit"].get("n_event"))
        put(f"{side}|oppq", res)


def part_paired(data):
    """H3. (a) 사전 계획 A4: |d_scoring| > |d_conceding| (득점 후 저하가 더 큼) — 창 재표집 부트스트랩, 단측 p.
    (b) 사후(리뷰어 요청): 같은 골에서 D = scoring net z − conceding net z (양수 = 득점 팀이 상대적으로 높음), 경기 클러스터 부트스트랩."""
    rng = np.random.default_rng(21); out = {}
    Ss = Standardised(data["score"], "state"); Sc = Standardised(data["concede"], "state")
    es, pls = frames(Ss, ANCHOR); ec, plc = frames(Sc, ANCHOR)
    es = es.reset_index(); ec = ec.reset_index()
    for fr in (es, ec):
        fr["goal"] = fr.match_id.astype(str) + "#" + fr.goal_index.astype(int).astype(str)
    # (a) 계획 A4
    pls_r = pls.reset_index(); plc_r = plc.reset_index()
    for col in ("z", "zp"):
        e1, p1, e2, p2 = es[col].to_numpy(), pls_r[col].to_numpy(), ec[col].to_numpy(), plc_r[col].to_numpy()
        d_s, d_c = d_sep(e1, p1), d_sep(e2, p2)
        # window resampling (submitted convention; sides resampled independently)
        dd = np.array([abs(d_sep(rng.choice(e1, len(e1)), rng.choice(p1, len(p1)))) - abs(d_sep(rng.choice(e2, len(e2)), rng.choice(p2, len(p2)))) for _ in range(2000)])
        # joint match-cluster resampling: one draw of matches, both teams' events and event-cell controls of the drawn matches enter together
        mids = np.unique(np.concatenate([es.match_id.astype(str).to_numpy(), ec.match_id.astype(str).to_numpy()]))
        ix = {}
        for name, fr in (("e1", es), ("p1", pls_r), ("e2", ec), ("p2", plc_r)):
            mcol = fr.match_id.astype(str).to_numpy(); ix[name] = {mm: np.where(mcol == mm)[0] for mm in np.unique(mcol)}
        arr = {"e1": e1, "p1": p1, "e2": e2, "p2": p2}
        def take(name, draw):
            sel = [ix[name][mm] for mm in draw if mm in ix[name]]
            return arr[name][np.concatenate(sel)] if sel else arr[name][:0]
        dj = []
        for _ in range(5000):
            draw = rng.choice(mids, len(mids))
            dj.append(abs(d_sep(take("e1", draw), take("p1", draw))) - abs(d_sep(take("e2", draw), take("p2", draw))))
        dj = np.array(dj)
        out[f"A4_abs_d_scoring_minus_abs_d_conceding_{col}"] = {"d_scoring": round(d_s, 4), "d_conceding": round(d_c, 4), "diff_abs": round(abs(d_s) - abs(d_c), 4),
                                                                 "ci95": [round(float(np.percentile(dj, 2.5)), 4), round(float(np.percentile(dj, 97.5)), 4)],
                                                                 "p_one_sided_scoring_larger": round(float(np.mean(dj <= 0)), 5), "n_boot": 5000, "resampling": "joint match cluster",
                                                                 "n_matches": int(len(mids)),
                                                                 "window_boot": {"ci95": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)],
                                                                                 "p_one_sided_scoring_larger": round(float(np.mean(dd <= 0)), 5), "n_boot": 2000, "resampling": "windows, sides independent (submitted convention)"},
                                                                 "definition": "pre-specified A4: |d after scoring| − |d after conceding|; H3 predicts > 0; primary inference by joint match-cluster bootstrap"}
    # (b) 같은 골 대응 차이
    m = es.merge(ec, on="goal", suffixes=("_s", "_c"))
    for col in ("z", "zp"):
        a = m[f"{col}_s"].to_numpy().copy(); b = m[f"{col}_c"].to_numpy().copy()
        for arr in (a, b):
            lo, hi = np.percentile(arr, [1, 99]); np.clip(arr, lo, hi, out=arr)
        diff = a - b; match = m.match_id_s.to_numpy(); um = np.unique(match); idx = {u: np.where(match == u)[0] for u in um}
        bs = np.array([diff[np.concatenate([idx[u] for u in rng.choice(um, len(um))])].mean() for _ in range(2000)])
        out[f"paired_{col}"] = {"n_goals": int(len(m)), "mean_D_scoring_minus_conceding": round(float(diff.mean()), 4),
                                "ci95_match_cluster": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                "p_two_sided": round(float(2 * min(np.mean(bs <= 0), np.mean(bs >= 0))), 5), "sd_D": round(float(diff.std(ddof=1)), 4),
                                "d_z_paired": round(float(diff.mean() / diff.std(ddof=1)), 4), "median_D": round(float(np.median(diff)), 4),
                                "share_scoring_higher_pct": round(float((diff > 0).mean() * 100), 1),
                                "definition": "post hoc (reviewer-requested): D = scoring team's net z − conceding team's net z in the same window of the same goal; d_z = mean(D)/SD(D), not comparable with the event-vs-control Cohen's d"}
    put("paired", out); log("paired", {k: (v.get("diff_abs"), v.get("mean_D_scoring_minus_conceding")) for k, v in out.items()})


def part_depend(data):
    """의존 구조: 유닛/경기/팀 클러스터 부트스트랩(5,000), 유닛 내 순열 귀무, 팀+유닛 중첩 LMM(수렴 진단),
    유닛 수준 대비의 교차 랜덤효과 모형(팀 × 경기; statsmodels variance components, 단일 그룹), 두 방향 군집 강건 SE."""
    import statsmodels.api as sm, statsmodels.formula.api as smf
    for side, df in data.items():
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR)
        res = {c: cluster_inference(ev, pl, c, n=5000) for c in ("unit_id", "match_id", "team")}
        res["perm_within_unit"] = perm_within_unit(ev, pl, n=2000)
        res["perm_within_unit_pooled"] = perm_within_unit(ev, pl, n=2000, col="zp")
        res["lmm_team_unit"] = lmm(ev, pl, "z", groups="team", vc={"unit": "0 + C(unit_id)"})
        log(side, "lmm team+unit", res["lmm_team_unit"].get("converged"), res["lmm_team_unit"].get("params"))
        # 유닛 수준 대비: D_u = mean(event z) − mean(control z) (각각 1/99 윈저화 후), 팀 × 경기 교차 랜덤 절편(단일 그룹 + vc)
        e = ev.reset_index(); p = pl.reset_index()
        for fr in (e, p):
            lo, hi = np.percentile(fr.z, [1, 99]); fr["y"] = fr.z.clip(lo, hi)
        u = e.groupby("unit_id").agg(ye=("y", "mean"), team=("team", "first"), match_id=("match_id", "first"), ne=("y", "size")).join(p.groupby("unit_id").y.mean().rename("yc"))
        u["D"] = u.ye - u.yc; u = u.reset_index(); u["one"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                md = smf.mixedlm("D ~ 1", u, groups=u["one"], vc_formula={"team": "0 + C(team)", "match": "0 + C(match_id)"}); r = md.fit(reml=True, method=["lbfgs", "cg"], maxiter=1000)
                vc_names = list(md.exog_vc.names) if hasattr(md, "exog_vc") and hasattr(md.exog_vc, "names") else [f"vc{i}" for i in range(len(np.atleast_1d(r.vcomp)))]
                res["crossed_unit_level"] = {"intercept": round(float(r.params["Intercept"]), 4), "se": round(float(r.bse["Intercept"]), 4), "p": float(r.pvalues["Intercept"]),
                                             "var_components": {k: round(float(v), 5) for k, v in zip(vc_names, np.atleast_1d(r.vcomp))}, "resid_var": round(float(r.scale), 5), "converged": bool(r.converged),
                                             "n_units": int(len(u)), "n_teams": int(u.team.nunique()), "n_matches": int(u.match_id.nunique()),
                                             "definition": "unit-level contrast D_u = mean event net z − mean control net z (own cell), crossed random intercepts for team and match"}
            except Exception as ex:  # noqa: BLE001
                res["crossed_unit_level"] = {"error": str(ex)}
        # 유닛 수준 대비의 단순 평균과 두 방향 군집 SE
        gr = np.column_stack([pd.factorize(u.team)[0], pd.factorize(u.match_id)[0]])
        f2 = sm.OLS(u.D, u[["one"]]).fit(cov_type="cluster", cov_kwds={"groups": gr})
        res["crossed_unit_level"]["mean_D"] = round(float(u.D.mean()), 4); res["crossed_unit_level"]["se_twoway_team_match"] = round(float(f2.bse.iloc[0]), 4)
        log(side, "crossed unit-level", res["crossed_unit_level"])
        # 창 수준 두 방향 군집 SE (유닛 고정효과)
        for col in ("z", "zp"):
            e = ev.reset_index().copy(); p = pl.reset_index().copy()
            for fr in (e, p):
                lo, hi = np.percentile(fr[col], [1, 99]); fr["y"] = fr[col].clip(lo, hi)
            e["event"] = 1.0; p["event"] = 0.0; d = pd.concat([e, p], ignore_index=True); g = d.groupby("unit_id")
            d["y_dm"] = d.y - g.y.transform("mean"); d["ev_dm"] = d.event - g.event.transform("mean")
            groups = np.column_stack([pd.factorize(d.team)[0], pd.factorize(d.match_id)[0]])
            fit = sm.OLS(d.y_dm, d[["ev_dm"]]).fit(cov_type="cluster", cov_kwds={"groups": groups}); fit1 = sm.OLS(d.y_dm, d[["ev_dm"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.unit_id)[0]})
            res[f"twoway_{col}"] = {"b_event": round(float(fit.params.iloc[0]), 4), "se_twoway_team_match": round(float(fit.bse.iloc[0]), 4), "ci95_twoway": [round(float(x), 4) for x in fit.conf_int().iloc[0]],
                                    "p_twoway": float(fit.pvalues.iloc[0]), "se_unit_cluster": round(float(fit1.bse.iloc[0]), 4), "n": int(len(d)), "n_teams": int(d.team.nunique()), "n_matches": int(d.match_id.nunique())}
        put(f"{side}|depend", res); log(side, "depend", {k: v.get("ci95") for k, v in res.items() if isinstance(v, dict) and "ci95" in v})


def part_crossed(data):
    """유닛 수준 대비의 교차 랜덤효과 모형만 다시 적합해 depend 결과에 넣는다(part_depend 의 코드와 동일)."""
    import statsmodels.api as sm, statsmodels.formula.api as smf
    for side, df in data.items():
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR); res = RES.get(f"{side}|depend", {})
        e = ev.reset_index(); p = pl.reset_index()
        for fr in (e, p):
            lo, hi = np.percentile(fr.z, [1, 99]); fr["y"] = fr.z.clip(lo, hi)
        u = e.groupby("unit_id").agg(ye=("y", "mean"), team=("team", "first"), match_id=("match_id", "first"), ne=("y", "size")).join(p.groupby("unit_id").y.mean().rename("yc"))
        u["D"] = u.ye - u.yc; u = u.reset_index(); u["one"] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("D ~ 1", u, groups=u["one"], vc_formula={"team": "0 + C(team)", "match": "0 + C(match_id)"}); r = md.fit(reml=True, method=["lbfgs", "cg"], maxiter=1000)
        vc_names = list(md.exog_vc.names) if hasattr(md, "exog_vc") and hasattr(md.exog_vc, "names") else [f"vc{i}" for i in range(len(np.atleast_1d(r.vcomp)))]
        gr = np.column_stack([pd.factorize(u.team)[0], pd.factorize(u.match_id)[0]]); f2 = sm.OLS(u.D, u[["one"]]).fit(cov_type="cluster", cov_kwds={"groups": gr})
        res["crossed_unit_level"] = {"intercept": round(float(r.params["Intercept"]), 4), "se": round(float(r.bse["Intercept"]), 4), "p": float(r.pvalues["Intercept"]),
                                     "var_components": {k: round(float(v), 5) for k, v in zip(vc_names, np.atleast_1d(r.vcomp))}, "resid_var": round(float(r.scale), 5), "converged": bool(r.converged),
                                     "n_units": int(len(u)), "n_teams": int(u.team.nunique()), "n_matches": int(u.match_id.nunique()), "mean_D": round(float(u.D.mean()), 4), "se_twoway_team_match": round(float(f2.bse.iloc[0]), 4),
                                     "definition": "unit-level contrast D_u = mean event net z − mean control net z (own cell), crossed random intercepts for team and match"}
        put(f"{side}|depend", res); log(side, "crossed unit-level", res["crossed_unit_level"])


def part_sens(data):
    """민감도: 최소 셀 3/6/8/10, 비중첩 통제, 15분 내 추가 골 없는 공통 골(세 창), 같은 골의 참조/주, 전체 길이 창, 퇴장 규칙, 통제 범위."""
    for side, df in data.items():
        res = {}
        for mn in (3, 6, 8, 10):
            S = Standardised(df, "state", min_n=mn); ev, pl = frames(S, ANCHOR); res[f"min_cell_{mn}"] = both(ev, pl, n=2000)
        S = Standardised(df, "state", control_filter=lambda b: (b.t_start.astype(float) % 360 == 0)); ev, pl = frames(S, ANCHOR)
        res["nonoverlap_controls"] = both(ev, pl, n=2000)
        Sr = Standardised(df, "pre", control_filter=lambda b: (b.t_start.astype(float) % 360 == 0)); evr, plr = frames(Sr, ANCHOR); res["nonoverlap_controls_reference"] = both(evr, plr, n=2000)
        # 공통 골 표본: 15분 안에 다른 골이 없는 골 (세 창 모두 단일 골 노출)
        S = Standardised(df, "state"); fr = {w: frames(S, w) for w in WINDOWS}
        ev10 = fr[ANCHOR][0]; common_ids = ev10[ev10.next_goal_dt >= 900].index
        common = fr["post5"][0].index.intersection(fr["post10"][0].index).intersection(fr["post15"][0].index).intersection(common_ids)
        res["goals_in_all_windows"] = {}
        for w in WINDOWS:
            ev_w = fr[w][0].loc[common]; pl_w = S.frame("control", cells=ev_w.reset_index()[["unit_id", "ctx"]].drop_duplicates())
            res["goals_in_all_windows"][w] = both(ev_w, pl_w, n=2000)
        res["goals_in_all_windows"]["n"] = int(len(common))
        Sr = Standardised(df, "pre"); evr, plr = frames(Sr, ANCHOR); common2 = evr.index.intersection(ev10.index)
        res["same_goals_ref_vs_prim"] = {"n": int(len(common2)), "ref": both(evr.loc[common2], plr, n=2000), "prim": both(ev10.loc[common2], fr[ANCHOR][1], n=2000)}
        full = ev10[ev10.window_dur.astype(float) >= 299.9]; res["full_length_windows"] = {**both(full, fr[ANCHOR][1], n=2000), "n_truncated": int(len(ev10) - len(full))}
        for lab, excl in (("after_dismissal", "after_dismissal"), ("no_exclusion", "none")):
            d2 = load(side, exclusion=excl); S2 = Standardised(d2, "state"); ev2, pl2 = frames(S2, ANCHOR); res[f"dismissal_{lab}"] = both(ev2, pl2, n=2000)
        put(f"{side}|sens", res); log(side, "sens", {k: (v["unit"]["d"] if "unit" in v else "...") for k, v in res.items()})


def part_standard(data):
    """표준화 진단: 통제 창 수 밴드, 기저선 변동 삼분위, LOO·블록 LOO 통제, 셀 내 짝지은 차이, clipping × 분모 격자, 상태 내 시간 추세(사건별 실제 거리)."""
    import statsmodels.formula.api as smf
    BANDS = [("3-4", 3, 4), ("5-8", 5, 8), ("9-15", 9, 15), ("16+", 16, 10 ** 9)]
    for side, df in data.items():
        res = {}
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR)
        res["bands_state"] = {lab: {"n": int(((ev.bn >= lo) & (ev.bn <= hi)).sum()), "mean_z": round(float(ev[(ev.bn >= lo) & (ev.bn <= hi)].z.mean()), 3),
                                    "median_z": round(float(ev[(ev.bn >= lo) & (ev.bn <= hi)].z.median()), 3)} for lab, lo, hi in BANDS}
        res["bands_state"]["control_median_z"] = round(float(pl.z.median()), 3)
        res["bands_state"]["tail_below_minus5_pct"] = [round(float((ev.z < -5).mean() * 100), 3), round(float((pl.z < -5).mean() * 100), 4)]
        cuts = np.quantile(ev.rel, [1 / 3, 2 / 3]); terc = {}
        for lab, lo, hi in (("low", -np.inf, cuts[0]), ("mid", cuts[0], cuts[1]), ("high", cuts[1], np.inf)):
            e = ev[(ev.rel > lo) & (ev.rel <= hi)]; c = pl[(pl.rel > lo) & (pl.rel <= hi)]
            terc[lab] = {"n": int(len(e)), "unit": estimate(e.z.to_numpy(), c.z.to_numpy(), n=1000), "pooled": estimate(e.zp.to_numpy(), c.zp.to_numpy(), n=1000)}
        cells = S.cells.merge(ev.reset_index()[["unit_id", "ctx", "rel"]].drop_duplicates(), on=["unit_id", "ctx"])
        for lab, lo, hi in (("low", -np.inf, cuts[0]), ("mid", cuts[0], cuts[1]), ("high", cuts[1], np.inf)):
            cc = cells[(cells.rel > lo) & (cells.rel <= hi)]; terc[lab]["baseline_mean_pct"] = {m: round(float(cc[cc.metric == m].bm.mean() * 100), 2) for m in EXEC}
        res["sd_terciles"] = {"cuts": [round(float(c), 3) for c in cuts], **terc}
        # LOO 통제(사건 셀)
        _, pl_loo = frames(S, ANCHOR, loo=True)
        res["loo_controls"] = {**both(ev, pl_loo, n=2000), "control_mean_z_loo": round(float(wins(pl_loo.z.to_numpy()).mean()), 4), "control_mean_z_insample": round(float(wins(pl.z.to_numpy()).mean()), 4)}
        # 블록 LOO: 통제 창과 겹치는 이웃 창(시작 시각 차 < 300 s, 같은 셀)까지 기저선에서 뺀 표준화
        res["block_loo_controls"] = block_loo(df, ev, S)
        # 셀 내 짝지은 차이(사건 z − 같은 셀 LOO 통제 평균), 유닛 클러스터 부트스트랩
        cell = pl_loo.reset_index().groupby(["unit_id", "ctx"]).agg(cz=("z", "mean"), czp=("zp", "mean")).reset_index()
        e2 = ev.reset_index().merge(cell, on=["unit_id", "ctx"]); rng = np.random.default_rng(9); pr = {}
        for col, cc in (("z", "cz"), ("zp", "czp")):
            dz = (e2[col] - e2[cc]).to_numpy(); lo, hi = np.percentile(dz, [1, 99]); dz = np.clip(dz, lo, hi)
            units = e2.unit_id.to_numpy(); uu = np.unique(units); idx = {u: np.where(units == u)[0] for u in uu}
            bs = [dz[np.concatenate([idx[u] for u in rng.choice(uu, len(uu))])].mean() for _ in range(2000)]
            pr[col] = {"mean": round(float(dz.mean()), 4), "ci95_unit_cluster": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "d_z": round(float(dz.mean() / dz.std(ddof=1)), 4), "n": int(len(dz))}
        res["paired_within_cell"] = pr
        # 사건별 셀 평균 대비(표본 내): 사건 z − 셀 통제 평균 z (정의상 통제 평균 ≈ 0)
        cell_in = pl.reset_index().groupby(["unit_id", "ctx"]).agg(cz=("z", "mean")).reset_index(); e3 = ev.reset_index().merge(cell_in, on=["unit_id", "ctx"])
        res["paired_within_cell_insample"] = {"mean": round(float((e3.z - e3.cz).mean()), 4), "mean_control_z_in_event_cells": round(float(e3.cz.mean()), 6), "n": int(len(e3))}
        # clipping × 분모 격자(같은 사건·통제 집합)
        grid = {}
        for cl, fn in (("none", d_raw), ("separate_1_99", d_sep), ("joint_1_99", d_joint), ("separate_5_95", lambda a, b: d_sep(a, b, 5, 95))):
            grid[cl] = {"unit": round(fn(ev.z.to_numpy(), pl.z.to_numpy()), 4), "pooled": round(fn(ev.zp.to_numpy(), pl.zp.to_numpy()), 4),
                        "unit_loo": round(fn(ev.z.to_numpy(), pl_loo.z.to_numpy()), 4)}
        res["clip_denominator_grid"] = grid
        # 상태 내 시간 추세: 사건 셀의 통제 창 풀링 z ~ (통제 창 시작 − 사건 창 시작)/10분, 유닛 랜덤 절편; 사건별 실제 거리로 예측
        c = pl.reset_index().copy(); e_start = ev.reset_index()[["unit_id", "ctx", "elapsed"]].rename(columns={"elapsed": "ev_start"})
        c = c.merge(e_start, on=["unit_id", "ctx"]); c["rel10"] = (c.elapsed.astype(float) - c.ev_start.astype(float)) / 600.0
        lo, hi = np.percentile(c.zp, [1, 99]); c["y"] = c.zp.clip(lo, hi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); r = smf.mixedlm("y ~ rel10", c, groups=c["unit_id"]).fit(reml=True)
        slope = float(r.params["rel10"]); mean_rel = float(c.rel10.mean()); zp_sd = float(c.y.std(ddof=1)); gap = -slope * mean_rel
        res["time_trend"] = {"slope_per_10min": round(slope, 5), "se": round(float(r.bse["rel10"]), 5), "p": float(r.pvalues["rel10"]), "converged": bool(r.converged),
                             "mean_control_distance_10min": round(mean_rel, 3), "min_control_distance_10min": round(float(c.rel10.min()), 3),
                             "predicted_event_minus_control_zp": round(gap, 4), "zp_sd_control": round(zp_sd, 4), "predicted_gap_d": round(gap / zp_sd, 4), "n": int(len(c)),
                             "calculation": "controls of the event cells; distance = control window start − event window start (10-min units, measured per event); predicted gap = −slope × mean distance; d = gap / SD of control pooled z"}
        put(f"{side}|standard", res); log(side, "standard", {"loo": res["loo_controls"]["unit"]["d"], "block_loo": res["block_loo_controls"]["unit"]["d"], "grid": grid["joint_1_99"], "trend_d": res["time_trend"]["predicted_gap_d"]})


def block_loo(df, ev, S):
    """블록 LOO: 각 통제 창을 같은 셀에서 자신과 시작 시각 차이가 300 s 미만인(겹치는) 창들과 함께 뺀 기저선으로 표준화."""
    base = S.base.merge(ev.reset_index()[["unit_id", "ctx"]].drop_duplicates(), on=["unit_id", "ctx"], how="inner")
    base = base.merge(S.cells, on=["unit_id", "metric", "ctx"], how="inner")
    base["t0"] = base.elapsed.astype(float)
    rows = []
    for (u, m, c), g in base.groupby(["unit_id", "metric", "ctx"]):
        v = g.value.to_numpy(); t = g.t0.to_numpy(); n = len(v)
        for i in range(n):
            keep = np.abs(t - t[i]) >= 300
            if keep.sum() < 2:
                rows.append((u, g.event_id.iloc[i], m, np.nan)); continue
            mu = v[keep].mean(); sd = v[keep].std(ddof=1)
            rows.append((u, g.event_id.iloc[i], m, DIR[m] * (v[i] - mu) / sd if sd > 0 else np.nan))
    z = pd.DataFrame(rows, columns=["unit_id", "event_id", "metric", "z"]).dropna()
    zz = z.groupby(["unit_id", "event_id"]).z.mean()
    return {"unit": estimate(ev.z.to_numpy(), zz.to_numpy(), n=2000), "n_control": int(len(zz)), "definition": "control standardised against its cell's baseline after removing itself and every control window whose start lies within 300 s (overlapping neighbours)"}


def part_indicators(data):
    for side, df in data.items():
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR); cols = [f"z_{m}" for m in EXEC]
        res = {"event_pearson": ev[cols].corr().round(3).to_dict(), "event_spearman": ev[cols].corr("spearman").round(3).to_dict(),
               "control_pearson": pl[cols].corr().round(3).to_dict(), "control_spearman": pl[cols].corr("spearman").round(3).to_dict(),
               "n_event_k3": int((ev.k == 3).sum()), "n_control_k3": int((pl.k == 3).sum())}
        mats = [ev[cols].corr(), ev[cols].corr("spearman"), pl[cols].corr(), pl[cols].corr("spearman")]
        res["max_abs_r"] = round(float(max(abs(m_.loc[a, b]) for m_ in mats for a in cols for b in cols if a != b)), 3)
        put(f"{side}|indicators", res); log(side, "indicators max|r|", res["max_abs_r"])


def part_nearest(data):
    """시간 근접 대조창(사건 셀 안, 같은 period). 5–10분 사건 창이 끝난 뒤 같은 period 에 남은 통제창을 near(사건 창 종료 후 ≤ 10분 시작)와
    far(그 뒤)로 나누고, near 와 far 통제가 모두 있는 사건으로 한정한다(공통 사건). 사건별 own-cell 차이(사건 z − near 평균, 사건 z − far 평균)와
    unit-cluster 부트스트랩 구간, 그리고 같은 공통 사건에 대한 d. 거리는 사건 창 종료부터 통제창 시작까지(초, period 안; 하프타임 제외).
    15분 제외 규칙 때문에 같은 period 의 첫 통제창은 골 뒤 ≥ 15분(= 사건 창 종료 뒤 ≥ 5분)에 시작한다."""
    rng = np.random.default_rng(7)
    for side, df in data.items():
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR)
        e = ev.reset_index(); p = pl.reset_index()
        e["ev_end"] = e.elapsed.astype(float) + e.window_dur.astype(float)
        m = p.merge(e[["unit_id", "ctx", "event_id", "ev_end", "period"]].rename(columns={"event_id": "ev_id", "period": "ev_period"}), on=["unit_id", "ctx"])
        m = m[m.period.astype(int) == m.ev_period.astype(int)]        # same period only (period, not half: 'half' pools both extra-time periods as ET)
        m["dist"] = m.elapsed.astype(float) - m.ev_end                 # seconds from the end of the event window to the control start
        m = m[m.dist >= 0]
        m["near"] = m.dist <= 600
        agg = m.groupby(["unit_id", "ev_id"]).agg(z_near=("z", lambda x: x[m.loc[x.index, "near"]].mean()), z_far=("z", lambda x: x[~m.loc[x.index, "near"]].mean()),
                                                  zp_near=("zp", lambda x: x[m.loc[x.index, "near"]].mean()), zp_far=("zp", lambda x: x[~m.loc[x.index, "near"]].mean()),
                                                  n_near=("near", "sum"), n_far=("near", lambda x: int((~x).sum())),
                                                  dist_near=("dist", lambda x: x[m.loc[x.index, "near"]].mean()), dist_far=("dist", lambda x: x[~m.loc[x.index, "near"]].mean()))
        both_ = agg[(agg.n_near > 0) & (agg.n_far > 0)].reset_index()
        ee = e.merge(both_, left_on=["unit_id", "event_id"], right_on=["unit_id", "ev_id"])
        res = {"n_events_total": int(len(e)), "n_events_with_same_period_controls": int(agg.shape[0]), "n_events_common": int(len(ee)),
               "n_controls_near": int(both_.n_near.sum()), "n_controls_far": int(both_.n_far.sum()),
               "distance_near_min": _q(both_.dist_near / 60), "distance_far_min": _q(both_.dist_far / 60),
               "definition": "controls of the event's own cell in the same period, starting after the 5–10 min event window; near = start within 10 min after the event window end, far = later; events restricted to those with at least one near and one far control; distances in minutes of period clock (half-time not counted); per-event differences = event net z − mean control net z"}
        units = ee.unit_id.to_numpy(); uu = np.unique(units); idx = {u: np.where(units == u)[0] for u in uu}
        for col in ("z", "zp"):
            ez = ee[col].to_numpy(); lo, hi = np.percentile(ez, [1, 99]); ez = np.clip(ez, lo, hi)
            out = {}
            for lab in ("near", "far"):
                cz = ee[f"{col}_{lab}"].to_numpy(); diff = ez - cz
                bs = np.array([diff[np.concatenate([idx[u] for u in rng.choice(uu, len(uu))])].mean() for _ in range(2000)])
                out[f"diff_{lab}"] = {"mean": round(float(diff.mean()), 4), "ci95_unit_cluster": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
            dd = (ee[f"{col}_far"] - ee[f"{col}_near"]).to_numpy()   # far − near control mean per event (positive = later controls higher)
            bs = np.array([dd[np.concatenate([idx[u] for u in rng.choice(uu, len(uu))])].mean() for _ in range(2000)])
            out["control_far_minus_near"] = {"mean": round(float(dd.mean()), 4), "ci95_unit_cluster": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
            # Cohen's d on the common events against their near and their far controls
            sub_ids = set(zip(ee.unit_id, ee.ev_id))
            mm = m[[(u, i) in sub_ids for u, i in zip(m.unit_id, m.ev_id)]]
            for lab, mask in (("near", mm.near), ("far", ~mm.near)):
                pz = mm.loc[mask, col].to_numpy()
                out[f"d_{lab}"] = round(float(d_sep(ee[col].to_numpy(), pz)), 4)
            out["n_events"] = int(len(ee))
            res[col] = out
        log(side, "nearest", res["n_events_common"], res["z"]["diff_near"], res["z"]["diff_far"], res["zp"]["diff_near"], res["zp"]["diff_far"])
        put(f"{side}|nearest", res)


def _q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {"median": round(float(np.median(x)), 1), "p25": round(float(np.percentile(x, 25)), 1), "p75": round(float(np.percentile(x, 75)), 1), "min": round(float(x.min()), 1), "max": round(float(x.max()), 1)} if len(x) else {}


def _exposure_mask(full: pd.DataFrame) -> pd.Series:
    """parquet 직접 적재용: 골 원장으로 단일 골 노출 사건 창을 표시(다른 골이 창 종료 전)."""
    led = pd.read_parquet(ROOT / "data/interim/goal_ledger.parquet")[["match_id", "goal_index", "next_goal_dt"]]
    ctx = pd.DataFrame(full.context.tolist(), index=full.index)
    mid = full.unit_id.str.rsplit("-", n=1).str[-1]; gi = pd.to_numeric(ctx.get("goal_index"), errors="coerce")
    key = pd.DataFrame({"match_id": mid, "goal_index": gi}).merge(led, on=["match_id", "goal_index"], how="left")
    hi = full.window.map(HI).to_numpy(); other = (key.next_goal_dt.to_numpy() < hi) & full.window.str.startswith("post").to_numpy()
    return pd.Series(other, index=full.index)


def part_binom(data):
    """원 단위 모형: 창 수준 이항 GLM(성공/시도)·포아송(미스컨트롤/터치, 파울/수비 행동), (unit, state) 셀 고정효과(사건이 있는 셀), 유닛 클러스터 강건 SE."""
    from glum import GeneralizedLinearRegressor
    import scipy.sparse as sp
    specs = {"short_pass": ("sp_completed", "sp_attempts", "binomial"), "miscontrol": ("miscontrols", "touches", "poisson"), "foul": ("fouls", "def_actions", "poisson")}
    for side in ("score", "concede"):
        full = pd.read_parquet(ROOT / "data/features" / FILES[side])
        ctx = pd.DataFrame(full.context.tolist()); full = full.assign(score_state=ctx.score_state.values, dism=ctx.dismissal_in_window.astype(bool).values)
        full = full[~full.dism & ~_exposure_mask(full).to_numpy()]
        wide = full.pivot_table(index=["unit_id", "event_id", "window", "score_state"], columns="metric", values="value", aggfunc="first").reset_index()
        wide["is_post"] = wide.window.str.startswith("post"); res = {}
        for lab, (num, den, fam) in specs.items():
            d = wide[[num, den, "unit_id", "event_id", "window", "score_state", "is_post"]].dropna(); d = d[d[den] > 0]; d["cell"] = d.unit_id + "|" + d.score_state
            ctrl_n = d[~d.is_post].groupby("cell").size(); ev_cells = set(d[(d.is_post) & (d.window == ANCHOR)].cell); keep = set(ctrl_n[ctrl_n >= MIN_N].index) & ev_cells
            dd = d[d.cell.isin(keep) & ((~d.is_post) | (d.window == ANCHOR))].copy()
            cells = pd.Categorical(dd.cell); X_cell = sp.csr_matrix((np.ones(len(dd)), (np.arange(len(dd)), cells.codes)), shape=(len(dd), len(cells.categories)))
            X = sp.hstack([sp.csr_matrix(dd.is_post.astype(float).to_numpy()[:, None]), X_cell]).tocsr(); cl = pd.factorize(dd.unit_id)[0]
            if fam == "binomial":
                y = (dd[num] / dd[den]).to_numpy(); w = dd[den].to_numpy(); model = GeneralizedLinearRegressor(family="binomial", alpha=0, fit_intercept=False, max_iter=500); model.fit(X, y, sample_weight=w)
                beta = float(model.coef_[0]); se = float(model.std_errors(X=X, y=y, sample_weight=w, robust=True, clusters=cl)[0])
                p0 = float((dd[~dd.is_post][num].sum() / dd[~dd.is_post][den].sum())); pp = 100 * (1 / (1 + np.exp(-(np.log(p0 / (1 - p0)) + beta))) - p0)
                res[lab] = {"family": "binomial", "log_odds_event": round(beta, 4), "se_cluster_unit": round(se, 4), "odds_ratio": round(float(np.exp(beta)), 4),
                            "or_ci95": [round(float(np.exp(beta - 1.96 * se)), 4), round(float(np.exp(beta + 1.96 * se)), 4)], "control_rate_pct": round(p0 * 100, 2),
                            "marginal_pp_at_control_rate": round(float(pp), 3), "n_windows": int(len(dd)), "n_cells": int(len(cells.categories)), "n_event_windows": int(dd.is_post.sum())}
            else:
                y = dd[num].to_numpy(); off = np.log(dd[den].to_numpy()); model = GeneralizedLinearRegressor(family="poisson", alpha=0, fit_intercept=False, max_iter=500); model.fit(X, y, offset=off)
                beta = float(model.coef_[0]); se = float(model.std_errors(X=X, y=y, offset=off, robust=True, clusters=cl)[0]); r0 = float(dd[~dd.is_post][num].sum() / dd[~dd.is_post][den].sum())
                res[lab] = {"family": "poisson", "log_rate_ratio_event": round(beta, 4), "se_cluster_unit": round(se, 4), "rate_ratio": round(float(np.exp(beta)), 4),
                            "rr_ci95": [round(float(np.exp(beta - 1.96 * se)), 4), round(float(np.exp(beta + 1.96 * se)), 4)], "control_rate_pct": round(r0 * 100, 2),
                            "marginal_pp_at_control_rate": round(float(100 * r0 * (np.exp(beta) - 1)), 3), "n_windows": int(len(dd)), "n_cells": int(len(cells.categories)), "n_event_windows": int(dd.is_post.sum())}
            log(side, "glm", lab, res[lab])
        put(f"{side}|binom", res)


def part_descriptors(data):
    """측정 변수(이름은 계산 그대로): 패스 수, 팀 패스 비중(팀 패스/전체 패스), 30야드 이상 패스 비중, 온볼 행동 평균 x, 수비 행동 평균 x, 압박 수, 패스 전진 거리, 단거리 패스 시도.
    사건 − 사건 셀 통제 평균(원 단위), 유닛 클러스터 부트스트랩."""
    DESC = ["pass_volume", "possession_share", "long_pass_share", "action_x_mean", "def_action_x_mean", "pressure_freq", "pass_progression", "sp_attempts"]
    rng = np.random.default_rng(13)
    for side in ("score", "concede"):
        full = pd.read_parquet(ROOT / "data/features" / FILES[side]); full = full[full.metric.isin(DESC)].copy()
        ctx = pd.DataFrame(full.context.tolist()); full["score_state"] = ctx.score_state.values; full["dism"] = ctx.dismissal_in_window.astype(bool).values
        full = full[~full.dism & ~_exposure_mask(full).to_numpy()]; full["is_post"] = full.window.str.startswith("post"); res = {}
        for w in WINDOWS:
            res[w] = {}
            for m in DESC:
                d = full[full.metric == m]; base = d[~d.is_post]; b = base.groupby(["unit_id", "score_state"])["value"].agg(bm="mean", bn="count").reset_index(); b = b[b.bn >= MIN_N]
                e = d[(d.is_post) & (d.window == w)].merge(b, on=["unit_id", "score_state"]); de = (e.value - e.bm).to_numpy()
                units = e.unit_id.to_numpy(); uu = np.unique(units); idx = {u: np.where(units == u)[0] for u in uu}
                bs = [de[np.concatenate([idx[u] for u in rng.choice(uu, len(uu))])].mean() for _ in range(1000)]
                res[w][m] = {"diff": round(float(de.mean()), 4), "ci95_unit_cluster": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                             "control_mean": round(float(e.bm.mean()), 4), "event_mean": round(float(e.value.mean()), 4), "n": int(len(de)), "d_z": round(float(de.mean() / de.std(ddof=1)), 4)}
        put(f"{side}|descriptors", res); log(side, "descriptors", {m: res[ANCHOR][m]["diff"] for m in DESC})


def part_clipgrid(data):
    """R2 · clipping × 분모 격자 재계산(part_standard 와 같은 사건·통제 집합)에 계획서 규칙(사건 창만 1/99 윈저화, 통제 창은 그대로)을 추가한다."""
    for side, df in data.items():
        S = Standardised(df, "state"); ev, pl = frames(S, ANCHOR); _, pl_loo = frames(S, ANCHOR, loo=True)
        d_event_only = lambda a, b: cohen_d(wins(a, 1, 99), np.asarray(b, float)[~np.isnan(np.asarray(b, float))])
        grid = {}
        for cl, fn in (("none", d_raw), ("event_only_1_99", d_event_only), ("separate_1_99", d_sep), ("joint_1_99", d_joint), ("separate_5_95", lambda a, b: d_sep(a, b, 5, 95))):
            grid[cl] = {"unit": round(fn(ev.z.to_numpy(), pl.z.to_numpy()), 4), "pooled": round(fn(ev.zp.to_numpy(), pl.zp.to_numpy()), 4),
                        "unit_loo": round(fn(ev.z.to_numpy(), pl_loo.z.to_numpy()), 4)}
        # 다음 피리어드의 통제 창: 15분 제외 규칙이 피리어드 안에서만 적용되므로, 사건 셀의 통제 창 중 사건보다 뒤 피리어드에 있는 비중과
        # 그 가운데 사건 창 시작 후 15분(경기 시간) 안에 시작하는 비중을 기록한다(사건 × 통제 창 쌍 기준; 셀의 통제는 그 셀의 모든 사건과 짝지음).
        e = ev.reset_index()[["unit_id", "ctx", "period", "elapsed"]].rename(columns={"period": "ev_period", "elapsed": "ev_start"})
        c = pl.reset_index()[["unit_id", "ctx", "period", "elapsed"]]
        pairs = e.merge(c, on=["unit_id", "ctx"])
        later = pairs.period.astype(int) > pairs.ev_period.astype(int)
        within15 = later & ((pairs.elapsed - pairs.ev_start) < 900)
        lp = {"n_event_control_pairs": int(len(pairs)), "pct_control_in_later_period": round(float(later.mean() * 100), 2),
              "pct_control_in_later_period_within_15min_of_event_window": round(float(within15.mean() * 100), 2),
              "pct_event_cells_with_any_later_period_control": round(float(pairs.assign(l=later).groupby(["unit_id", "ctx"]).l.any().mean() * 100), 2),
              "n_control_windows_unique": int(len(c)), "pct_control_windows_in_period_2_or_later": round(float((c.period.astype(int) >= 2).mean() * 100), 2)}
        put(f"{side}|standard", {**RES.get(f"{side}|standard", {}), "clip_denominator_grid": grid, "later_period_controls": lp})
        log(side, "clipgrid", {k: v["unit"] for k, v in grid.items()}, "later-period", lp)


def part_indpooled(data):
    """R2 · 지표별 효과를 세 척도에서 나란히: within-unit z 의 d, pooled-denominator z 의 d, 사건·통제 중앙값 차(within-unit z).
    Results 의 '원 단위는 개선, within-unit 지표는 음수' 설명용(리뷰어 2, 포인트 4). 사건 셀 통제, 각각 1/99 윈저화, 유닛 클러스터 부트스트랩 2,000."""
    for side, df in data.items():
        S = Standardised(df, "state"); res = {}
        for w in WINDOWS:
            ev, pl = frames(S, w); res[w] = {}
            for m in EXEC:
                e_u = ev[f"z_{m}"]; p_u = pl[f"z_{m}"]
                # 지표별 pooled z: (value − cell mean) / 지표의 풀링 SD = z_unit × (cell SD / pooled SD); frame 은 지표별 bs 를 평균 rel 로만 가지므로 다시 계산
                ev_m = S.post[(S.post.window == w) & (S.post.metric == m)].merge(S.cells, on=["unit_id", "metric", "ctx"]).set_index(["unit_id", "event_id"])
                cells = ev.reset_index()[["unit_id", "ctx"]].drop_duplicates()
                pl_m = S.base[S.base.metric == m].merge(cells, on=["unit_id", "ctx"]).merge(S.cells, on=["unit_id", "metric", "ctx"]).set_index(["unit_id", "event_id"])
                sgn = DIR[m]; sp = S.pooled[m]
                e_p = sgn * (ev_m.value - ev_m.bm) / sp; p_p = sgn * (pl_m.value - pl_m.bm) / sp
                e_p = e_p[ev_m.bs.replace(0, np.nan).notna()]; p_p = p_p[pl_m.bs.replace(0, np.nan).notna()]
                evu = pd.DataFrame({"z": e_u.dropna()}); plu = pd.DataFrame({"z": p_u.dropna()})
                evp = pd.DataFrame({"z": e_p.dropna()}); plp = pd.DataFrame({"z": p_p.dropna()})
                cu = cluster_boot_d(evu, plu, "unit_id", n=2000); cpd = cluster_boot_d(evp, plp, "unit_id", n=2000)
                res[w][m] = {"d_unit": cu["d"], "d_unit_ci95": cu["ci95"], "d_pooled": cpd["d"], "d_pooled_ci95": cpd["ci95"],
                             "median_diff_unit_z": round(float(np.median(evu.z) - np.median(plu.z)), 4), "mean_diff_unit_z": round(float(evu.z.mean() - plu.z.mean()), 4),
                             "mean_diff_pooled_z": round(float(evp.z.mean() - plp.z.mean()), 4),
                             "event_tail_below_minus3_pct": round(float((evu.z < -3).mean() * 100), 2), "control_tail_below_minus3_pct": round(float((plu.z < -3).mean() * 100), 2),
                             "n_event": int(len(evu)), "n_control": int(len(plu))}
            log(side, "indpooled", w, {m: (res[w][m]["d_unit"], res[w][m]["d_pooled"]) for m in EXEC})
        put(f"{side}|indicators_pooled", res)


def part_mdes(data):
    from scipy.stats import norm
    z = norm.ppf(1 - 0.005) + norm.ppf(0.90)
    for side in ("score", "concede"):
        dep = RES.get(f"{side}|depend", {}); res = {}
        for c in ("unit_id", "match_id", "team"):
            if c in dep:
                res[f"mdes_{c}"] = round(float(z * dep[c]["se"]), 4)
        m = RES.get(f"{side}|state|{ANCHOR}")
        if m:
            ne, nc = m["unit"]["n_event"], m["unit"]["n_control"]; res["mdes_iid_formula"] = round(float(z * np.sqrt(1 / ne + 1 / nc)), 4); res["n_event"] = ne; res["n_control"] = nc
        res["z_multiplier"] = round(float(z), 3); put(f"{side}|mdes", res); log(side, "mdes", res)


def part_clusterw(data):
    """모든 창·매칭 키의 유닛 클러스터 추론(95%·99% CI, 단측 p; 유닛/풀링 분모) — 표 3·그림 2 의 구간. 앵커는 part_main 의 5,000 재표집 값을 유지."""
    for side, df in data.items():
        for key in ("pre", "state"):
            S = Standardised(df, key)
            for w in WINDOWS:
                if w == ANCHOR and f"{side}|{key}|{w}" in RES and "cluster_unit" in RES[f"{side}|{key}|{w}"]:
                    continue
                ev, pl = frames(S, w)
                r = RES.get(f"{side}|{key}|{w}", {})
                r["cluster_unit"] = cluster_inference(ev, pl, "unit_id", n=2000); r["cluster_unit_pooled"] = cluster_inference(ev, pl, "unit_id", n=2000, col="zp")
                put(f"{side}|{key}|{w}", r); log(side, key, w, "cluster", r["cluster_unit"]["d"], r["cluster_unit"]["ci95"], "pooled", r["cluster_unit_pooled"]["d"], r["cluster_unit_pooled"]["ci95"])


def part_variant(name, features_dir):
    fd = Path(features_dir)
    for side in ("score", "concede"):
        df = load(side, fd, continuous_clock=(name == "v2_cont")); r = {}
        for key in ("pre", "state"):
            S = Standardised(df, key)
            for w in WINDOWS:
                ev, pl = frames(S, w); r[f"{key}|{w}"] = both(ev, pl, n=1000)
        put(f"{side}|variant|{name}", r); log(side, "variant", name, {k: v["unit"]["d"] for k, v in r.items()})


def merge(files):
    base = ROOT / "docs/ijpas_r2.json"; acc = json.loads(base.read_text()) if base.exists() else {}
    for f in files:
        part = json.loads(Path(f).read_text())
        for k, v in part.items():   # one-level deep merge: a part that adds sub-keys (e.g. clusterw) does not drop the others
            if isinstance(v, dict) and isinstance(acc.get(k), dict):
                acc[k] = {**acc[k], **v}
            else:
                acc[k] = v
    save_json(acc, base); print("✓ merged", len(acc), "keys →", base)


def main(argv):
    parts = argv or ["all"]
    if parts[0] == "merge":
        merge(parts[1:]); return
    if parts[0] == "variant":
        part_variant(parts[1], parts[2]); return
    data = {s: load(s) for s in ("score", "concede")}
    put("_meta", {**RES.get("_meta", {}), "generated": time.strftime("%Y-%m-%d %H:%M"), "python": platform.python_version(),
                  "versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "scipy", "statsmodels", "glum", "pyarrow")},
                  "rows": {s: int(len(d)) for s, d in data.items()}, "exclusion": "dismissal_window", "exposure": "single_goal", "control_scope": "event_cells",
                  "winsorisation": "separate 1/99 (event, control)", "min_cell": MIN_N, "anchor": ANCHOR})
    allp = ["main", "cells", "flow", "strata", "loco", "oppq", "paired", "depend", "crossed", "sens", "standard", "indicators", "nearest", "binom", "descriptors", "mdes", "clusterw"]
    for p in (allp if "all" in parts else parts):
        log("▶", p); globals()[f"part_{p}"](data)
    log("✓", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
