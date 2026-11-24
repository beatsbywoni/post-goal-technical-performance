"""R1 · 패스 수준 자료: 압박 없는 단거리 패스의 위치·방향 보정(expected completion) 과 패스 분포(수량·구역).

  python3 -m analysis.momentum.r1_passes extract <events_dir_or_"sb">   # 패스 수준 표 → data/interim/passes_unpressured_short.parquet
  python3 -m analysis.momentum.r1_passes analyse                          # 창 배정 + xPass 모형 + 사건/통제 비교 → docs/ijpas_r1_passes.json

events_dir: 경기별 statsbombpy 이벤트 프레임을 pickle(gzip) 로 저장한 폴더(<match_id>.pkl.gz). "sb" 를 주면 statsbombpy 로 받는다(Mac).
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.momentum.r1_lib import ANCHOR, FILES, MIN_N, WINDOWS, load, save_json  # noqa: E402
from src.football.adapter import PERIOD_START_MIN, SHOOTOUT_PERIOD  # noqa: E402

PASSES = ROOT / "data/interim/passes_unpressured_short.parquet"
OUT = ROOT / "docs/ijpas_r2_passes.json"


def extract(events_dir: str):
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/sb_competitions.yaml", encoding="utf-8"))
    comps = [c for tier in cfg["tiers"].values() for c in tier]
    rows = []
    if events_dir == "sb":
        from statsbombpy import sb
        warnings.filterwarnings("ignore")
        pairs = []
        for c in comps:
            m = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
            pairs += [(c["competition"], int(mid)) for mid in m.match_id]
        getter = lambda mid: sb.events(match_id=mid)
    else:
        ml = json.loads((ROOT / "data/interim/match_list.json").read_text())
        pairs = [(comp, mid) for comp, season, mid in ml]
        getter = lambda mid: pd.read_pickle(Path(events_dir) / f"{mid}.pkl.gz")
    t0 = time.time()
    for i, (comp, mid) in enumerate(pairs):
        ev = getter(mid)
        p = ev[(ev.type == "Pass")].copy()
        if "period" in p.columns:
            p = p[pd.to_numeric(p.period, errors="coerce").fillna(1).astype(int) != SHOOTOUT_PERIOD]
        p["pass_length"] = pd.to_numeric(p.pass_length, errors="coerce")
        p = p[(p.pass_length < 20) & (p.under_pressure != True)]  # noqa: E712
        if p.empty:
            continue
        loc = p.location.apply(lambda l: l if isinstance(l, (list, tuple, np.ndarray)) and len(l) >= 2 else [np.nan, np.nan])
        end = p.pass_end_location.apply(lambda l: l if isinstance(l, (list, tuple, np.ndarray)) and len(l) >= 2 else [np.nan, np.nan])
        per = pd.to_numeric(p.period, errors="coerce").fillna(1).astype(int) if "period" in p.columns else pd.Series(1, index=p.index)
        rows.append(pd.DataFrame({
            "match_id": str(mid), "competition": comp, "team": p.team.values, "period": per.values,
            "t_in": (p.minute.astype(float) * 60 + pd.to_numeric(p.second, errors="coerce").fillna(0) - per.map(PERIOD_START_MIN).fillna(0) * 60).values,
            "x": [l[0] for l in loc], "y": [l[1] for l in loc], "end_x": [l[0] for l in end], "end_y": [l[1] for l in end],
            "length": p.pass_length.values, "angle": pd.to_numeric(p.pass_angle, errors="coerce").values if "pass_angle" in p.columns else np.nan,
            "completed": p.pass_outcome.isna().values, "play_pattern": p.play_pattern.values if "play_pattern" in p.columns else None,
            "body_part": p.pass_body_part.values if "pass_body_part" in p.columns else None,
            "height": p.pass_height.values if "pass_height" in p.columns else None}))
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(pairs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(rows, ignore_index=True); PASSES.parent.mkdir(exist_ok=True, parents=True); df.to_parquet(PASSES, index=False)
    print("✓", PASSES, len(df), "passes")


def analyse():
    import statsmodels.api as sm
    passes = pd.read_parquet(PASSES)
    passes = passes.dropna(subset=["x", "y", "length"])
    res = {}
    for side in ("score", "concede"):
        df = load(side)   # 퇴장 창 제외 적용
        # 창 목록(사건 + 통제) with (match, team, period, t_start, dur)
        win = df[df.metric == "short_pass_acc"][["unit_id", "event_id", "window", "match_id", "team", "period", "t_start", "window_dur", "score_state", "is_post", "elapsed"]].drop_duplicates(["unit_id", "event_id", "window"])
        win["period"] = win.period.astype(int); win["t_start"] = win.t_start.astype(float); win["window_dur"] = win.window_dur.astype(float)
        # 패스에 창 배정: (match, team, period) 그룹별 searchsorted → (pass_idx, window_idx) 배열(메모리 절약)
        passes["_i"] = np.arange(len(passes))
        win = win.reset_index(drop=True)
        pg = {k: (g.t_in.to_numpy(), g._i.to_numpy()) for k, g in passes.sort_values("t_in").groupby(["match_id", "team", "period"])}
        starts, counts, widx = [], [], []
        for wi, r in enumerate(win.itertuples(index=False)):
            g = pg.get((r.match_id, r.team, r.period))
            if g is None:
                continue
            t, ii = g; i0, i1 = np.searchsorted(t, r.t_start, "left"), np.searchsorted(t, r.t_start + r.window_dur, "left")
            if i1 > i0:
                starts.append(ii[i0:i1]); widx.append(np.full(i1 - i0, wi))
        A = pd.DataFrame({"_i": np.concatenate(starts), "_w": np.concatenate(widx)})
        A = A.merge(win[["unit_id", "event_id", "window", "is_post", "score_state"]].reset_index().rename(columns={"index": "_w"}), on="_w").merge(passes, on="_i")
        del starts, widx
        # xPass 모형: 통제 창 패스로 적합(사건 창 제외), 위치·길이·방향·플레이 패턴
        A["dx"] = A.end_x - A.x; A["dy"] = A.end_y - A.y
        A["cos"] = np.cos(A.angle.fillna(0)); A["sin"] = np.sin(A.angle.fillna(0))
        A["non_regular_play"] = (~A.play_pattern.isin(["Regular Play"])).astype(float) if A.play_pattern.notna().any() else 0.0
        X = pd.DataFrame({"len": A.length, "len2": A.length ** 2, "x": A.x / 120, "x2": (A.x / 120) ** 2, "y": A.y / 80, "y2": (A.y / 80) ** 2, "yc": np.abs(A.y - 40) / 40,
                          "cos": A.cos, "sin": A.sin, "fwd": (A.dx > 0).astype(float), "non_regular_play": A.non_regular_play, "xcos": (A.x / 120) * A.cos})
        X = sm.add_constant(X)
        ctrl = ~A.is_post.astype(bool)
        # 고유 패스로 적합(중첩 통제 창에 여러 번 배정된 패스는 한 번만), 경기 단위 5-fold cross-fitting: 각 패스의 기대 성공률은
        # 그 패스의 경기를 포함하지 않는 폴드의 모형에서 예측(표본 외). 보고: 고유 패스 수, 배정 행 수, 표본 외 AUC, calibration(10분위).
        uniq = A[ctrl].drop_duplicates("_i"); n_unique_ctrl = int(len(uniq)); n_rows_ctrl = int(ctrl.sum())
        matches = np.sort(A.match_id.unique()); rng_f = np.random.default_rng(5); fold_of = dict(zip(matches, rng_f.integers(0, 5, len(matches))))
        A["fold"] = A.match_id.map(fold_of).astype(int); uniq_fold = uniq.match_id.map(fold_of).to_numpy()
        A["xp"] = np.nan; params_folds = []
        for f in range(5):
            tr = uniq[uniq_fold != f]; model_f = sm.Logit(tr.completed.astype(float), X.loc[tr.index]).fit(disp=0, maxiter=200)
            te = A.fold == f; A.loc[te, "xp"] = model_f.predict(X[te]); params_folds.append(model_f.params)
        model = sm.Logit(uniq.completed.astype(float), X.loc[uniq.index]).fit(disp=0, maxiter=200)   # 전체 고유 통제 패스 적합(계수 보고용)
        assert A.xp.notna().all()
        yc = uniq.completed.to_numpy().astype(float); pc = A.loc[uniq.index, "xp"].to_numpy()
        auc = _auc(yc, pc)                                                                  # 표본 외(cross-fitted) AUC, 고유 통제 패스
        dec = pd.qcut(pc, 10, labels=False, duplicates="drop"); calib = pd.DataFrame({"y": yc, "p": pc, "dec": dec}).groupby("dec").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        calib_slope = float(np.polyfit(calib.pred, calib.obs, 1)[0]); brier = float(np.mean((pc - yc) ** 2))
        # 창 수준 집계: 관측 성공률, 기대 성공률, 패스 수, 구역 분포
        A["third"] = pd.cut(A.x, [-1, 40, 80, 121], labels=["defensive", "middle", "attacking"]); A["fwd"] = (A.dx > 0).astype(float)
        wsum = A.groupby(["unit_id", "event_id", "window", "is_post", "score_state"]).agg(n=("completed", "size"), obs=("completed", "mean"), exp=("xp", "mean"), x_mean=("x", "mean"),
                                                                                           def_share=("third", lambda s: (s == "defensive").mean()), att_share=("third", lambda s: (s == "attacking").mean()),
                                                                                           fwd_share=("dx", lambda s: (s > 0).mean()), len_mean=("length", "mean")).reset_index()
        wsum["oe"] = wsum.obs - wsum.exp
        base = wsum[~wsum.is_post.astype(bool)]
        cells = base.groupby(["unit_id", "score_state"]).agg(bn=("oe", "size"), oe_b=("oe", "mean"), obs_b=("obs", "mean"), exp_b=("exp", "mean"), n_b=("n", "mean"), x_b=("x_mean", "mean"),
                                                             def_b=("def_share", "mean"), att_b=("att_share", "mean"), fwd_b=("fwd_share", "mean"), len_b=("len_mean", "mean")).reset_index()
        cells = cells[cells.bn >= MIN_N]
        rng = np.random.default_rng(17)
        out = {"xpass_model": {"n_unique_control_passes": n_unique_ctrl, "n_control_pass_window_rows": n_rows_ctrl, "n_unique_passes_all": int(A._i.nunique()),
                               "auc_crossfit_oos": round(auc, 3), "brier_oos": round(brier, 4), "calibration_slope_deciles": round(calib_slope, 3),
                               "calibration_deciles": [[round(float(a), 3), round(float(b), 3), int(c)] for a, b, c in zip(calib.pred, calib.obs, calib.n)],
                               "folds": 5, "fold_unit": "match", "params_full_fit": {k: round(float(v), 4) for k, v in model.params.items()},
                               "note": "fitted to unique unpressured short passes in control windows; each pass's expected completion comes from the fold model that excludes its match"}}
        for w in WINDOWS:
            e = wsum[(wsum.is_post.astype(bool)) & (wsum.window == w)].merge(cells, on=["unit_id", "score_state"])
            units = e.unit_id.to_numpy(); uu = np.unique(units); idx = {u: np.where(units == u)[0] for u in uu}
            def cb(v):
                bs = [v[np.concatenate([idx[u] for u in rng.choice(uu, len(uu))])].mean() for _ in range(1000)]
                return [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
            out[w] = {"n_events": int(len(e)),
                      "observed_minus_control_pp": cb((e.obs - e.obs_b).to_numpy() * 100),
                      "expected_minus_control_pp": cb((e.exp - e.exp_b).to_numpy() * 100),
                      "obs_minus_exp_vs_control_pp": cb((e.oe - e.oe_b).to_numpy() * 100),
                      "passes_per_window_diff": cb((e.n - e.n_b).to_numpy()), "passes_per_window_control": round(float(e.n_b.mean()), 2),
                      "start_x_diff_yards": cb((e.x_mean - e.x_b).to_numpy()), "start_x_control": round(float(e.x_b.mean()), 2),
                      "defensive_third_share_diff_pp": cb((e.def_share - e.def_b).to_numpy() * 100), "defensive_third_share_control_pct": round(float(e.def_b.mean() * 100), 2),
                      "attacking_third_share_diff_pp": cb((e.att_share - e.att_b).to_numpy() * 100), "attacking_third_share_control_pct": round(float(e.att_b.mean() * 100), 2),
                      "forward_share_diff_pp": cb((e.fwd_share - e.fwd_b).to_numpy() * 100), "length_diff_yards": cb((e.len_mean - e.len_b).to_numpy())}
            print(side, w, {k: v for k, v in out[w].items() if k in ("observed_minus_control_pp", "expected_minus_control_pp", "obs_minus_exp_vs_control_pp", "passes_per_window_diff")}, flush=True)
        res[side] = out
    save_json(res, OUT); print("✓", OUT)


def _auc(y, p):
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        analyse()
