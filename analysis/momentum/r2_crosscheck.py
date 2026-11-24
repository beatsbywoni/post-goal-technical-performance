#!/usr/bin/env python3
"""R1 second analysis · 독립 교차검증: parquet + 골 원장에서 직접(r1_lib 를 쓰지 않고) 앵커 창의 주 추정(d, n), 참조 추정, 흐름 수를 다시 계산한다.

  python3 -m analysis.momentum.r2_crosscheck  → docs/ijpas_r2_crosscheck.json
규칙: 퇴장 창 제외; 단일 골 노출(창 종료 전 다른 골 제외); 사건 상태 = 원장의 state_after(팀 관점), 참조 = state_before;
통제 셀 = (unit, state) 의 통제 창 ≥ 3; 통제는 사건 셀의 창만; z = dir × (x − mean)/sd (셀 내), net z = 지표 평균; 사건·통제 각각 1/99 클리핑; d = 평균차/합동 SD.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF = json.loads((ROOT / "docs/ijpas_r2.json").read_text())
EXEC = {"short_pass_acc": 1.0, "miscontrol_rate": -1.0, "foul_per_def_action": -1.0}
FN = {"score": "football_momentum_long.parquet", "concede": "football_long.parquet"}
HI = {"post5": 300.0, "post10": 600.0, "post15": 900.0}
TOL = 0.0015


def swap(s): a, b = s.split("-"); return f"{b}-{a}"


def cohen(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = np.clip(a, *np.percentile(a, [1, 99])); b = np.clip(b, *np.percentile(b, [1, 99]))
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def main():
    led = pd.read_parquet(ROOT / "data/interim/goal_ledger.parquet"); out = {}
    for side in ("score", "concede"):
        raw = pd.read_parquet(ROOT / "data/features" / FN[side]); raw = raw[raw.metric.isin(EXEC)]
        ctx = pd.DataFrame(raw.context.tolist(), index=raw.index)
        df = pd.DataFrame({"unit_id": raw.unit_id.values, "event_id": raw.event_id.values, "window": raw.window.values, "metric": raw.metric.values, "value": raw.value.values,
                           "state_ws": ctx.score_state.values, "dism": ctx.dismissal_in_window.astype(bool).values, "goal_index": pd.to_numeric(ctx.goal_index, errors="coerce").values})
        df["match_id"] = df.unit_id.str.rsplit("-", n=1).str[-1]; df = df[~df.dism]
        post = df.window.str.startswith("post")
        m = df.loc[post, ["match_id", "goal_index"]].astype({"goal_index": int}).merge(led[["match_id", "goal_index", "state_before", "state_after", "next_goal_dt"]], on=["match_id", "goal_index"], how="left")
        assert m.state_after.notna().all()
        st_after = m.state_after if side == "score" else m.state_after.map(swap); st_before = m.state_before if side == "score" else m.state_before.map(swap)
        df["state"] = df.state_ws; df.loc[post, "state"] = st_after.to_numpy(); df["state_pre"] = df.state_ws; df.loc[post, "state_pre"] = st_before.to_numpy()
        other = (m.next_goal_dt.to_numpy() < df.loc[post, "window"].map(HI).to_numpy()); keep = np.ones(len(df), bool); keep[np.where(post)[0][other]] = False
        pw = df.loc[post, ["unit_id", "event_id", "window"]].assign(other=other); n_other10 = int(pw[(pw.window == "post10") & pw.other].drop_duplicates(["unit_id", "event_id"]).shape[0])
        df = df[keep]
        base = df[~df.window.str.startswith("post")]
        res = {}
        for key, col in (("state", "state"), ("pre", "state_pre")):
            cells = base.groupby(["unit_id", "metric", col]).value.agg(["mean", "std", "count"]).reset_index(); cells = cells[cells["count"] >= 3]
            for w in ("post5", "post10", "post15"):
                ev = df[df.window == w].merge(cells, on=["unit_id", "metric", col])
                ev["z"] = ev.metric.map(EXEC) * (ev.value - ev["mean"]) / ev["std"].replace(0, np.nan); ev = ev[ev.z.notna()]
                evz = ev.groupby(["unit_id", "event_id"]).z.mean()
                ec = ev[["unit_id", col]].drop_duplicates()
                ctl = base.merge(ec, on=["unit_id", col]).merge(cells, on=["unit_id", "metric", col]); ctl["z"] = ctl.metric.map(EXEC) * (ctl.value - ctl["mean"]) / ctl["std"].replace(0, np.nan); ctl = ctl[ctl.z.notna()]
                clz = ctl.groupby(["unit_id", "event_id"]).z.mean()
                d = cohen(evz.to_numpy(), clz.to_numpy()); ref = REF[f"{side}|{key}|{w}"]["unit"]
                res[f"{key}|{w}"] = {"d": round(d, 4), "n_event": int(len(evz)), "n_control": int(len(clz)), "ref_d": ref["d"], "ref_n_event": ref["n_event"], "ref_n_control": ref["n_control"],
                                     "agree": bool(abs(d - ref["d"]) < TOL and len(evz) == ref["n_event"] and len(clz) == ref["n_control"])}
                print(side, key, w, res[f"{key}|{w}"], flush=True)
        # flow count (anchor): events with another goal before the window end (among events with window data, dismissal windows removed first in both paths? no: ledger flow counts dismissal after exposure)
        fl = REF[f"{side}|flow"]["state|post10"]
        res["flow_other_goal_before_end"] = {"recomputed": n_other10, "ref": fl["other_goal_before_window_end"], "note": "reference counts goals before the dismissal step; this recount removes dismissal windows first, so it is a lower bound and equals the reference when no multi-goal window also contains a dismissal"}
        out[side] = res
    out["all_agree"] = bool(all(v["agree"] for s in ("score", "concede") for k, v in out[s].items() if isinstance(v, dict) and "agree" in v))
    (ROOT / "docs/ijpas_r2_crosscheck.json").write_text(json.dumps(out, indent=1)); print("✓ all agree:", out["all_agree"])


if __name__ == "__main__":
    main()
