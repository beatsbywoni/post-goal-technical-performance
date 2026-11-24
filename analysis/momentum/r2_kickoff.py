#!/usr/bin/env python3
"""R2 · 리스타트(킥오프) 제외 민감도 (Reviewer 2, R1 결정문 포인트 2).

  python3 -m analysis.momentum.r2_kickoff build  [--cache-dir data/raw/sb_events]   # 두 변형의 롱포맷 피처 생성
  python3 -m analysis.momentum.r2_kickoff estimate                                   # 표 3 재추정 → docs/ijpas_r2_kickoff.json
  python3 -m analysis.momentum.r2_kickoff check                                      # 기본(제외 없음) 재적재가 기존 피처와 동일한지 확인

변형(src/football/adapter_fast.events_to_long_fast 의 restart_exclusion):
  kickoff_possession : play_pattern == "From Kick Off" 이벤트(킥오프로 시작한 소유권 전체, 양 팀)를 지표 계산에서 제외
  first60s           : 어느 골이든 뒤 60초와 각 피리어드 첫 60초의 이벤트를 지표 계산에서 제외
사건·통제 창 모두에, 득점·실점 양쪽에, 세 창 모두에 같은 규칙을 적용한다. 골·창 격자·골 원장·단일 골 노출·셀(≥3)은 주 분석과 같다.
추정: 주 분석과 같은 규약(사건·통제 각각 1/99 윈저화, 사건 셀 통제, 유닛 클러스터 부트스트랩 2,000; 앵커 5,000) 과 원 단위(%p).
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from analysis.momentum.r1_lib import (ANCHOR, EXEC, FILES, MIN_N, WINDOWS, Standardised, cluster_inference, ctx_key, load,  # noqa: E402
                                      save_json)
from analysis.momentum.run_ijpas_r2 import frames, raw_units  # noqa: E402
from src.football.adapter_fast import events_to_long_fast  # noqa: E402

VARIANTS = {"kickoff_possession": "kickoff_excl", "first60s": "kickoff_60s"}   # adapter restart_exclusion → feature sub-directory
KEYS = {"kickoff_possession": "kickoff_possession", "first60s": "kickoff_60s"}     # result keys "<side>|<key>" read by the build scripts
OUT = ROOT / "docs/ijpas_r2_kickoff.json"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def build(cache_dir: Path, variants: dict, out_root: Path, limit: int | None = None):
    """변형 × 모드를 하나씩 적재해 parquet 으로 쓴다(메모리: 한 번에 한 파일). 레코드 계약 검증은 표본(앞 200,000 레코드)에 적용한다;
    구조는 주 분석과 같은 adapter 가 만들므로 전수 검증은 sb_ingest 와 동일하게 선택 사항이다."""
    from src.core.contract import validate_records
    ml = json.load(open(ROOT / "data/interim/match_list.json"))
    if limit:
        ml = ml[:limit]
    missing = [mid for _, _, mid in ml if not (cache_dir / f"{mid}.pkl.gz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} matches missing from cache, e.g. {missing[:5]}")
    for v, sub in variants.items():
        for mode in ("score", "concede"):
            recs = []
            for i, (comp, season, mid) in enumerate(ml, 1):
                ev = pd.read_pickle(cache_dir / f"{mid}.pkl.gz")
                recs.extend(events_to_long_fast(ev, match_id=str(mid), competition=comp, event_mode=mode, variant="v2", restart_exclusion=v))
                if i % 400 == 0:
                    log(f"{v} {mode} …{i}/{len(ml)}")
            validate_records(recs[:200000])
            d = out_root / sub; d.mkdir(parents=True, exist_ok=True)
            p = d / FILES[mode]; pd.DataFrame(recs).to_parquet(p, index=False); log("✓", p, len(recs), "rows")
            del recs


def check(cache_dir: Path, n: int | None = None):
    """restart_exclusion=None 재적재가 기존 피처(data/features)와 레코드 단위로 같은지 확인하고 결과를 docs/ijpas_r2_kickoff.json 의 _cache_check 에 기록한다.
    n=None 이면 전체 2,404경기; 캐시 매니페스트(경기 수, 날짜, statsbombpy·pandas 버전)도 남긴다."""
    from importlib.metadata import version as _ver
    ml = json.load(open(ROOT / "data/interim/match_list.json"))
    if n:
        rng = np.random.default_rng(0); ml = [ml[i] for i in rng.choice(len(ml), n, replace=False)]
    cols = ["unit_id", "event_id", "window", "metric", "value"]
    refs = {}
    for mode in ("score", "concede"):
        r = pd.read_parquet(ROOT / "data/features" / FILES[mode], columns=cols)
        r["match_id"] = r.unit_id.str.rsplit("-", n=1).str[-1]
        refs[mode] = r.sort_values("match_id", kind="stable").reset_index(drop=True)
    def slice_match(mode, mid):
        a = refs[mode].match_id.to_numpy(); i0 = np.searchsorted(a, mid, "left"); i1 = np.searchsorted(a, mid, "right")
        return refs[mode].iloc[i0:i1]
    bad = {"score": [], "concede": []}; n_rows = {"score": 0, "concede": 0}
    for i, (comp, season, mid) in enumerate(ml, 1):
        ev = pd.read_pickle(cache_dir / f"{mid}.pkl.gz")
        for mode in ("score", "concede"):
            new = pd.DataFrame(events_to_long_fast(ev, match_id=str(mid), competition=comp, event_mode=mode, variant="v2"))
            old = slice_match(mode, str(mid))
            k = ["unit_id", "event_id", "window", "metric"]
            if len(new) == 0 and len(old) == 0:
                continue
            a = new[k + ["value"]].sort_values(k).reset_index(drop=True) if len(new) else new
            b = old[k + ["value"]].sort_values(k).reset_index(drop=True)
            ok = len(a) == len(b) and len(a) > 0 and (a[k].values == b[k].values).all() and np.allclose(a.value.to_numpy(), b.value.to_numpy(), atol=1e-9)
            n_rows[mode] += int(len(b))
            if not ok:
                bad[mode].append(mid); log("✗ differs", mode, mid, len(a), len(b))
        if i % 400 == 0:
            log(f"…{i}/{len(ml)} checked")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["_cache_check"] = {"date": time.strftime("%Y-%m-%d"), "matches_checked": len(ml), "matches_in_cache": len(list(cache_dir.glob("*.pkl.gz"))),
                           "rows_compared": n_rows, "matches_differing": bad, "identical": all(len(v) == 0 for v in bad.values()),
                           "statsbombpy": _ver("statsbombpy"), "pandas": pd.__version__, "numpy": np.__version__,
                           "note": "re-ingestion of the freshly downloaded open-data events without any exclusion, compared record by record with the deposited data/features parquet files"}
    save_json(res, OUT)
    for mode in bad:
        log(f"{mode}: {len(ml) - len(bad[mode])}/{len(ml)} matches identical to data/features")


def share_removed(base_dir: Path, var_dir: Path, mode: str) -> dict:
    """변형에서 제거된 이벤트의 비중: 창별 비압박 단거리 패스 시도(sp_attempts)·패스 수(pass_volume)·터치(touches) 합계의 감소분."""
    out = {}
    cols = ["unit_id", "event_id", "window", "metric", "value"]
    a = pd.read_parquet(base_dir / FILES[mode], columns=cols); b = pd.read_parquet(var_dir / FILES[mode], columns=cols)
    for m in ("sp_attempts", "pass_volume", "touches", "def_actions"):
        aa = a[a.metric == m].groupby("window").value.sum(); bb = b[b.metric == m].groupby("window").value.sum()
        out[m] = {w: {"base": float(aa.get(w, 0)), "variant": float(bb.get(w, 0)), "removed_pct": round(100 * (1 - float(bb.get(w, 0)) / float(aa.get(w, 1))), 2)} for w in ["post5", "post10", "post15", "baseline"]}
    return out


def estimate(feature_root: Path):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for v, sub in VARIANTS.items():
        fd = feature_root / sub
        for side in ("score", "concede"):
            df = load(side, fd); r = {"n_rows": int(len(df))}
            for key in ("pre", "state"):
                S = Standardised(df, key)
                for w in WINDOWS:
                    ev, pl = frames(S, w)
                    nb = 5000 if w == ANCHOR else 2000
                    cu = cluster_inference(ev, pl, "unit_id", n=nb); cp = cluster_inference(ev, pl, "unit_id", n=nb, col="zp")
                    r[f"{key}|{w}"] = {"cluster_unit": cu, "cluster_unit_pooled": cp,
                                       "n_event": int(len(ev)), "n_control": int(len(pl)), "n_units_event": int(ev.reset_index().unit_id.nunique())}
                    if key == "state":
                        r[f"{key}|{w}"]["raw"] = raw_units(df, key, w)
                    log(v, side, key, w, "d", cu["d"], cu["ci95"], "pooled", cp["d"], "raw sp", r[f"{key}|{w}"].get("raw", {}).get("short_pass_acc", {}).get("diff_pp"))
            r["share_removed"] = share_removed(ROOT / "data/features", fd, side)
            res[f"{side}|{KEYS[v]}"] = r; save_json(res, OUT)
    res["_meta"] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "variants": {KEYS[v]: "play_pattern == 'From Kick Off' events excluded from indicator computation (both teams, all windows)" if v == "kickoff_possession"
                                                                                else "events within 60 s after any goal and within the first 60 s of each period excluded (all windows)" for v in VARIANTS},
                    "conventions": "as the primary analysis: goal ledger, single-goal exposure, dismissal exclusion, event-cell controls, cells >= 3, separate 1/99 winsorisation; unit-cluster bootstrap (5,000 at post10, 2,000 elsewhere); raw units = event minus own-cell control mean, %p"}
    save_json(res, OUT); log("✓", OUT)


def shares_analysed(feature_root: Path):
    """분석에 실제로 들어간 창만 대상으로 한 제거 비중(추가, 2026-11-08): 변형 추정에 쓰인 이벤트 창과 그 셀의 통제창에서
    비압박 단거리 패스 시도(sp_attempts)가 얼마나 제거되었는지. share_removed(전체 골 창·전체 통제창 기준)와 구분하여
    results[f"{side}|{KEYS[v]}"]["share_removed_analysed"]에 기록한다. 부트스트랩을 다시 돌리지 않는다."""
    res = json.loads(OUT.read_text())
    for side in ("score", "concede"):
        base = pd.read_parquet(ROOT / "data/features" / FILES[side], columns=["unit_id", "event_id", "window", "metric", "value"])
        base = base[base.metric == "sp_attempts"].copy(); base["key"] = base.unit_id + "|" + base.event_id
        for v, sub in VARIANTS.items():
            fd = feature_root / sub; df = load(side, fd); S = Standardised(df, "state")
            var = pd.read_parquet(fd / FILES[side], columns=["unit_id", "event_id", "window", "metric", "value"])
            var = var[var.metric == "sp_attempts"].copy(); var["key"] = var.unit_id + "|" + var.event_id
            out = {}
            for w in WINDOWS:
                ev, pl = frames(S, w); e = ev.reset_index(); p = pl.reset_index()
                ev_ids = set(e.unit_id + "|" + e.event_id); pl_ids = set(p.unit_id + "|" + p.event_id)
                def tot(d, ids, win): return float(d[(d.window == win) & (d.key.isin(ids))].value.sum())
                be, ve = tot(base, ev_ids, w), tot(var, ev_ids, w); bc, vc = tot(base, pl_ids, "baseline"), tot(var, pl_ids, "baseline")
                out[w] = {"n_event": len(ev_ids), "event_base": be, "event_removed_pct": round(100 * (1 - ve / be), 2),
                          "n_control": len(pl_ids), "control_base": bc, "control_removed_pct": round(100 * (1 - vc / bc), 2),
                          "note": "unpressured short passes attempted in the analysed event windows and in the control windows of their cells (variant estimate); base = primary features, variant = features with the exclusion"}
                log(v, side, w, "analysed-window removal %", out[w]["event_removed_pct"], out[w]["control_removed_pct"])
            res[f"{side}|{KEYS[v]}"]["share_removed_analysed"] = {"sp_attempts": out}
    save_json(res, OUT)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "estimate", "check", "shares"]); ap.add_argument("--cache-dir", default="data/raw/sb_events")
    ap.add_argument("--feature-root", default="data/features"); ap.add_argument("--limit", type=int, help="build: first N matches; check: random N matches (default all)")
    a = ap.parse_args(argv)
    cache = ROOT / a.cache_dir; froot = ROOT / a.feature_root
    if a.cmd == "build":
        build(cache, VARIANTS, froot, a.limit)
    elif a.cmd == "check":
        check(cache, a.limit)
    elif a.cmd == "shares":
        shares_analysed(froot)
    else:
        estimate(froot)


if __name__ == "__main__":
    main()
