"""행동적 부정성 비대칭 — net-z 확증 배터리 (A1–A5). 피벗판.

주 측정 = 순(net) oriented z 평균(방향). 1/99 윈저화. 사건 vs 위약 SMD.
A2(성공 무부양)는 부트스트랩 90% CI ⊂ (−0.15, 0.15) 등가검정(TOST 근사).
결과 → docs/results_momentum.md.

실행: bash scripts/momentum.sh run   (또는 python3 -m analysis.momentum.run_asymmetry)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "prereg_confirmatory"))

from src.core.tilt_index import run_pipeline, standardize     # noqa: E402
from src.core.config import direction_map, load_metric_meta   # noqa: E402
from src.lol.adapter import l1_to_long, l2_compact_to_long    # noqa: E402

OUT = ROOT / "docs" / "results_momentum.md"
RNG = np.random.default_rng(42)


def _wins(a, lo=1, hi=99):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) < 10:
        return a
    return np.clip(a, np.percentile(a, lo), np.percentile(a, hi))


def _d(a, b):
    a, b = _wins(a), _wins(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp else float("nan")


def netz_arrays(records, domain, mf, window="post10"):
    """사건 윈도우 net-z 배열, 위약 net-z 배열 (unit×event 단위 평균)."""
    o = run_pipeline(records, domain=domain, match_fields=mf, baseline_window="baseline")
    z = o["z"]; dirmap = direction_map(load_metric_meta(domain))
    df = pd.DataFrame(records)
    bl = standardize(df[df.window == "baseline"], o["baselines"], dirmap, mf)

    def collapse(zz, win):
        s = zz[(zz["metric_class"] == "execution") & zz["z"].notna() & (zz["window"] == win)]
        return s.groupby(["unit_id", "event_id"])["z"].mean().to_numpy()

    return collapse(z, window), collapse(bl, "baseline")


def boot_ci(ev, pl, n=2000, q=(2.5, 97.5)):
    ds = []
    for _ in range(n):
        be = RNG.choice(ev, len(ev), replace=True); bp = RNG.choice(pl, len(pl), replace=True)
        ds.append(_d(be, bp))
    return np.nanpercentile(ds, q[0]), np.nanpercentile(ds, q[1])


def perm_null(ev, pl, n=2000):
    """위약을 무작위 2분할 → 귀무 d 분포의 95% 구간."""
    pool = np.concatenate([ev, pl]); k = len(ev); ds = []
    for _ in range(n):
        sh = RNG.permutation(pool)
        ds.append(_d(sh[:k], sh[k:]))
    return np.nanpercentile(ds, 2.5), np.nanpercentile(ds, 97.5)


def tost(ev, pl, bound=0.15, n=2000):
    """등가검정 근사: d 의 90% CI ⊂ (−bound,+bound) 이면 '효과 부재' 입증."""
    lo, hi = boot_ci(ev, pl, n=n, q=(5, 95))
    return lo, hi, (lo > -bound and hi < bound)


def d_winsor(ev, pl, lohi):
    a, b = np.asarray(ev, float), np.asarray(pl, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    lo, hi = lohi
    if lo > 0:
        a = np.clip(a, np.percentile(a, lo), np.percentile(a, hi))
        b = np.clip(b, np.percentile(b, lo), np.percentile(b, hi))
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    return (a.mean()-b.mean())/sp if sp else float("nan")


def log(msg):
    print(msg, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def lol_l1(success):
    rows = [json.loads(l) for l in (ROOT / "data/raw/lol_l1_features.jsonl").read_text().splitlines() if l.strip()]
    cs = np.array([r.get("cs_per_min", 0) for r in rows]); thr = np.percentile(cs, 1)
    rows = [r for r in rows if r.get("duration_s", 0) >= 300 and r.get("cs_per_min", 0) > thr]
    return l1_to_long(rows, k=2, success=success)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="LOCO 생략(샌드박스 검증용)")
    a = ap.parse_args(argv)
    OUT.write_text(f"# 부정성 비대칭 — net-z 확증·강건성 결과\n\n생성: {time.strftime('%Y-%m-%d %H:%M')}\n"
                   f"주 측정: net oriented z(방향), 1/99 윈저화, 사건 vs 위약 SMD.\n\n", encoding="utf-8")
    t0 = time.time()
    l2 = [json.loads(l) for l in (ROOT / "data/raw/lol_l2_compact.jsonl").read_text().splitlines() if l.strip()]
    fb_score = pd.read_parquet(ROOT / "data/features/football_momentum_long.parquet").to_dict("records")
    fb_conc = pd.read_parquet(ROOT / "data/features/football_long.parquet").to_dict("records")

    rows = [
        ("LoL 게임내", "실패(데스 클러스터)", lambda: netz_arrays(l2_compact_to_long(l2, event_key="death_ms"), "lol", ["minute_band"])),
        ("LoL 게임내", "성공(테이크다운)",   lambda: netz_arrays(l2_compact_to_long(l2, event_key="takedown_ms"), "lol", ["minute_band"])),
        ("축구",       "실패(실점)",         lambda: netz_arrays(fb_conc, "football", ["score_state"])),
        ("축구",       "성공(득점)",         lambda: netz_arrays(fb_score, "football", ["score_state"])),
        ("LoL 게임간", "실패(연패)",         lambda: netz_arrays(lol_l1(False), "lol", ["position"], "post1")),
        ("LoL 게임간", "성공(연승)",         lambda: netz_arrays(lol_l1(True),  "lol", ["position"], "post1")),
    ]
    log("## A1/A3 (실패=저하) · A2 (성공=무부양) — post10/post1, net-z d (vs 위약)")
    log("| 도메인 | 사건 | d | 95% CI | n | 판정 |")
    log("|---|---|---|---|---|---|")
    res, arr = {}, {}
    for dom, lab, fn in rows:
        ev, pl = fn(); d = _d(ev, pl); lo, hi = boot_ci(ev, pl)
        res[(dom, lab[:2])] = (d, lo, hi, len(ev)); arr[(dom, lab[:2])] = (ev, pl)
        if "성공" in lab:
            verdict = "무부양(등가)" if (lo > -0.15 and hi < 0.15) else ("향상" if lo > 0.15 else ("저하/안주" if hi < -0.15 else "불명확"))
        else:
            verdict = "저하 ✓" if hi < -0.15 else ("저하(경계)" if d < -0.10 else "약함")
        log(f"| {dom} | {lab} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {len(ev)} | {verdict} |")
    log("")

    log("## A4 도메인별 비대칭 (|실패 d| vs |성공 d|)")
    for dom in ["LoL 게임내", "축구", "LoL 게임간"]:
        f = res.get((dom, "실패")); s = res.get((dom, "성공"))
        if f and s:
            log(f"- **{dom}**: |실패|={abs(f[0]):.3f}  vs  |성공|={abs(s[0]):.3f}  → "
                f"{'실패가 큼(부정성 비대칭)' if abs(f[0])>abs(s[0]) else '성공이 큼(안주 우세)'}")
    log("")

    # ── 강건성 ──
    log("## R1. A2 등가검정 (TOST, 성공=무부양; 90% CI ⊂ ±0.15 이면 '향상 부재' 입증)")
    for dom in ["LoL 게임내", "축구", "LoL 게임간"]:
        ev, pl = arr[(dom, "성공")]
        lo, hi, ok = tost(ev, pl)
        log(f"- {dom} 성공: 90% CI [{lo:+.3f}, {hi:+.3f}] → {'등가(향상 부재) ✓' if ok else ('저하측' if hi<-0.15 else '불명확')}")
    log("")

    log("## R2. 순열 귀무검정 (위약 무작위 분할, 2000회)")
    for dom, lab in [("LoL 게임내", "실패"), ("LoL 게임내", "성공"), ("축구", "실패"), ("축구", "성공")]:
        ev, pl = arr[(dom, lab)]; lo, hi = perm_null(ev, pl); obs = res[(dom, lab)][0]
        log(f"- {dom} {lab}: 귀무 95% [{lo:+.3f}, {hi:+.3f}] vs 실제 {obs:+.3f}")
    log("")

    log("## R3. 윈저화 민감도 (raw / 1-99 / 5-95)")
    for dom, lab in [("LoL 게임내", "실패"), ("LoL 게임내", "성공"), ("축구", "성공")]:
        ev, pl = arr[(dom, lab)]
        log(f"- {dom} {lab}: raw={d_winsor(ev,pl,(0,0)):+.3f} | 1-99={d_winsor(ev,pl,(1,99)):+.3f} | 5-95={d_winsor(ev,pl,(5,95)):+.3f}")
    log("")

    log("## R4. 대안 윈도우 민감도 (LoL 게임내, 클러스터 span 3/5/8분)")
    for wm in [3, 5, 8]:
        ed, pd_ = netz_arrays(l2_compact_to_long(l2, window_min=wm, event_key="death_ms"), "lol", ["minute_band"])
        et, pt = netz_arrays(l2_compact_to_long(l2, window_min=wm, event_key="takedown_ms"), "lol", ["minute_band"])
        log(f"- span={wm}분: 실패 d={_d(ed,pd_):+.3f} (n={len(ed)}) | 성공 d={_d(et,pt):+.3f} (n={len(et)})")
    log("")

    if not a.quick:
        log("## R5. 축구 leave-one-competition-out (실점 & 득점 net-z d)")
        for recs, name in [(fb_conc, "실점"), (fb_score, "득점")]:
            df = pd.DataFrame(recs); df["competition"] = df["context"].apply(lambda c: (c or {}).get("competition"))
            ds = []
            for comp in sorted(df["competition"].dropna().unique()):
                sub = df[df.competition != comp].to_dict("records")
                ev, pl = netz_arrays(sub, "football", ["score_state"]); dd = _d(ev, pl); ds.append(dd)
                log(f"  −{comp[:24]:24s} {name}: d={dd:+.3f}")
            log(f"  → {name} 범위 [{min(ds):+.3f}, {max(ds):+.3f}]")
        log("")

    log(f"— 소요 {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
