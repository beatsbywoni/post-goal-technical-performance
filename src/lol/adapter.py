"""LoL 원자료(피처 jsonl) → 롱포맷 계약 변환 (§4, §12.2). 도메인 지식은 여기 + config.

L1(게임 간): 플레이어별 게임을 시간순 정렬해 '연패 사건'(직전 k게임 연속 패, 세션
연속 <12h)을 식별. 사건 직후 게임의 실행/선택 지표를 롱포맷으로 산출하고, 나머지
게임은 baseline 분포로 둔다(개인 내·포지션 매칭). H2 dose=streak_depth, H3=interval_band.

출력은 src/core/contract.LongRecord 와 동일 → 같은 src/core/ 로 무수정 처리(이식 주장).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 피처 키 → (config metric 키, metric_class)
METRIC_MAP = {
    "cs_per_min": ("cs_per_min_dev", "execution"),
    "skillshots_hit_per_min": ("skillshot_hit_pm", "execution"),
    "wards_per_min": ("vision_per_min", "choice"),
}
SESSION_GAP_H = 12.0


def interval_band(gap_h: float) -> str:
    m = gap_h * 60
    if m < 10:
        return "<10m"
    if m < 30:
        return "10-30m"
    if gap_h < 3:
        return "30m-3h"
    return ">3h"


def l1_to_long(rows: list[dict], k: int = 2, session_gap_h: float = SESSION_GAP_H,
               success: bool = False) -> list[dict]:
    """L1 피처 → 롱포맷 레코드. 직전 연속 k 이상 직후 게임 = 사건(window=post1).

    success=False → 연패 streak (틸트, 직전이 '승'이면 중단).
    success=True  → 연승 streak (모멘텀 자매연구, 직전이 '패'이면 중단).
    """
    by_unit: dict[str, list] = defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)

    out: list[dict] = []
    for unit, games in by_unit.items():
        games = sorted(games, key=lambda g: g.get("game_start_ms") or 0)
        for i, g in enumerate(games):
            # 직전 연속(패 또는 승) 수 (세션 연속: 인접 게임 간격 <12h)
            depth = 0
            for j in range(i - 1, -1, -1):
                a = games[j + 1].get("game_start_ms") or 0
                b = games[j].get("game_start_ms") or 0
                gap_h = (a - b) / 3_600_000 if a and b else 99
                broke = (not games[j].get("win")) if success else bool(games[j].get("win"))
                if gap_h >= session_gap_h or broke:
                    break
                depth += 1
            is_event = depth >= k

            ctx = {"position": g.get("position"), "patch": g.get("patch")}
            if is_event:
                prev = games[i - 1].get("game_start_ms") or 0
                cur = g.get("game_start_ms") or 0
                ctx["streak_depth"] = depth
                ctx["interval_band"] = interval_band((cur - prev) / 3_600_000) if prev else None
            window = "post1" if is_event else "baseline"

            for feat, (metric, cls) in METRIC_MAP.items():
                val = g.get(feat)
                if val is None:
                    continue
                out.append(dict(
                    unit_id=unit, domain="lol", event_id=str(g.get("match_id")),
                    window=window, metric=metric, value=float(val), metric_class=cls,
                    context=dict(ctx), schema_ver="v1",
                ))
    return out


def _l2_minute_band(t_ms: float) -> str:
    m = t_ms / 60_000
    return "0-15" if m < 15 else ("15-30" if m < 30 else "30+")


def l2_compact_to_long(rows: list[dict], window_min: int = 5,
                       baseline_step_min: int = 1, event_key: str = "death_ms") -> list[dict]:
    """게임별 컴팩트 시계열 → L2 롱포맷 (오프라인, 재페치 불요).

    event_key="death_ms"     → 데스 클러스터(틸트). event_key="takedown_ms" → 테이크다운
    클러스터(모멘텀 자매연구). window_min 내 2+ 사건의 첫 사건 기준 post5/10/15(각 5분)
    분당 CS 와, **post 와 동일한 5분 해상도의 슬라이딩 평시 윈도우**(step baseline_step_min)를
    기저선으로. → 해상도 일치로 z 비교 타당. unit=플레이어×게임, 분 구간 매칭.
    """
    import bisect
    POST = {"post5": (0, 5), "post10": (5, 10), "post15": (10, 15)}
    WLEN = window_min * 60_000
    out: list[dict] = []

    for r in rows:
        ft, cs = r["frame_ms"], r["cs"]
        if len(ft) < 2:
            continue
        deaths = sorted(r.get(event_key, []))
        end = ft[-1]

        def cs_val(t):
            if t <= ft[0]:
                return float(cs[0])
            if t >= ft[-1]:
                return float(cs[-1])
            i = bisect.bisect_right(ft, t) - 1
            t0, t1, c0, c1 = ft[i], ft[i + 1], cs[i], cs[i + 1]
            return float(c0 + (c1 - c0) * (t - t0) / (t1 - t0)) if t1 > t0 else float(c0)

        def rate(t0, t1):
            return (cs_val(t1) - cs_val(t0)) / ((t1 - t0) / 60_000)

        # 클러스터 (첫 데스 시각, 크기) — 크기는 H2 dose
        clusters, i = [], 0
        while i < len(deaths):
            j = i
            while j + 1 < len(deaths) and deaths[j + 1] - deaths[i] <= WLEN:
                j += 1
            if j - i + 1 >= 2:
                clusters.append((deaths[i], j - i + 1)); i = j + 1
            else:
                i += 1

        unit = f"{r['unit_id']}_{r['match_id']}"
        post_spans = []

        def emit(eid, win, val, t_ctx, extra=None):
            ctx = {"minute_band": _l2_minute_band(t_ctx), "match_id": r["match_id"]}
            if extra:
                ctx.update(extra)
            out.append(dict(unit_id=unit, domain="lol", event_id=eid, window=win,
                            metric="cs_per_min_dev", value=float(val), metric_class="execution",
                            context=ctx, schema_ver="v1"))

        for ci, (ct, csize) in enumerate(clusters):
            for w, (a, b) in POST.items():
                t0, t1 = ct + a * 60_000, ct + b * 60_000
                if t1 > end:
                    continue            # 게임 종료에 잘리는 윈도우 제외(절단 편향 방지)
                emit(f"cluster_{ci}", w, rate(t0, t1), t0,   # 분 구간 = 윈도우 시작 기준
                     {"cluster_size": csize})
                post_spans.append((t0, t1))

        # 기저선: post 와 동일한 5분 슬라이딩 윈도우 (사건 윈도우 비겹침)
        step = baseline_step_min * 60_000
        t0 = 0
        while t0 + WLEN <= end:
            t1 = t0 + WLEN
            if not any(t0 < hi and t1 > lo for lo, hi in post_spans):
                emit(f"bl_{t0}", "baseline", rate(t0, t1), t0)   # 윈도우별 고유 id(위약)
            t0 += step
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LoL L1 → 롱포맷")
    ap.add_argument("--input", default="data/raw/lol_l1_features.jsonl")
    ap.add_argument("--out", default="data/features/lol_l1_long.parquet")
    ap.add_argument("-k", type=int, default=2)
    args = ap.parse_args(argv)

    import pandas as pd
    from src.core.contract import validate_records

    rows = _read_jsonl(Path(args.input))
    recs = l1_to_long(rows, k=args.k)
    validate_records(recs)                       # 계약 검증
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(out, index=False)
    n_evt = sum(1 for r in recs if r["window"] == "post1")
    print(f"✓ {out} — {len(recs)} 레코드 (사건 post1 {n_evt}, k={args.k}) 검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
