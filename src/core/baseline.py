"""② 개인/팀 내 기저선 — 맥락 매칭 평시 분포 (§3).

기저선 윈도우 행에서 (unit, metric, 맥락키)별 평균 b 와 표준편차 s 를 산출한다.
이는 개인 내(within-unit) 비교의 기준이며, 표준화 z = (x − b)/s 에 쓰인다(§3 ②③).
"""
from __future__ import annotations

import pandas as pd

from .windows import BASELINE, add_context_key


def compute_baselines(
    df: pd.DataFrame,
    match_fields: list[str],
    baseline_window: str = BASELINE,
    min_n: int = 3,
) -> pd.DataFrame:
    """(unit_id, metric, ctx) 별 기저선 통계.

    반환 컬럼: unit_id, metric, ctx, b_mean, b_std, b_n.
    min_n 미만 표본은 불안정 → 제외(이후 z 계산에서 매칭 실패 = 결측 처리).
    """
    base = df[df["window"] == baseline_window]
    if base.empty:
        return pd.DataFrame(columns=["unit_id", "metric", "ctx", "b_mean", "b_std", "b_n"])
    base = add_context_key(base, match_fields)
    g = base.groupby(["unit_id", "metric", "ctx"])["value"]
    stats = g.agg(b_mean="mean", b_std="std", b_n="count").reset_index()
    return stats[stats["b_n"] >= min_n].reset_index(drop=True)
