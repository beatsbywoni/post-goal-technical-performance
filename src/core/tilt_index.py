"""③ 표준화 이탈 z + ④ 합성 틸트 지수 BTS (§3, §4.4).

z_ikw = dir_k · (x − b_ik) / s_ik   (dir: higher_better=+1, lower_better=−1)
  → z<0 이 '악화'를 의미하도록 방향 정규화.
BTS_iw = Σ_k λ_k · max(0, −z_ikw)   (실행 트랙 1차, 선택 트랙 별도 → H5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import compute_baselines
from .config import direction_map, load_metric_meta, uniform_weights
from .windows import add_context_key, to_frame


def standardize(
    post: pd.DataFrame,
    baselines: pd.DataFrame,
    direction: dict[str, float],
    match_fields: list[str],
) -> pd.DataFrame:
    """사건후 행에 기저선을 맥락 매칭 조인해 방향 정규화 z 를 계산."""
    p = add_context_key(post, match_fields)
    m = p.merge(baselines, on=["unit_id", "metric", "ctx"], how="left")
    d = m["metric"].map(direction).fillna(1.0)
    std = m["b_std"].replace(0, np.nan)
    m["z"] = d * (m["value"] - m["b_mean"]) / std
    return m


def composite_bts(
    z_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    metric_class: str = "execution",
    side: str = "deterioration",
) -> pd.DataFrame:
    """(unit, event, window)별 단측 합성지수. 결측 z 행은 제외.

    side="deterioration" → BTS = Σ λ·max(0,−z)  (틸트; z<0=악화)
    side="improvement"   → BMS = Σ λ·max(0,+z)  (모멘텀; z>0=향상, 자매 연구)
    """
    sub = z_df[(z_df["metric_class"] == metric_class) & z_df["z"].notna()].copy()
    signed = sub["z"] if side == "improvement" else -sub["z"]
    sub["mag"] = signed.clip(lower=0)
    w = sub["metric"].map(weights) if weights else 1.0
    sub["contrib"] = (w if weights else 1.0) * sub["mag"]
    bts = (
        sub.groupby(["unit_id", "event_id", "window"])["contrib"]
        .sum().reset_index().rename(columns={"contrib": "BTS"})
    )
    return bts


def run_pipeline(
    records,
    domain: str,
    match_fields: list[str],
    min_baseline_n: int = 3,
    baseline_window: str = "baseline",
    config_dir=None,
    side: str = "deterioration",
) -> dict:
    """레코드 → (기저선 → z → 실행/선택 BTS) 일괄 실행. 도메인 무지: meta 만 주입.

    baseline_window 로 도메인별 기저선 윈도우명을 지정(LoL='baseline', 축구도 동일
    규약 사용 권장). 같은 코드가 두 도메인을 처리한다(이식 주장).
    """
    meta = load_metric_meta(domain, config_dir)
    direction = direction_map(meta)
    df = to_frame(records)
    baselines = compute_baselines(df, match_fields, baseline_window=baseline_window,
                                  min_n=min_baseline_n)
    we, wc = uniform_weights(meta, "execution"), uniform_weights(meta, "choice")
    post = df[df["window"] != baseline_window]
    z = standardize(post, baselines, direction, match_fields)
    # 위약(placebo) = 기저선 윈도우를 가짜 사건으로 본 BTS (event_id 고유 → 윈도우 단위).
    # BTS=Σmax(0,−z)는 ≥0이므로 H 검정은 '사건 vs 위약'이 정답('vs 0' 아님).
    zb = standardize(df[df["window"] == baseline_window], baselines, direction, match_fields)
    return {
        "z": z,
        "bts_execution": composite_bts(z, we, "execution", side),
        "bts_choice": composite_bts(z, wc, "choice", side),
        "placebo_execution": composite_bts(zb, we, "execution", side),
        "placebo_choice": composite_bts(zb, wc, "choice", side),
        "baselines": baselines,
    }
