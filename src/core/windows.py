"""① 사건 → 윈도우 정렬 유틸 + 롱포맷 프레임 변환 (§3).

core 의 다른 모듈이 공유하는 헬퍼: 맥락키(context-matching) 생성, 레코드↔DataFrame.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

BASELINE = "baseline"
LONG_COLUMNS = [
    "unit_id", "domain", "event_id", "window", "metric", "value", "metric_class", "context",
]


def to_frame(records: Iterable) -> pd.DataFrame:
    """list[LongRecord] 또는 list[dict] → DataFrame (context 는 dict 컬럼 유지)."""
    rows = []
    for r in records:
        d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        rows.append({k: d.get(k) for k in LONG_COLUMNS})
    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    df["context"] = df["context"].apply(lambda c: c if isinstance(c, dict) else {})
    return df


def context_key(context: dict, fields: list[str]) -> tuple:
    """맥락 매칭 키 — 지정 필드값 튜플 (예: (score_state, minute_band))."""
    return tuple((context or {}).get(f) for f in fields)


def add_context_key(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["ctx"] = out["context"].apply(lambda c: context_key(c, fields))
    return out


def split_baseline_post(df: pd.DataFrame, baseline_window: str = BASELINE):
    """기저선 윈도우 행과 사건후(post) 행을 분리."""
    base = df[df["window"] == baseline_window].copy()
    post = df[df["window"] != baseline_window].copy()
    return base, post
