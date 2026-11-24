"""롱포맷 데이터 계약 (§12.2) — core가 아는 유일한 형식.

모든 도메인 어댑터의 출력은 `LongRecord` 로 검증된다. 잘못된 도메인 데이터가
`src/core/` 에 도달할 수 없도록 강제하는 것이 목적이다.

두 층의 검증:
  1) 구조 검증 — `LongRecord` (pydantic): 타입·enum·유한값·schema_ver.
  2) config 정합 검증 — `validate_records`: metric 키가 해당 도메인 config 에
     정의돼 있고 metric_class 가 config 선언과 일치하는지.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

SCHEMA_VER = "v1"
Domain = Literal["lol", "football"]
MetricClass = Literal["execution", "choice", "descriptor"]   # descriptor: v2(2026-10) 카운트·전술 기술 지표, 합성에 미포함

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class LongRecord(BaseModel):
    """롱포맷 한 행 = (unit, event, window, metric) 하나의 관측."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str
    domain: Domain
    event_id: str
    window: str
    metric: str
    value: float
    metric_class: MetricClass
    context: dict[str, Any] = {}
    schema_ver: Literal["v1"] = SCHEMA_VER

    @field_validator("unit_id", "event_id", "window", "metric")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("빈 문자열 불가")
        return v

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value 는 유한 실수여야 함 (NaN/inf 금지)")
        return v


# ── config 정합 검증 ────────────────────────────────────────────
def load_metric_registry(domain: str, config_dir: Path | None = None) -> dict[str, str]:
    """configs/<domain>.yaml 의 metric → class 매핑을 반환."""
    cdir = config_dir or _CONFIG_DIR
    cfg = yaml.safe_load((cdir / f"{domain}.yaml").read_text(encoding="utf-8"))
    return {name: spec["class"] for name, spec in cfg["metrics"].items()}


def validate_records(
    records: Iterable[LongRecord | dict],
    config_dir: Path | None = None,
) -> list[LongRecord]:
    """레코드를 구조 + config 정합 양쪽으로 검증해 정상 레코드 리스트를 반환.

    위반 시 ValueError 를 던진다. 도메인별 registry 는 1회만 로드해 캐시.
    """
    registries: dict[str, dict[str, str]] = {}
    out: list[LongRecord] = []
    for i, rec in enumerate(records):
        r = rec if isinstance(rec, LongRecord) else LongRecord(**rec)
        reg = registries.get(r.domain)
        if reg is None:   # 도메인별 registry 는 1회만 로드(setdefault 는 인자를 매번 평가하므로 쓰지 않음)
            reg = registries[r.domain] = load_metric_registry(r.domain, config_dir)
        if r.metric not in reg:
            raise ValueError(
                f"[{i}] metric '{r.metric}' 가 {r.domain} config 에 없음"
            )
        if reg[r.metric] != r.metric_class:
            raise ValueError(
                f"[{i}] metric '{r.metric}' class 불일치: "
                f"config={reg[r.metric]} != record={r.metric_class}"
            )
        out.append(r)
    return out
