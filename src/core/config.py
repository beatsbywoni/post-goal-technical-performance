"""core 가 config 의 지표 메타(클래스·방향·가중)를 읽는 얇은 로더.

도메인 무지 유지: core 는 '데이터로서의' 메타만 받는다(코드에 도메인 지식 없음).
direction 의미: higher_better=+1, lower_better=-1, context=0(BTS 제외 — 선택 트랙 별도).
"""
from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_DIR = {"higher_better": 1.0, "lower_better": -1.0, "context": 0.0}


def load_metric_meta(domain: str, config_dir: Path | None = None) -> dict[str, dict]:
    """{metric: {'class': str, 'direction': float}} 반환."""
    cdir = config_dir or _CONFIG_DIR
    cfg = yaml.safe_load((cdir / f"{domain}.yaml").read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for name, spec in cfg["metrics"].items():
        out[name] = {
            "class": spec["class"],
            "direction": _DIR.get(spec.get("direction", "higher_better"), 1.0),
        }
    return out


def direction_map(meta: dict[str, dict]) -> dict[str, float]:
    return {m: d["direction"] for m, d in meta.items()}


def uniform_weights(meta: dict[str, dict], metric_class: str) -> dict[str, float]:
    """해당 클래스 지표에 균등 가중(λ=1) — U4 1차 설정."""
    return {m: 1.0 for m, d in meta.items() if d["class"] == metric_class}
