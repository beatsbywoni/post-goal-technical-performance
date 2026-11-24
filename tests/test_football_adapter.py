"""src/football/adapter.py 검증 — 합성 StatsBomb 이벤트로 'core 무수정 통과' 증명.

LoL 과 동일한 src.core.tilt_index.run_pipeline 이 축구 어댑터 출력을 그대로 처리하는지
확인한다(이식 주장). 실행: python3 -m pytest tests/test_football_adapter.py -q
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.contract import validate_records          # noqa: E402
from src.core.tilt_index import run_pipeline             # noqa: E402  (LoL 과 동일 함수)
from src.football.adapter import events_to_long          # noqa: E402


def _passrow(team, minute, complete):
    return {"type": "Pass", "team": team, "minute": minute, "second": 0,
            "pass_length": 10, "pass_outcome": None if complete else "Incomplete",
            "under_pressure": None, "shot_statsbomb_xg": None, "shot_outcome": None}


def _synthetic_match() -> pd.DataFrame:
    rows = []
    # 평시(0~29분): A 팀 정확한 단거리 패스 (분별 acc 1.0 또는 0.667 → 분산 확보)
    for m in range(30):
        inc = m % 3                       # 비주기적 변동 → 5분 윈도우 간 분산 확보
        for i in range(3):
            rows.append(_passrow("A", m, complete=(i >= inc)))
    # 실점 직후(30~34분 = post5): A 패스 정확도 급락 (acc ≈ 0.33)
    for m in range(30, 35):
        for i in range(3):
            rows.append(_passrow("A", m, complete=(i >= 2)))
    # B 팀 30분 득점 → A 의 실점 사건
    rows.append({"type": "Shot", "team": "B", "minute": 30, "second": 0,
                 "pass_length": None, "pass_outcome": None,
                 "under_pressure": None, "shot_statsbomb_xg": 0.4, "shot_outcome": "Goal"})
    # v2: 실점 뒤 새 점수상태(0-1)의 평시 구간(50~89분)이 있어야 같은 상태 통제 창이 생긴다
    # (v1 은 사후 창을 골 이전 상태에 붙였기 때문에 이 구간 없이도 통과했다 — 그 자체가 2026-09-14 에 수정된 오류)
    for m in range(50, 90):
        inc = m % 3
        for i in range(3):
            rows.append(_passrow("A", m, complete=(i >= inc)))
    return pd.DataFrame(rows)


def test_adapter_output_passes_contract():
    recs = events_to_long(_synthetic_match(), match_id="m1")
    assert recs, "레코드가 생성되어야 함"
    # 계약 검증 통과(= 잘못된 데이터 아님), 전부 football 도메인
    assert len(validate_records(recs)) == len(recs)
    assert all(r["domain"] == "football" for r in recs)
    # baseline + post5 윈도우가 모두 존재
    windows = {r["window"] for r in recs}
    assert "baseline" in windows and "post5" in windows


def test_core_runs_unmodified_on_football():
    recs = events_to_long(_synthetic_match(), match_id="m1")
    # LoL 과 동일한 run_pipeline — baseline_window 만 규약대로 지정
    out = run_pipeline(recs, domain="football", match_fields=["score_state"],
                       baseline_window="baseline")
    bts = out["bts_execution"].set_index(["event_id", "window"])["BTS"]
    # A 의 실점 사건 post5 에서 실행 BTS > 0 (패스 정확도 급락 검출)
    assert ("concede_0_A", "post5") in bts.index
    assert bts.loc[("concede_0_A", "post5")] > 0


def test_score_transition_label():
    from src.football.adapter import concede_events, extract_goals
    df = _synthetic_match()
    goals = extract_goals(df)
    evs = concede_events(goals, ["A", "B"])
    a_ev = [e for e in evs if e["team"] == "A"][0]
    assert a_ev["score_transition"] == "draw_to_behind"   # 0-0 → 0-1
