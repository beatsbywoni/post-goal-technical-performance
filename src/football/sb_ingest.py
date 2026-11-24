"""StatsBomb 오픈데이터 적재 → 롱포맷 피처 (§5.1). [맥북 실행]

configs/sb_competitions.yaml(U3 확정 대회)을 읽어 경기별 이벤트를 받아 adapter 로
롱포맷 레코드를 만들고 data/features/football_long.parquet 으로 저장한다.
인증 불요(오픈 데이터). 샌드박스는 StatsBomb 호스트 차단 → 맥북에서 실행.

  python3 -m src.football.sb_ingest --limit 1      # 스모크(1경기)
  python3 -m src.football.sb_ingest                # U3 전체 (시간 소요)
  python3 -m src.football.sb_ingest --mode score --fast --cache-dir data/raw/sb_events
      # R1: --fast 는 src/football/adapter_fast (벡터화, 출력 동일) 를 쓰고, --cache-dir 는 경기별 이벤트를
      #     pickle 로 캐시해 재실행 시 내려받지 않는다. adapter.py 와 adapter_fast.py 의 출력이 같음은
      #     tests/test_football_adapter.py 와 analysis/audit/compare_adapters.py 로 확인한다.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore", module="statsbombpy")

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "sb_competitions.yaml"
OUT = ROOT / "data" / "features" / "football_long.parquet"


def selected_comps() -> list[dict]:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    rows = []
    for tier in cfg["tiers"].values():
        rows.extend(tier)
    return rows


def run(limit: int | None = None, max_matches: int | None = None,
        mode: str = "concede", fast: bool = False, cache_dir: str | None = None) -> int:
    from statsbombpy import sb
    if fast:
        from .adapter_fast import events_to_long_fast

        def events_to_long(ev, match_id, competition, event_mode):
            return events_to_long_fast(ev, match_id=match_id, competition=competition, event_mode=event_mode, variant="v2")
    else:
        from .adapter import events_to_long
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    from src.core.contract import validate_records  # 계약 검증

    out_path = OUT if mode == "concede" else (
        ROOT / "data" / "features" / "football_momentum_long.parquet")
    comps = selected_comps()
    if limit:
        comps = comps[:limit]

    all_records: list[dict] = []
    for c in comps:
        try:
            matches = sb.matches(competition_id=c["competition_id"], season_id=c["season_id"])
        except Exception as e:  # noqa: BLE001
            print(f"! matches 실패 {c['competition']} {c['season']}: {e}", file=sys.stderr)
            continue
        mids = matches["match_id"].tolist()
        if max_matches:
            mids = mids[:max_matches]
        print(f"▶ {c['competition']} {c['season']}: {len(mids)}경기", flush=True)
        for i, mid in enumerate(mids, 1):
            cp = cache / f"{mid}.pkl.gz" if cache else None
            try:
                if cp is not None and cp.exists():
                    ev = pd.read_pickle(cp)
                else:
                    ev = sb.events(match_id=mid)
                    if cp is not None:
                        ev.to_pickle(cp, compression="gzip")
            except Exception as e:  # noqa: BLE001
                print(f"  ! events {mid}: {e}", file=sys.stderr, flush=True)
                continue
            recs = events_to_long(ev, match_id=str(mid), competition=c["competition"],
                                  event_mode=mode)
            all_records.extend(recs)
            if i % 50 == 0:
                print(f"    …{i}/{len(mids)} (누적 {len(all_records):,} 레코드)", flush=True)

    # 계약 검증 (잘못된 데이터가 core 에 도달 못 하게)
    validate_records(all_records)
    df = pd.DataFrame(all_records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✓ {out_path} — {len(df)} 롱포맷 레코드 (mode={mode}, 검증 통과)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="StatsBomb → 롱포맷 적재")
    ap.add_argument("--limit", type=int, help="대회-시즌 수 제한(스모크)")
    ap.add_argument("--max-matches", type=int, help="대회당 경기 수 제한(스모크)")
    ap.add_argument("--mode", choices=["concede", "score"], default="concede",
                    help="concede=틸트(실점), score=모멘텀(득점)")
    ap.add_argument("--fast", action="store_true", help="adapter_fast (vectorised, identical output) 사용")
    ap.add_argument("--cache-dir", help="경기별 이벤트 pickle 캐시 디렉터리")
    args = ap.parse_args(argv)
    return run(limit=args.limit, max_matches=args.max_matches, mode=args.mode, fast=args.fast, cache_dir=args.cache_dir)


if __name__ == "__main__":
    sys.exit(main())
