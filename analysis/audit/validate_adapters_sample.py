"""R1 audit · reference adapter versus vectorised adapter on a targeted sample of matches (shoot-outs, extra time, dismissals,
first-half stoppage-time goals, random): records must be identical.  python3 -m analysis.audit.validate_adapters_sample [CACHE_DIR]"""
import sys, glob, math, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from src.football.adapter import events_to_long
from src.football.adapter_fast import events_to_long_fast
from src.core.contract import validate_records
CACHE = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'data/raw/sb_events')
files = sorted(glob.glob(f'{CACHE}/*.pkl.gz'))
random.seed(3)
# pick: shootout matches, ET matches, red-card matches, P1 stoppage goals, random
picked = {}
for f in files:
    ev = pd.read_pickle(f)
    tags = []
    if (ev.period == 5).any(): tags.append('shootout')
    elif (ev.period >= 3).any(): tags.append('ET')
    for c in ('foul_committed_card', 'bad_behaviour_card'):
        if c in ev.columns and ev[c].isin(['Red Card', 'Second Yellow']).any(): tags.append('red')
    g = ev[((ev.type == 'Shot') & (ev.shot_outcome == 'Goal')) | (ev.type == 'Own Goal For')]
    if ((g.period == 1) & (g.minute >= 45)).any(): tags.append('p1stop')
    for t in tags:
        picked.setdefault(t, [])
        if len(picked[t]) < 4: picked[t].append(f)
    if sum(len(v) for v in picked.values()) >= 16 and all(len(picked.get(k, [])) >= 3 for k in ('shootout', 'ET', 'red', 'p1stop')): break
sample = sorted({f for v in picked.values() for f in v} | set(random.sample(files, 8)))
print('sample', len(sample), {k: len(v) for k, v in picked.items()})
def cmp(A, B, label):
    if len(A) != len(B): print('LEN', label, len(A), len(B)); return False
    nd = 0
    for a, b in zip(A, B):
        for k in ('unit_id', 'event_id', 'window', 'metric', 'metric_class', 'context'):
            if a[k] != b[k]:
                nd += 1
                if nd <= 3: print('DIFF', label, k, a['event_id'], a['window'], a['metric'], a[k], '|', b[k])
        if not math.isclose(a['value'], b['value'], rel_tol=1e-12, abs_tol=1e-12):
            nd += 1
            if nd <= 3: print('VAL', label, a['event_id'], a['window'], a['metric'], a['value'], b['value'])
    return nd == 0
ok = 0; tot = 0; t_real = t_fast = 0
for f in sample:
    ev = pd.read_pickle(f); mid = f.split('/')[-1].split('.')[0]
    for mode in ('score', 'concede'):
        t = time.time(); A = events_to_long(ev, match_id=mid, competition='X', event_mode=mode); t_real += time.time() - t
        validate_records(A)
        t = time.time(); B = events_to_long_fast(ev, match_id=mid, competition='X', event_mode=mode, variant='v2'); t_fast += time.time() - t
        # order-insensitive comparison: sort both by key
        key = lambda r: (r['unit_id'], r['event_id'], r['window'], r['metric'])
        A.sort(key=key); B.sort(key=key)
        r = cmp(A, B, f'{mid}/{mode}'); ok += r; tot += 1
        print(mid, mode, len(A), 'OK' if r else 'MISMATCH', flush=True)
print(f'{ok}/{tot} identical; real {t_real:.1f}s fast {t_fast:.1f}s')
