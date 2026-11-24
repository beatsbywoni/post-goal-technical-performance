"""IJPAS Track A(축구 단독) 표 자료.
 (1) 대회별 표본 표: 대회·성별·시즌·경기·팀-경기 유닛·앵커 윈도우 사건 수
 (2) 지표별 d (post5/10/15, 득점·실점) — 등가중 합성이 한 지표에 끌려가지 않았는지
 (3) 원 단위 Δ short-pass completion (%p) by window
 (4) LMM b (SE) by window
→ docs/tracka_tables.json
  python3 -m analysis.momentum.run_tracka_tables
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.core.windows import add_context_key
from src.core.config import direction_map, load_metric_meta
from src.core.tilt_index import run_pipeline, standardize
from analysis.momentum.run_asymmetry import _d, boot_ci

RNG = np.random.default_rng(7)
EXEC = ["short_pass_acc", "miscontrol_rate", "foul_per_def_action"]
FILES = {"score": "football_momentum_long.parquet", "concede": "football_long.parquet"}
venue = json.loads((ROOT / "data/interim/football_venue.json").read_text())
WOMEN = lambda c: "Women" in c or "WSL" in c or "Super League" in c and "Women" in c

def sex_of(comp): return "Women" if ("Women" in comp) else "Men"

out = {"competitions": {}, "metric_d": {}, "raw_sp": {}, "lmm": {}}
units_by_comp = defaultdict(set); matches_by_comp = defaultdict(set); seasons_by_comp = defaultdict(set)
for mid, v in venue.items():
    matches_by_comp[v["competition"]].add(mid); seasons_by_comp[v["competition"]].add(v.get("season", ""))

for side, fn in FILES.items():
    df = pd.read_parquet(ROOT / "data/features" / fn)
    df["comp"] = df.context.map(lambda c: c.get("competition")); df["mid"] = df.context.map(lambda c: str(c.get("match_id")))
    recs = df.to_dict("records")
    o = run_pipeline(recs, domain="football", match_fields=["score_state"], baseline_window="baseline")
    dirmap = direction_map(load_metric_meta("football"))
    bl = standardize(df[df.window == "baseline"], o["baselines"], dirmap, ["score_state"])
    z = o["z"]
    # units per competition (any analysable window: unit appears in z with finite value or in baseline)
    for u, c in df[["unit_id", "comp"]].drop_duplicates().itertuples(index=False): units_by_comp[c].add(u)
    # events at anchor per competition
    ev10 = z[(z.metric_class == "execution") & z.z.notna() & (z.window == "post10")]
    u2c = dict(df[["unit_id", "comp"]].drop_duplicates().itertuples(index=False))
    ev10 = ev10.groupby(["unit_id", "event_id"]).size().reset_index(); ev10["comp"] = ev10.unit_id.map(u2c)
    for c, n in ev10.groupby("comp").size().items():
        out["competitions"].setdefault(c, {})[f"events_{side}"] = int(n)
    # per-metric d and net-z per window, LMM per window
    import statsmodels.formula.api as smf
    for w in ["post5", "post10", "post15"]:
        for m in EXEC:
            e = z[(z.metric == m) & z.z.notna() & (z.window == w)].groupby(["unit_id", "event_id"])["z"].mean().to_numpy()
            p = bl[(bl.metric == m) & bl.z.notna()].groupby(["unit_id", "event_id"])["z"].mean().to_numpy()
            lo, hi = np.percentile(np.concatenate([e, p]), [1, 99]); e = np.clip(e, lo, hi); p = np.clip(p, lo, hi)
            d = _d(e, p); clo, chi = boot_ci(e, p, n=1000)
            out["metric_d"][f"{side}|{w}|{m}"] = [round(float(d), 3), round(float(clo), 3), round(float(chi), 3), int(len(e))]
            print(f"{side} {w} {m}: d={d:+.3f} [{clo:+.2f}, {chi:+.2f}] n={len(e)}", flush=True)
        # raw short-pass delta
        sub = add_context_key(df[df.metric == "short_pass_acc"].copy(), ["score_state"])
        b = sub[sub.window == "baseline"].groupby(["unit_id", "ctx"])["value"].agg(bm="mean", bn="count").reset_index(); b = b[b.bn >= 3]
        evr = sub[sub.window == w].merge(b, on=["unit_id", "ctx"]).dropna(subset=["value", "bm"])
        per = evr.groupby(["unit_id", "event_id"]).agg(x=("value", "mean"), bm=("bm", "mean")).reset_index()
        dlt = ((per.x - per.bm) * 100).to_numpy(); bs = [RNG.choice(dlt, len(dlt), replace=True).mean() for _ in range(1000)]
        out["raw_sp"][f"{side}|{w}"] = [round(float(dlt.mean()), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2), int(len(dlt)),
                                        round(float(per.bm.mean() * 100), 1), round(float(per.x.mean() * 100), 1)]
        print(f"{side} {w} raw short-pass: {dlt.mean():+.2f} pp n={len(dlt)}", flush=True)
        # LMM
        def collapse(zz, win, ev):
            s = zz[(zz["metric_class"] == "execution") & zz["z"].notna() & (zz["window"] == win)]
            g = s.groupby(["unit_id", "event_id"])["z"].mean().reset_index(); g["event"] = ev; return g
        dat = pd.concat([collapse(z, w, 1), collapse(bl, "baseline", 0)])
        lo, hi = np.percentile(dat["z"], [1, 99]); dat["z"] = dat["z"].clip(lo, hi)
        mm = smf.mixedlm("z ~ event", dat, groups=dat["unit_id"]).fit(reml=True)
        out["lmm"][f"{side}|{w}"] = [round(float(mm.params["event"]), 3), round(float(mm.bse["event"]), 3), float(mm.pvalues["event"]), round(float(mm.cov_re.iloc[0, 0]), 4)]
        print(f"{side} {w} LMM b={mm.params['event']:+.3f} (SE {mm.bse['event']:.3f}) var_unit={mm.cov_re.iloc[0,0]:.4f}", flush=True)

for c in matches_by_comp:
    out["competitions"].setdefault(c, {}).update({"sex": sex_of(c), "seasons": sorted(s for s in seasons_by_comp[c] if s),
                                                  "matches": len(matches_by_comp[c]), "units": len(units_by_comp[c])})
(ROOT / "docs/tracka_tables.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("✓ docs/tracka_tables.json")
