# Replication package: Does technical performance decline after a goal?

Code, derived data, analysis plans and results for

> Jeong, H. (2026). Does technical performance decline after a goal? A within-team comparison with play at the
> same score state in ten football competitions. *International Journal of Performance Analysis in Sport*.
> https://doi.org/10.1080/24748668.2026.2769735

The study compares the technical performance (short-pass completion, miscontrols, fouls) of the scoring and the
conceding team after a goal with the same team's play at the same score state, using StatsBomb open data
(2,404 matches, 6,528 goals, ten men's and women's competitions). Every number in the article and its
Supplementary Material is stored in the JSON files under `results/`. Those files are produced by the code here
from the derived data here, and the derived data are produced by the same code from the open event data.

## Contents

```
plans/
  B_plan_v1_2026-06-10_commit_744b6fc.md     analysis plan, version 1, as committed on 10 June 2026
  B_plan_v2_2026-06-11_commit_4e7c72a.md     analysis plan, version 2, as committed on 11 June 2026
  B_plan_version_history.txt                 dated record of the plan and of every later change
data/derived/                                derived from StatsBomb open data (see "Data source and licence")
  football_momentum_long.parquet             long-format windows around scoring (one row per unit, event, window, indicator)
  football_long.parquet                      the same around conceding
  heldout_football_momentum_long.parquet     held-out replication competitions, scoring
  heldout_football_long.parquet              held-out replication competitions, conceding
  goal_ledger.parquet                        one record per goal (score before and after, time to the next goal,
                                             own goal, penalty, dismissal context); 6,528 goals
  goal_ledger_heldout.parquet                the same for the held-out competitions
  passes_unpressured_short.parquet           pass-level table for the expected-completion model
  match_list.json                            the 2,404 analysed matches (competition, season, match id)
  match_stage.json, match_stage_heldout.json match stage per match (StatsBomb competition_stage)
  football_venue.json                        home and away team and final score per match (opponent quality)
  _v1_submitted/                             windows of the first submitted version, before the October 2026
                                             pipeline correction (kept for the record; see plans/B_plan_version_history.txt)
src/football/                                ingestion and window construction (adapter.py row by row,
                                             adapter_fast.py vectorised with identical output)
src/core/                                    configuration, data contract and within-unit standardisation
analysis/momentum/                           every estimate reported in the article (commands below)
analysis/audit/                              adapter comparison and re-ingestion checks
configs/                                     indicator definitions and competition lists
figures/make_figures.py                      Figures 1 to 4 from the results files
tests/test_football_adapter.py               unit tests of the football adapter
results/R2/                                  final results (second revision, November 2026)
results/R1/                                  results of the first revision (October 2026)
logs/                                        console logs of the second-revision runs
```

The final results are `results/R2/ijpas_r2.json`, `results/R2/ijpas_r2_kickoff.json` and the files
`results/R1/ijpas_r2_*.json` and `results/R1/ijpas_r1_goals.json`, which did not change at the second revision.
`results/R2/ijpas_r2.json` equals `results/R1/ijpas_r2.json` plus the parts added at the second revision
(`<side>|indicators_pooled`, and `clip_denominator_grid` and `later_period_controls` inside `<side>|standard`).
The files `results/R1/ijpas_r1.json`, `ijpas_r1_crosscheck.json`, `ijpas_r1_heldout.json`, `ijpas_r1_passes.json`
and `ijpas_r1_timecourse.json` are superseded first-pass results of the first revision, kept for the record.

The scripts that typeset the manuscript and the Supplementary Material are not included, because they contain
the text of the article. They insert every number from the JSON files here; none was typed by hand.

## Where each result is

`<side>` is `score` or `concede`; `<w>` is `post5`, `post10` (the 5 to 10 min anchor window) or `post15`.
Unless another file is named, keys are in `results/R2/ijpas_r2.json`.

| Article | Results file and key |
|---|---|
| Table 1 | matches and units: `results/R2/tracka_tables.json` (`competitions`); goals: `results/R1/ijpas_r1_goals.json`; events: `<side>\|strata` |
| Table 3 | reference analysis `<side>\|pre\|<w>`, primary analysis `<side>\|state\|<w>` (raw change in `raw`) |
| Table 4, Table S0 (continued, extends Table 4) | `<side>\|sens`, `<side>\|standard`, `<side>\|depend`, `<side>\|state_half\|post10`, `<side>\|exposure_all\|post10`, `<side>\|variant\|v2_own`, `<side>\|variant\|v2_cont`, `<side>\|nearest`, `paired`; restart rows in `results/R2/ijpas_r2_kickoff.json` |
| Table S1 | `<side>\|flow` |
| Table S2 | `<side>\|indicators_pooled` |
| Table S3 | `<side>\|standard` |
| Table S4 | `raw` inside `<side>\|pre\|<w>` and `<side>\|state\|<w>` |
| Table S5 | `<side>\|strata`, `<side>\|oppq`, `<side>\|loco` |
| Table S6 | `<side>\|binom` |
| Table S7 | `results/R1/ijpas_r2_passes.json`; descriptors `<side>\|descriptors` |
| Table S8 | `<side>\|depend`, `<side>\|mdes`, model diagnostics in `<side>\|state\|post10`, `_meta` |
| Table S9 | `results/R1/ijpas_r2_heldout.json` |
| Table S10, Figure 4 | `results/R1/ijpas_r2_timecourse.json` |
| Table S11 | `results/R2/ijpas_r2_kickoff.json` |
| Figure 1 | `<side>\|flow`, `results/R1/ijpas_r1_goals.json` |
| Figure 2 | `cluster_unit` and `cluster_unit_pooled` inside `<side>\|pre\|<w>` and `<side>\|state\|<w>`; `loo_controls` inside `<side>\|standard` |
| Figure 3 | `<side>\|pre\|post10`, `<side>\|state\|post10`, `<side>\|strata` |
| Independent re-computation of Table 3 | `results/R1/ijpas_r2_crosscheck.json` (0 disagreements) |

Only the `competitions` part of `tracka_tables.json` is used in the article.

## Reproducing the results

Run all commands from the repository root.

```
pip install -r requirements.txt
mkdir -p data/features data/interim data/raw docs
```

**(a) Figures from the deposited results** (no analysis is run):

```
cp results/R1/*.json docs/ && cp results/R2/*.json docs/     # the R2 ijpas_r2.json replaces the R1 file
python3 figures/make_figures.py                               # figures/output/Figure_1..4 (.png, .tif, .eps)
```

The regenerated PNG files are pixel-identical to the figures submitted with the article.

**(b) Every estimate from the derived data:**

```
cp data/derived/*football*_long.parquet data/features/
cp data/derived/goal_ledger*.parquet data/derived/match_stage*.json data/derived/passes_unpressured_short.parquet \
   data/derived/football_venue.json data/derived/match_list.json data/interim/
cp results/R1/ijpas_r1_goals.json docs/
python3 -m analysis.momentum.run_ijpas_r2 all                 # docs/ijpas_r2.json (about 3 h on two cores)
python3 -m analysis.momentum.run_ijpas_r2 clipgrid indpooled  # parts added at the second revision (not in "all")
python3 -m analysis.momentum.r2_crosscheck                    # independent re-computation of Table 3
python3 -m analysis.momentum.r1_passes analyse                # docs/ijpas_r2_passes.json
python3 -m analysis.momentum.r1_heldout analyse               # docs/ijpas_r2_heldout.json
python3 -m analysis.momentum.run_tracka_tables                # docs/tracka_tables.json (memory-intensive)
```

Parts of `run_ijpas_r2` can be run separately and merged (one-level deep merge), for example
`R2_OUT=docs/part_main.json python3 -m analysis.momentum.run_ijpas_r2 main cells flow`, then
`python3 -m analysis.momentum.run_ijpas_r2 merge docs/part_main.json`. The two adapter variants in Table S0
(continued) are added with `python3 -m analysis.momentum.run_ijpas_r2 variant v2_own <dir>` and
`... variant v2_cont <dir>`, where `<dir>` holds the windows re-ingested under that variant
(`analysis/audit/reingest_variants.py` and `assemble_variants.py`). Random seeds are fixed in the scripts.

**(c) Derived data and the remaining results from the open event data** (downloads about 2,400 matches):

```
cp data/derived/match_list.json data/interim/                 # the sample (fixed on 4 June 2026)
python3 -m analysis.momentum.r2_fetch_events --cache-dir data/raw/sb_events --workers 4   # re-run until no failures
python3 -m analysis.momentum.r2_fetch_events --heldout --workers 4                       # data/raw/sb_events_heldout
python3 -m src.football.sb_ingest --mode score   --fast --cache-dir data/raw/sb_events
python3 -m src.football.sb_ingest --mode concede --fast --cache-dir data/raw/sb_events
python3 -m src.football.sb_venue                                                 # data/interim/football_venue.json
python3 -m analysis.momentum.r1_heldout ingest data/raw/sb_events_heldout
python3 -m analysis.momentum.r1_goals data/raw/sb_events                         # docs/ijpas_r1_goals.json
python3 -m analysis.momentum.r1_passes extract data/raw/sb_events
python3 -m analysis.momentum.r1_goal_ledger data/raw/sb_events
python3 -m analysis.momentum.r1_goal_ledger data/raw/sb_events_heldout heldout
python3 -m analysis.momentum.r1_match_stage && python3 -m analysis.momentum.r1_match_stage heldout
python3 -m analysis.momentum.r1_timecourse data/raw/sb_events                    # docs/ijpas_r2_timecourse.json
python3 -m analysis.momentum.r2_kickoff check                                    # re-ingestion equals the derived data
python3 -m analysis.momentum.r2_kickoff build                                    # windows with the restart removed
python3 -m analysis.momentum.r2_kickoff estimate && python3 -m analysis.momentum.r2_kickoff shares
# then (b) from run_ijpas_r2 onwards
```

Checks of the pipeline (they need the event cache from (c)):

```
python3 -m analysis.audit.validate_adapters_sample data/raw/sb_events   # row-by-row vs vectorised adapter, sample
python3 -m analysis.audit.reingest_real 0 1 data/raw/sb_events && python3 -m analysis.audit.compare_adapters
python3 -m analysis.audit.reingest_variants 0 1 data/raw/sb_events && python3 -m analysis.audit.assemble_variants assemble
mkdir -p data/features/_v1_submitted && cp data/derived/_v1_submitted/*.parquet data/features/_v1_submitted/
python3 -m analysis.audit.assemble_variants verify                       # v1 variant equals the submitted windows
pytest tests
```

## Reproducibility checks already performed

- The row-by-row adapter and the vectorised adapter give identical records on the full sample
  (2,546,268 records around scoring and 2,537,600 around conceding; `results/R1/ijpas_r1_adapter_comparison.json`).
- All twelve Table 3 cells were re-computed by an implementation written independently of `r1_lib.py`:
  0 disagreements in d, number of events and number of control windows (`results/R1/ijpas_r2_crosscheck.json`).
- The raw events were downloaded again on 8 November 2026, and their re-ingestion reproduced the derived data
  record for record for all 2,404 matches (`results/R2/ijpas_r2_kickoff.json`, key `_cache_check`).
- Figures 1 to 4 regenerated from the results files are pixel-identical to the submitted figures.

## Pre-specification

The analysis plan followed the template for preregistering secondary data analyses but was not deposited in a
public registry, so the study is pre-specified rather than preregistered. The two committed plan versions and the
dated record of every later change are in `plans/`; analyses added after the plan are marked post hoc in the
article.

## Data source and licence

Data source: StatsBomb open data (https://github.com/statsbomb/open-data), used under the terms of that
repository (its `LICENSE.pdf` user agreement), which ask users to state StatsBomb as the data source. The raw
events are not redistributed here; `data/derived/` contains windows, indicators, goal records and pass records
derived from them. Anyone reusing the derived data should also credit StatsBomb and follow those terms.

Code is released under the MIT licence (`LICENSE`). Plans, results and documentation are released under
CC BY 4.0 (`LICENSE-DATA`); the derived data are released under CC BY 4.0 to the extent of the author's rights,
subject to the StatsBomb terms above.

## Software

Python 3.13; numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, glum 3.4, pyarrow 26.0.1,
statsbombpy 1.22.0, PyYAML 6.0.3, matplotlib 3.11.2, pillow 12.3.0, pydantic 2.13.5. Code comments are partly
in Korean.

## Citation

Please cite the article above. To cite this package, use its Zenodo DOI (see `CITATION.cff`).

## SHA-256 of the data, plans and results (first 16 hex digits)

```
25aa1b0151046999  data/derived/_v1_submitted/football_long.parquet
479d27b80a459d1e  data/derived/_v1_submitted/football_momentum_long.parquet
abaca6698df7d3a9  data/derived/football_long.parquet
d462c8845d6e3870  data/derived/football_momentum_long.parquet
8b7537b31879c42c  data/derived/football_venue.json
d28e36d4f09d1f6d  data/derived/goal_ledger.parquet
dacdcffcd85d16cf  data/derived/goal_ledger_heldout.parquet
032415e6c9a0749e  data/derived/heldout_football_long.parquet
534f35bfa23109c4  data/derived/heldout_football_momentum_long.parquet
7e234742f62c79ca  data/derived/match_list.json
e9c0cd73ccecd821  data/derived/match_stage.json
e2505ee38138cd74  data/derived/match_stage_heldout.json
739fd045d2a89d52  data/derived/passes_unpressured_short.parquet
241ca2e8c59648ab  plans/B_plan_v1_2026-06-10_commit_744b6fc.md
64437b6f37c2399f  plans/B_plan_v2_2026-06-11_commit_4e7c72a.md
4b4dfeb253c12a6f  plans/B_plan_version_history.txt
c267ce0783513b84  results/R1/ijpas_r1.json
94babe1aed4dcb2d  results/R1/ijpas_r1_adapter_comparison.json
34749d1531dc722c  results/R1/ijpas_r1_crosscheck.json
bc6a40a0e881710a  results/R1/ijpas_r1_goals.json
ebc35994b5543d6e  results/R1/ijpas_r1_heldout.json
69b62b95723abe3e  results/R1/ijpas_r1_passes.json
5a7c3c5c695f137e  results/R1/ijpas_r1_timecourse.json
0a332a7bed889180  results/R1/ijpas_r2.json
f1a67c91c235233e  results/R1/ijpas_r2_crosscheck.json
4410ad1c4fa67aa4  results/R1/ijpas_r2_heldout.json
52a324518b8e7ae0  results/R1/ijpas_r2_passes.json
286fa6b73297d903  results/R1/ijpas_r2_timecourse.json
ac388dab6a207251  results/R2/ijpas_r2.json
6bb948929a57a345  results/R2/ijpas_r2_kickoff.json
a8369699f14d4044  results/R2/tracka_tables.json
```
