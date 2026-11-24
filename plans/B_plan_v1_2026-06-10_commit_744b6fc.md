# Preregistration (draft v1) — Behavioral Momentum Signature

**Title.** When success compounds: a behavioral momentum signature across esports and
football, and its asymmetry with tilt

**Author.** [withheld for double-anonymous review]

**Type.** Preregistration of secondary-data analysis (template after van den Akker et
al., 2021). Target journal: *Psychology of Sport and Exercise*.

---

## 1. Study information

### 1.1 Background and aims
Psychological momentum — the perception and behavioral reality that success breeds
further success — is a long-standing but contested construct in sport psychology, and
its laboratory cousin, the "hot hand," has been debated since Gilovich, Vallone, and
Tversky (1985) and partly rehabilitated by the selection-bias correction of Miller and
Sanjurjo (2018). A companion study established a *behavioral tilt signature* — a
telemetry-derived measure of performance deterioration after failure — using a
domain-agnostic pipeline applied to *League of Legends* and StatsBomb football. This
study applies the identical pipeline to the *positive* side: whether a run of success
is followed by measurable improvement in execution relative to a within-person,
context-matched baseline, and whether such improvement generalizes across an esport and
a traditional sport. A central aim is to test the **asymmetry** between the behavioral
impact of failure and of success within the same units.

### 1.2 Hypotheses
- **M1 (existence).** Within a game, execution after a success run exceeds
  context-matched placebo windows (behavioral momentum signature, BMS > placebo).
- **M2 (dose–response).** BMS increases monotonically with success-run length, after
  applying the Miller–Sanjurjo correction for streak-conditioning bias.
- **M3 (decay).** Across-game momentum attenuates as the inter-game interval lengthens.
- **M4 (generalization, competing predictions).** After scoring in football, execution
  either (a) improves (momentum) or (b) deteriorates (complacency; Redwood-Brown,
  2008). Both directions are registered; the pure success-run event provides a
  complacency-free cross-check.
- **M5 (failure–success asymmetry).** Within the same units, the absolute behavioral
  effect of failure (tilt) exceeds that of success (momentum): |d_tilt| > |d_momentum|,
  a behavioral negativity bias.

## 2. Data description
- **Source 1 (esport).** Ranked solo-queue *League of Legends* match and timeline data
  from the Korean server via the official Riot Games API. Game-level features (Layer 1;
  N = 92,230 games, 1,715 players) are already extracted; within-game timelines (Layer
  2; 7,998 games) will be re-extracted once to add takedown timestamps required for the
  success-cluster event (the existing extraction retained only death timestamps).
- **Source 2 (sport).** StatsBomb open event data (~1.11M long-format records across
  ten men's and women's competitions), already ingested.
- **Prior knowledge of the data.** I have previously analyzed the *failure* side of
  these same data (the tilt study). I have **not** analyzed the success-side composite
  (BMS), the success-event definitions, or M1–M5 outcomes below. The Layer-2 takedown
  series does not yet exist and will be created after this registration is archived.

## 3. Variables

### 3.1 Events (independent)
- LoL across-game: **win streak** = ≥2 consecutive wins within a continuous session
  (inter-game gap < 12 h); stratified by streak depth (M2) and inter-game interval (M3).
- LoL within-game: **takedown cluster** = ≥2 kills/assists within a 5-minute span,
  anchored on the first takedown (mirror of the death cluster).
- Football: **scoring a goal** (primary), labeled by score transition; **success run**
  (secondary) = a run of successful high-value actions within five minutes.

### 3.2 Baseline and placebo
Within-unit, context-matched baseline (same score state / game-minute band), estimated
from five-minute sliding windows; placebo windows are the unit's own baseline windows
with no preceding success. The confirmatory contrast is event vs. placebo.

### 3.3 Outcome metrics (execution; dependent)
Identical to the companion study. LoL: per-minute creep-score deviation and solo-death
rate. Football: unpressured short-pass accuracy, per-touch miscontrol, fouls per
defensive action. Each is converted to a within-unit z oriented so that positive values
mean improvement; the **behavioral momentum signature** aggregates the improvement side,
BMS = Σ_k λ_k · max(0, +z_ikw), with uniform weights λ_k (discriminative weighting as a
sensitivity analysis).

## 4. Analysis plan
- **Primary test.** For each event, a standardized mean difference (Cohen's d) between
  post-event BMS and matched placebo BMS, at five-/ten-/fifteen-minute windows (primary:
  the post-respawn window for LoL, post-10 cross-domain anchor).
- **M2.** Spearman trend of BMS on success-run length, with the Miller–Sanjurjo
  correction and a 2,000-sample permutation null.
- **M3.** Trend of across-game BMS on inter-game interval band.
- **M4.** Event-vs-placebo SMD after scoring; both improvement and deterioration are
  admissible outcomes; the success-run event is analyzed identically as a cross-check.
- **M5.** Within-unit contrast of |d| for failure (reusing the registered tilt
  estimates) vs success on the same metrics, tested by paired bootstrap of the
  per-unit absolute effects.
- **Inference / SESOI.** Smallest effect size of interest d ≥ 0.15 with p < .005; null
  results evaluated by two one-sided equivalence tests against the same bound.
- **Confirmatory models.** Linear mixed models with a random intercept for unit on the
  oriented execution z, regressing on event status, complementing the effect-size
  analysis.
- **Robustness (registered).** Permutation null, bootstrap 95% CIs,
  leave-one-competition-out (football), leave-one-metric-out, alternative within-game
  windows (3/8 min) and streak depths.

## 5. Exclusions and missing data
LoL: remakes (< 5 min) and abnormal queues excluded; games truncated before a full
post-event window are dropped (as in the companion study), yielding conservative
estimates. Football: windows containing a red card are excluded to remove the manpower
confound. Missing metric values are excluded pairwise per metric.

## 6. Open parameters to be locked before confirmatory analysis
Success-cluster threshold (2+ vs 3+ takedowns / 5 min), the operational definition and
threshold of the football "success run," and the M5 asymmetry contrast (absolute SMD vs
ratio) will be fixed from a registered exploratory subset and reported as such; their
alternatives are sensitivity analyses.

## 7. Open science
Code, configuration files, the de-identified derived features, and this registration
will be posted (anonymized view-only for peer review). Riot data are not redistributed
per the developer terms; only de-identified aggregated features are shared. StatsBomb
data are openly available and cited as a dataset.
