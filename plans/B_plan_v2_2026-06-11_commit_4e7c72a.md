# Preregistration (draft v2 — pivoted) — Behavioral Negativity Asymmetry

**Title.** No behavioral hot hand: success fails to boost skilled performance across
esports and football (with a negativity asymmetry in esports and complacency in football)

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
study applies the identical pipeline to the *positive* side and tests a **negativity
asymmetry**: whether failure leaves a clean behavioral deterioration while success
does not produce a corresponding behavioral improvement, within the same units and
across an esport and a traditional sport. (An exploratory pass, disclosed in §2.3,
showed that a one-sided composite conflates directional change with volatility; the
confirmatory measure here is therefore the *net* oriented z.)

### 1.2 Hypotheses
The signed outcome is the **net oriented z** (z̄; negative = deterioration, positive =
improvement). The confirmatory contrast is event-window z̄ versus matched placebo z̄.
- **A1 (failure scar).** After failure (within-game death cluster; across-game losing
  streak), net z is negative (deterioration) in both domains. SESOI |d| ≥ 0.15, p < .005.
- **A2 (no success boost).** After success (takedown cluster; win streak), net z does
  *not* show improvement; tested by two one-sided equivalence tests (TOST) against
  ±0.15 to affirm the absence of a meaningful positive effect.
- **A3 (football complacency).** After scoring, net z is negative (paradoxical
  deterioration), after matching on score state to control for the lead.
- **A4 (domain-specific asymmetry pattern; not universal).** In esports the absolute
  failure effect exceeds the absolute success effect (|d_failure| > |d_success|), a
  negativity asymmetry; in football, scoring produces a *larger* deterioration than
  conceding (complacency exceeds the conceding effect). The shared conclusion is the
  absence of a success boost (A2), not a universal "bad is stronger than good."
- **A5 (asymmetric dynamics).** Failure deterioration shows a monotone dose–response
  (streak/cluster depth) and rest-decay, whereas success shows no positive dose–response
  — the asymmetry is in dynamics, not only magnitude. Miller–Sanjurjo correction is
  applied to any streak-conditioned estimate.

*(Transparency: draft-v1 framed a symmetric "momentum" mirror with one-sided
composites; an exploratory pass — disclosed in §2.3 — showed that measure conflated
direction with volatility, prompting the pivot to net z and the asymmetry hypotheses
above. The earlier draft is retained in version history.)*

## 2. Data description
- **Source 1 (esport).** Ranked solo-queue *League of Legends* match and timeline data
  from the Korean server via the official Riot Games API. Game-level features (Layer 1;
  N = 92,230 games, 1,715 players) are already extracted; within-game timelines (Layer
  2; 7,998 games) will be re-extracted once to add takedown timestamps required for the
  success-cluster event (the existing extraction retained only death timestamps).
- **Source 2 (sport).** StatsBomb open event data (~1.11M long-format records across
  ten men's and women's competitions), already ingested.
- **Prior knowledge of the data (full disclosure).** I previously analyzed the
  *failure* side of these data (the companion tilt study). For the present study I also
  ran an **exploratory pass**: win-streak events (Layer 1), takedown-cluster events
  (Layer 2, re-extracted with takedown timestamps), and football scoring events. That
  pass revealed that a one-sided composite conflates direction with volatility and that,
  on the net oriented z, success shows no improvement (LoL d ≈ 0) while scoring shows
  paradoxical deterioration (football d ≈ −0.27). The hypotheses A1–A5 and the net-z
  measure above were formulated **in response to** that exploratory finding; they are
  therefore registered as confirmatory tests of an exploratory-generated pattern on the
  full data, with the winsorization, TOST, asymmetry bootstrap, mixed models, and
  robustness battery **not yet run**. This sequence is disclosed in the interest of
  transparency rather than presented as a blind prediction.

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

### 3.3 Outcome metric (execution; dependent)
Execution metrics are identical to the companion study (LoL: per-minute creep-score
deviation, solo-death rate; football: unpressured short-pass accuracy, per-touch
miscontrol, fouls per defensive action). Each is converted to a within-unit, oriented
z (positive = improvement). The **confirmatory outcome is the net oriented z**,
z̄_iw = mean_k z_ikw, per event window. Extreme post-event values (e.g., creep-score
rate after a kill cluster) are **winsorized at the 1st/99th percentile** before
analysis. One-sided composites (Σ max(0, −z) and Σ max(0, +z)) are reported only as
*volatility* descriptors, not as confirmatory evidence, because a salient event raises
both sides together.

## 4. Analysis plan
- **Primary test.** For each event, a standardized mean difference (Cohen's d) of net z
  between post-event windows and matched placebo windows, at five-/ten-/fifteen-minute
  windows (cross-domain anchor: post-10).
- **A1 / A3.** One-sided tests that net z < 0 after failure (A1) and after scoring (A3).
- **A2.** Two one-sided equivalence tests (TOST) that the success-event net-z effect
  lies within ±0.15 — affirming the *absence* of a meaningful improvement (behavioral
  hot-hand absence).
- **A4.** Within-unit contrast of |d_failure| vs |d_success| on the same metrics and
  units, tested by paired bootstrap of per-unit absolute effects.
- **A5.** Spearman trend of net z on failure-run depth (expected monotone) vs
  success-run depth (expected null), with the Miller–Sanjurjo correction on any
  streak-conditioned estimate and a 2,000-sample permutation null; across-game decay by
  inter-game interval band.
- **Inference / SESOI.** Smallest effect size of interest |d| ≥ 0.15 with p < .005;
  absence claims via TOST against ±0.15.
- **Confirmatory models.** Linear mixed models with a random intercept for unit on net
  oriented z, regressing on event status, complementing the effect-size analysis.
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
