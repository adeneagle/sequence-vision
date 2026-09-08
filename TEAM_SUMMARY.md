# sequence_vision — results summary (2026-08-27/28)

## TL;DR

1. **CryoFM2 is a good density encoder — we had been reading it at the wrong depth and the
   wrong noise level.** At the right (tap, timestep) its features **beat raw density voxels** on
   secondary structure using **2-4x fewer dimensions**. At the configuration the project had been
   using, they lost by 16-17 points. Both free axes had been set near-worst, and neither had ever
   been validated against a downstream metric.
2. **The ESM-C -> density-feature alignment objective should be retired.** On simulated density
   the target is a deterministic function of the atomic model, so `sequence -> feature` is
   strictly dominated by `fold -> featurise` for any feature choice. Separately the objective is
   degenerate: R2 against a chosen feature is improvable by making that feature *less*
   informative.
3. **The missing signal in the alignment is frame-directional local geometry, not neighbour
   identity** — and the efficient way to obtain local geometry from sequence is to fold, not to
   train a pair network. A built-and-tested ESMFold2-style pair network confirmed this: its gain
   was **positional, not relational**.
4. **The "CryoFM2 features are an implicit steerable field" hypothesis is false.** The required
   `Q_R` is structure-dependent wherever the features are informative.

---

## 1. CryoFM2 as an encoder (O1)

Density-side benchmark, REAL experimental maps, 1484 chains, ~55.6k train / 11.3k test residues,
cluster-level split (672 mmseqs clusters @30%). Labels: per-residue secondary structure (3-class)
and amino-acid identity (20-class), both shipped with Cryo2StructData. Baseline = the central
**8^3 = 512 raw voxels of the identical frame-aligned box** (same dim, same locality, no network).

**The verdict moved three times:**

| configuration | SS | AA | reading |
|---|---|---|---|
| `up_blocks[0]`, t=10 *(what the project used)* | 0.588 | 0.113 | loses to raw by 16-17 pts |
| best tap, t=10 | 0.718 | 0.219 | tap-specific, deficit 3-6 pts |
| `up_blocks[0]`, t=500 | 0.713 | 0.134 | timestep worth as much as tap |
| **best tap x best t** | **0.784** | **0.248** | **beats raw on SS; 94.5% on AA** |

Best configurations: `up_blocks[1]` @ t=261 coupled -> **SS 0.7839 vs raw 0.7376** (256 dims, 2x
compression); `up_blocks[2]` @ t=500 -> **SS 0.7736** (128 dims, **4x compression**) and
**AA 0.2475 vs raw 0.2619**.

**Why the original configuration was bad:** `up_blocks[0]` was chosen for maximal per-residue pose
stability (0.826). Pose stability at fine scale is bought by spatial smoothing (~9-10 A
half-decay), which is exactly what destroys locally-decodable detail. t=10 was never chosen at
all — it is the near-clean end of a 1000-step trajectory, whereas the diffusion-features
literature puts the semantic optimum near t=261.

**Optimal depth tracks the physical scale of the task** (a coherence check a broken pipeline
would not produce): SS peaks at 3 A/token (helix pitch 5.4 A), AA at 1.5 A/token (side chains).

**Controls behaved as theory demands:** `conv_in` gives pretrained == random to 3-4 decimals on
both tasks (it is the first convolution, so trained weights cannot help); `up_blocks[3]`
pretrained is *worse* than random, independently reproducing the "pointwise stable, relationally
empty" result from an unrelated method.

## 2. Retiring the alignment objective (O0) and what replaces it

**Structural argument.** The target pipeline is `coords -> DensityCalculator -> volume -> CryoFM
-> activation`; coordinates are the only input. So on simulated density
`sequence -> CryoFM feature` is dominated by `sequence -> structure -> CryoFM feature` for every
choice of tap/timestep/frame. Only folding error separates them.

**Degeneracy.** `R2(ESM-C -> f(density))` improves as `f` becomes less informative (limit:
`f` constant, `R2 = 1`). Already observed twice: a trivial `[N, Rg, composition]` descriptor
scored 0.629 on the homolog diagnostic vs CryoFM's 0.145, and a hand-crafted descriptor family
won that diagnostic and was deleted for exactly this reason.

**Stage 0 — where the missing signal actually is** (ground-truth neighbourhood oracle, same split
as the alignment, all `global` values reproduce the alignment run exactly):

| feature | dim | R2 (global) | R2 (within-protein) |
|---|---|---|---|
| `seqwin3` | 140 | 0.036 | 0.044 |
| `esmc` | 1152 | 0.169 | 0.143 |
| `coord` (9 shell counts) | 9 | 0.165 | 0.119 |
| `shellcomp` (identity + distance) | 180 | 0.175 | 0.128 |
| `bb_only` (backbone dihedrals) | 16 | 0.028 | — |
| **`dirmom`** (frame-directional) | 81 | **0.401** | **0.413** |
| `full_oracle` | 446 | 0.410 | 0.420 |

Neighbour *identity* is near-worthless (+0.006 on top of `dirmom`); the signal is directional
arrangement, and it is tertiary rather than secondary structure (`dirmom_far`, excluding
|i-j|<=4, retains 96%).

**Also measured: 20.4% of the target variance is between-protein.** An oracle knowing only each
protein's mean scores 0.204 — above ESM-C's 0.169. ESM-C loses to that oracle (`r2_within`
-0.044) but retains real per-residue signal (0.143 centred). `dirmom` beats it decisively
(+0.247) and is *stronger* within-protein (0.413) than globally.

**Stage 1 — we built the ESMFold2-style pair network anyway. Its gain is positional:**

| arm | R2 | delta |
|---|---|---|
| baseline (per-residue MLP) | 0.1778 | — |
| + ESM-C outer-product pair channel | 0.1866 | +0.009 |
| + 4 triangle-multiplication blocks | 0.1844 | +0.007 |
| **+ pair channel fed ONLY \|i-j\|, no ESM-C** | **0.2034** | **+0.026** |

A pair channel carrying **no sequence information at all** beats the ESM-C one by 3x, in both of
two optimisation regimes. Provable mechanism: with only relpos, `pooled_i = g(i, L)` exactly — a
learned position-in-chain / chain-length feature. **The actionable win is therefore a few cheap
positional scalars on the per-residue head, not a pair network.**

## 3. The steerable-field hypothesis (O4) is false

Claim: `f(R.x)(Rv) ~ Q_R f(x)(v)` with `Q_R` a function of R alone, so channel space carries an
SO(3) representation decomposable into irreps.

**The evidence that motivated it never tested it.** The existing probe draws rotations *inside*
the per-structure loop and fits a fresh `Q` per (structure, rotation) from ~30 residues. That
shows only that *some* orthogonal map exists per structure — which permits `Q` to depend on the
input, and gives no single representation to decompose.

**Tested properly** (80 structures, shared rotations, one `Q_R` pooled over 48 structures, scored
on 32 held-out structures, all taps rank-sufficient):

| tap | per-structure `Q` | pooled `Q_R` (held out) | homomorphism |
|---|---|---|---|
| `conv_in` | 0.762 | **0.868** | 0.238 |
| `down_blocks[3]` | 0.879 | 0.507 | 0.235 |
| `mid_block` | 0.875 | **0.452** | 0.184 |
| `up_blocks[0]` | 0.817 | 0.473 | 0.100 |
| `up_blocks[3]` | 0.595 | **0.719** | 0.530 |

**A clean dichotomy:** at endpoint taps pooling *helps*, but only because `Q ~ identity` (the
trivial rep, forced by `out_channels: 1`) — a shared identity carries no information. At the
informative deep taps, pooling **collapses the fit by +0.32 to +0.42**. So **no tap has a `Q` that
is both nontrivial and structure-independent**, and `Q_{R1}Q_{R2}` reproduces the `R1R2` features
at only 0.10-0.30. The rotation-consistency is a per-structure alignment, not a group action.

Also: the cheap eigenvalue-degeneracy pre-test is **not measurable** — the statistic reverses
direction with the clustering tolerance, so it measures spectrum shape, not group structure.

**Survivors:** the whitened / general-linear fit (everything above assumes orthogonality), and
training a rotation-consistency term in, which *creates* the structure rather than assuming it.

## 4. Cross-cutting lesson

Every correction this session came from testing a **configuration** instead of trusting a single
operating point: tap and timestep for O1, per-structure vs pooled for O4, global vs per-protein
mean for the R2 metric. A measurement at one arbitrary operating point does not characterise a
representation — it made us too pessimistic about CryoFM's features and too optimistic about
their equivariance.

Three silent-failure bugs were also fixed: non-cubic maps were being dropped (~7%, biased toward
compact particles); extraction had no incremental cache on a partition that preempts hard
(`Restarts=22` observed); and `sbatch --export` truncates comma-containing arguments, which
produced a job that ran to completion while answering a different question.

---

# Figures (BUILT) and tables

Figures: `scripts/figs/team_summary/` (7 PNGs + README). Built by
`scripts/make_team_figures.py`; palette validated with `scripts/validate_palette.py`.
Lead set for a short talk: **F1, F4, F6**.

| file | claim |
|---|---|
| F1_verdict_moved | tap and timestep both set near-worst: −0.162 -> **+0.046** vs raw |
| F2_tap_x_timestep | timestep worth as much as depth; they compose |
| F3_depth_vs_scale | best depth matches the task's physical scale |
| F4_oracle_ladder | missing signal is directional geometry; 20% between-protein |
| F5_pair_channel | pair-channel gain is positional, not relational |
| F6_pooled_qr | no nontrivial structure-independent Q_R -> not a steerable field |
| F7_degeneracy_dropped | appendix: the degeneracy statistic reverses with tolerance |

## Original plan (kept for the design rationale)

## Conventions (apply to all panels)

* **Reuse the palette already established for `scripts/figs/experiment_summary/`** (blue ordinal
  ramp for ordered scales; the baseline-to-beat as a **red dashed horizontal line**). Consistency
  with the existing figure set matters more than novelty. Read the dataviz reference before
  writing chart code.
* **Random-weight control always in grey** — it is a null and should read as background, never as
  a third competing result.
* **INTEGRITY RULE — do not mix absolute accuracies across runs.** The tap sweep used
  `--per-chain 50` (raw = 0.7506 SS / 0.2780 AA); the timestep and composition runs used
  `--per-chain 40` (raw = 0.7376-0.7379 / 0.2619-0.2709). Any panel spanning both must plot
  **delta vs that run's own raw baseline**, with zero = raw. This is also the more legible
  encoding, since "above the line beats raw density" is the whole point.
* State n and the split on every panel: 1484 chains, 11.3k test residues, 672 mmseqs clusters
  @30% identity.

## Figures, priority ordered

**F1 — "The verdict moved three times" (lead figure).**
Waterfall / ordered bars, y = **SS accuracy minus raw voxels**, zero line = raw.
Four bars: `up_blocks[0]`@t10 (−0.163) -> best tap @t10 (−0.032) -> `up_blocks[0]`@t500 (−0.025)
-> best tap x best t (**+0.046**). Annotate bar 1 "the configuration we had been using".
*Carries the whole O1 story in one panel: two independently-set free parameters, each worth ~12
points, and we had both wrong.*

**F2 — Tap x timestep.** Line plot, x = timestep (log, 10/100/261/500), y = delta vs raw, one
line per tap, **markers only on measured points** (the grid is deliberately sparse — full 6-tap
sweep exists only at t=10). Two panels, SS and AA. Circle the project's original operating point.
*Shows the 2-D configuration space and that we sat in its worst corner.*

**F3 — Optimal depth tracks physical scale.** Line plot, x = A/token (log: 1.5, 3, 6, 12),
y = accuracy, two series (SS, AA), t=10 where all six taps exist. Mark peaks: SS at 3 A
(helix pitch 5.4 A), AA at 1.5 A (side chains). *A coherence check, not a performance claim.*

**F4 — Stage 0 oracle ladder (lead figure for the alignment story).** Horizontal grouped bars,
one group per feature (`seqwin3`, `esmc`, `coord`, `shellcomp`, `bb_only`, `dirmom`,
`full_oracle`), three series: **global R2**, **R2 vs per-protein-mean baseline**, **within-protein
centred R2**. Emphasise the zero line (ESM-C's `r2_within` is negative).
*Carries: the gap is directional geometry, not identity; and 20% of the headline R2 was
protein-level. Merging the between/within decomposition into this figure avoids a near-duplicate
panel.*

**F5 — Stage 1: the control wins.** Horizontal bars, four arms (baseline / +ESM-C pair /
+triangle blocks / +relpos-only), baseline as reference line, **relpos-only in a contrasting
alert colour**. Caption the mechanism: `pooled_i = g(i, L)` exactly, i.e. no sequence content.
*Carries: the pair channel's gain is positional, so the cheap fix is positional scalars.*

**F6 — O4 pooled Q_R dichotomy (second lead figure).** Paired bars per tap, taps ordered by
network depth (`conv_in` ... `up_blocks[3]`): **per-structure Q** vs **pooled held-out Q_R**.
Shade the two regimes: endpoints ("Q shared, but Q ~ identity — no information") and deep taps
("Q structure-dependent — no representation"). Overlay the homomorphism score as a line/marker
series on a secondary axis.
*Carries the refutation and, crucially, why it is not merely a weak result but a structured one.*

**F7 — Appendix: why the degeneracy test was dropped.** Line plot, x = clustering tolerance (log),
y = runs of length >= 3, two lines (pretrained, random), two panels (`mid_block`,
`up_blocks[1]`). The lines cross. *Shows the statistic measures spectrum shape, not group
structure — include only if the audience will ask why Task 1 is absent.*

**Minimum viable set for a 10-minute talk: F1, F4, F6.**

## Tables

* **T1 — O1 composition (primary data).** Rows = (tap, dim, setting); columns = SS, AA; raw and
  prior as footer rows. One table per `--per-chain` run, or add a "raw for this run" column.
* **T2 — Stage 0 oracle**, three R2 columns (global / vs per-protein-mean / within-centred) plus
  dim. Footnote that the 0.447 sequence-tracked and 0.849 pose ceilings are global-mean based and
  so are **not** comparable to the within-protein column.
* **T3 — O4 pooled `Q_R`**, all ten taps: channels, train samples/channel, rank-OK flag,
  per-structure, pooled-in, pooled-held-out, drop, homomorphism. Keep the rank column — it is what
  distinguishes this from the underpowered smoke that pointed the other way.
* **T4 — Options register** (O0 deprecated / O1 answered / O2 live / O3 weaker / O4 refuted), each
  with a one-line status and next action. This is the slide people will actually act on.

## Source data

`results/`: `o1_cryofm_benchmark.json`, `o1_{conv_in,mid_block,up_blocks1,up_blocks2,up_blocks3}.json`,
`o1_tsweep.json`, `o1_x{ub1,ub2}.json`, `stage0_oracle.json`, `stage0_within_protein.json`,
`stage1_pair_all.json`, `o4_proc_t{10,261d,261c}.json`, `o4_pooled_qr_t261c.json`,
`o4_spec_{mid,ub1}_t261c.json`.

Render on the login node (no GPU) following the recipe in CLAUDE.md — matplotlib needs the
libexpat preload in the migrated env.
