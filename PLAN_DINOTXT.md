# DINO.txt for cryo-EM density <-> sequence: reformulation and critique

Proposal: emulate DINO.txt — frozen vision backbone + trained text tower and lightweight
projection heads, CLIP-style contrastive — with **CryoFM2+CleanDIFT** as the vision backbone,
**ESM-C** as the "text" tower, and **rotation augmentation on the vision side** for
rotation-robust features.

---

## Part 1 — The plan as distinct steps

**S0. Fix the vision operating point.** Choose (tap, timestep, input coupling, granularity)
before anything else is built. Not a detail: measured 2026-08-28, moving from
`up_blocks[0]`/t=10 to `up_blocks[1]`/t=261-coupled changes secondary-structure decodability from
16 points *below* raw voxels to 4.6 points *above* them.

**S1. CleanDIFT distillation.** Fine-tune a copy of CryoFM2 that consumes a CLEAN map and emits
the teacher's features for all t, via zero-init FiLM heads conditioned on t. Output: a
timestep-free, deterministic feature extractor. Heads discarded at inference.

**S2. Choose descriptor granularity.** Global (one vector per map) for map-level retrieval, or
dense (per-residue / per-patch) for pixel-level alignment. DINO.txt does BOTH: `CLS + patch-mean`
for image level, patch tokens for pixel level.

**S3. Build the paired corpus.** Real maps + fitted models + sequences. Cryo2StructData: 7,361
entries at <=3 A; ~3,700 clusters at 30% identity. Cluster-level splits; length-matched
retrieval pools.

**S4. Sequence tower.** ESM-C, frozen, with a learned layer mix (measured: layer 32 > final;
ESMFold2-style mix adds +0.017 over best single layer).

**S5. Projection heads.** Two lightweight heads to a shared embedding space; LiT-style, both
towers frozen, only heads trained.

**S6. Contrastive objective + rotation augmentation.** InfoNCE/SigLIP over (map, sequence) pairs
in a batch; each map presented at random SO(3) rotations so the head learns rotation-robust
embeddings.

**S7. Evaluation.** Bidirectional retrieval (map->sequence, sequence->map) with controls; dense
alignment scored on a per-residue task.

**S8. Downstream application.** The task that justifies the whole thing.

---

## Part 2 — Critical evaluation

Ordered by severity. Every number below is measured in this project, not assumed.

### R1 — FATAL AS SPECIFIED: the global descriptor is explained by trivial covariates

DINO.txt's image-level objective aligns text to a **pooled** vision descriptor. We measured the
pooled CryoFM descriptor directly (homolog diagnostic, 300 homolog + 300 length-matched unrelated
pairs): R2 ceiling **0.145 raw**, and **-0.022 after partialling out `[N, Rg, AA composition]`**.
The trivial descriptor itself scores **0.629**.

So a global contrastive alignment would be aligning ESM-C to approximately *size and amino-acid
composition*. Retrieval would look excellent and mean nothing — and it fails in the exact shape
this project has already been burned by twice (the coordinate-leakage contact probe at AUC 0.997;
the trivial-descriptor win on the homolog diagnostic).

Worse, it is self-reinforcing: contrastive retrieval **rewards** any feature that separates
instances, and size separates instances beautifully.

*Mitigations, all mandatory:* length-matched negative pools (not random negatives); report
retrieval after regressing out (N, R_g, composition); include a "trivial tower" arm whose vision
input is only `[N, Rg]` — if it retrieves comparably, the result is void.

**The deeper implication: invert DINO.txt's emphasis.** Its *pixel-level* alignment is the branch
our data supports — the per-residue local-frame target has **0.447** sequence-tracked variance and
a supervised regression already reaches 0.187. Dense alignment is alive; global alignment is dead.

### R2 — SEVERE: a lightweight head cannot absorb rotation, and augmentation discards signal

The plan assumes rotation is a nuisance the head can learn away. Measured 2026-08-28, it is not:

* Fitting one orthogonal `Q_R` **shared across structures** collapses to **0.45-0.53** at the
  informative taps, versus **0.85-0.88** when `Q` is allowed to differ per structure. The
  homomorphism `Q_{R1}Q_{R2}` vs `R1R2` holds at only **0.10-0.30**.
* So rotation does NOT act as a single global linear map on channel space. A linear (or
  near-linear) projection head can absorb ONE global `Q`; it cannot absorb a
  *structure-dependent family* of them.

What rotation augmentation then does is force the head to **project onto the rotation-invariant
subspace** — i.e. to discard the rotation-varying component. At `up_blocks[1]` the single-frame
per-residue pose cosine under generic SO(3) is ~**0.29**, so roughly 70% of the per-residue
feature content is rotation-dependent. Training the head to be invariant means throwing most of
it away, and nothing in the objective tells you how much useful signal went with it.

*Concrete alternative, and it is strictly better:* **frame-average the features before the head
instead of learning invariance.** Averaging over the 24 lossless octahedral rotations is
transpose+flip (no interpolation), **exactly invariant to that group by construction**, needs no
training capacity, and measured **0.30-0.45 -> 0.63-0.77** pose cosine with per-residue top-1
0.03-0.08 -> 0.09-0.28. A lightweight head cannot learn what this gives for free. Cost is 24x
inference, zero parameters, zero overfitting risk. Rotation augmentation can then be a
*residual* robustness measure on top, not the primary mechanism.

### R3 — SEVERE: sample size is 5 orders of magnitude short for contrastive training

CLIP: ~400M pairs. DINO.txt: >=100M. Here: **~3,700 clusters**. Contrastive objectives are
especially sample-hungry because they learn from *contrasts*, and the number of informative
negatives is bounded by corpus diversity, not by residue count.

Direct evidence from this session: an 8M-parameter pair network on 1,128 training chains reached
best validation at **epoch 4 of 25** and then overfit hard, in two different optimisation
regimes. Frozen towers + tiny heads (LiT / ProteinCLIP regime) is the correct choice and the plan
gets that right — but it means the head will be near-linear, so expect a *small* gain, and do not
budget for a large trained alignment module.

*Mitigation:* dense (per-residue) pairs multiply the effective sample count ~200x, which is
another argument for R1's inversion. But residues within a chain are highly correlated, so the
effective N is far below the nominal count — the same trap that made chain-batched gradients
worse than IID residue batches today (baseline 0.165 vs 0.187).

### R4 — MODERATE: CleanDIFT's measured headroom here is modest

CleanDIFT's real contribution is **removing the `t` hyperparameter**, which we now know matters
a lot (SS +11.7 points from t=10 -> t=500). That is genuinely valuable.

But the *performance* headroom it can recover is the coupled-minus-decoupled gap, and that is
small: retrieval top-1 +0.045 to +0.071 at t>=250 (logged), and in today's classification runs
coupled beat decoupled by only **+0.016** (SS 0.6992 vs 0.6832 at t=261). So do not expect
CleanDIFT to be the thing that makes this work.

Two cautions: it is a **full 168M-parameter backbone fine-tune** on ~7k maps, so it can overfit
the corpus; and it is in tension with S5's "frozen vision tower" — sequencing resolves it
(distil, then freeze), but the plan should say so explicitly.

### R5 — MODERATE: the zero-training baseline is strong and must be in the table

`sequence -> ESMFold2 -> simulate density -> CryoFM features` requires **no training at all**
(9.4 s/1024 residues on an H100) and produces a descriptor in the same space as the vision tower.
For retrieval this is a formidable baseline. It is also the baseline that killed the regression
framing: on *simulated* density the target is a deterministic function of the atomic model.

The contrastive plan's escape is that the vision side here is **real** maps, so the sim-to-real
gap is the content — but that gap has never been measured (Phase 0d, specified and never run).
**If the gap is small, fold-then-simulate wins and no training is justified.** This is the single
cheapest experiment that could invalidate the whole plan, and it should run first.

### R6 — MODERATE: the downstream task is still unspecified

Retrieval of what, for whom? ModelAngelo already identifies proteins from <=4 A maps (build
backbone -> HMM -> proteome search), beating human experts. CryoDomain (AAAI-25) already does
two-tower density<->**structure** retrieval at 5-10 A (SCOPe top-1 57.5%). The unoccupied niche is
density<->**sequence**, which is a real gap — but the plan needs to name the case where
fold-then-search fails. Candidates: resolutions where model building fails (5-10 A), or
sequence-unknown samples where a proteome search is not available.

*Note CryoFM2 is trained at 1.5 A/voxel on SPA maps, so it is OOD at 5-10 A* — the regime where
the niche exists is the regime where the chosen backbone is weakest. That tension is unresolved.

### R7 — GENUINE STRENGTH: contrastive is the right objective for a one-to-many map

Sequence -> density is one-to-many: oligomeric state depends on concentration, pH and ligands,
not sequence alone. A regression objective blurs across those; a contrastive/retrieval objective
does not. This is a real advantage of the plan over everything this project has tried so far, and
it is independent of the criticisms above.

### R8 — GENUINE STRENGTH: it supplies the external metric the project has been missing

Retrieval accuracy is not of our choosing, so it cannot be gamed by making the target less
informative — the degeneracy that sank the regression objective (R2 improves as the target gets
simpler). Both towers' free parameters (tap, t, ESM-C layer) become hyperparameters tuned against
a fixed metric. That is exactly the right structure.

---

## Part 3 — Recommended modifications

1. **Invert the emphasis: dense alignment first, global second.** Global pooled CryoFM has no
   headroom past `[N, Rg, composition]`; the per-residue target has 0.447. Emulate DINO.txt's
   pixel-level branch, not its image-level branch.
2. **Impose rotation invariance by frame averaging, don't learn it.** 24-frame octahedral
   averaging is exact for that group, interpolation-free, parameter-free, and measured to roughly
   double pose consistency. Keep augmentation only as a residual.
3. **Fix S0 empirically before S1:** `up_blocks[1]` or `up_blocks[2]`; **coupled t~500** (peak; collapses by t=900) or **decoupled t~500-900** (saturated plateau, insensitive). Measured 2026-08-30, see GOALS.md high-t sweep.
4. **Sequence the towers:** CleanDIFT distil -> freeze -> train heads. State it, to resolve the
   frozen-vs-fine-tuned contradiction.
5. **Controls, decided before any run:** length-matched negatives; retrieval after partialling
   (N, R_g, composition); a `[N, Rg]`-only trivial tower; the fold-then-simulate arm; and a
   random-weight CryoFM tower (an untrained UNet already reached pooled retrieval top-1 0.58-0.68
   against chance 0.10).

## Cheapest experiments that could kill the plan, in order

* **P1 (hours, CPU).** Retrieval with a **trivial vision tower** = `[N, R_g, AA composition]`
  only. If that retrieves near the ceiling on the intended pool, the global objective is void
  before any model is trained.
* **P2 (Phase 0d; ~1 GPU-day).** Measure the simulated-vs-experimental gap in CryoFM feature
  space. If it is small, fold-then-simulate dominates and no training is justified.
* **P3 (~1 GPU-day).** Measure the **rotation-invariant variance fraction** at the chosen tap,
  with and without frame averaging. That number is the hard ceiling on what any rotation-robust
  head can retain, and it tells you whether augmentation is discarding 20% or 70% of the signal.

Run P1 and P3 before writing any training code; P2 before committing to the training run.

---

# Part 4 — Concrete design (drafted 2026-08-30)

## 4.0 A correction that partly rehabilitates the global objective

R1 above declared the global/pooled objective dead on the basis of the homolog diagnostic:
pooled CryoFM `up_blocks[1]` R2 ceiling **0.145 raw -> -0.022 after partialling `[N, Rg, AA
comp]`**. But that measurement, like every CryoFM number predating 2026-08-27, was taken at
**t=10** — which the timestep sweep then showed to be the worst point on that axis (SS +11.7
points from t=10 -> t=500; the same tap went from 16 points below raw voxels to 4.6 above).

**So "pooled is dead" inherits the bad-operating-point caveat and must be re-measured before it
is believed.** That makes it gate P0 below. If pooled CryoFM at t~500 has real headroom past
`[N, Rg, composition]`, the straightforward global two-tower design is back on the table and this
plan gets much simpler. I flagged the pooled result as decisive earlier without noticing it
carried the same defect I had just spent two days correcting elsewhere.

## 4.1 Two stages, and why the split matters

The data constraint is the whole design problem: ~3,700 paired clusters, versus ~37,000 EMDB maps
at <=4 A. So put every data-hungry operation on the *unpaired* side.

**Stage A — vision tower (unpaired, ~37k maps, no sequences needed).**
CleanDIFT distillation *with a rotation-consistency term* — the user's proposal, and the right
place for it:

    L = L_cleandift( proj_k(f_theta(x_0); t),  f_teacher^k(x_t; t) )
      + lambda * L_rot,    L_rot = 1 - cos( f_theta(g . x)(g . v),  f_theta(x)(v) )

* Target is **invariance at matched material points**, not raw invariance — consistent with
  `out_channels: 1` (a scalar velocity field that co-rotates). Raw invariance would fight the
  denoising objective.
* `g` drawn from the **24 proper cube rotations** for the interpolation-free backbone of the
  loss (verified: all det=+1; a marked right-handed triad keeps its triple-product sign under all
  24, and inverts under a single flip — so chirality is preserved and we are averaging over a
  subgroup of SO(3), not O(3)). Add generic SO(3) at low weight to cover the continuum, since the
  octahedral group is finite and only buys pose 0.63-0.77, not 1.0.
* **Guardrail: monitor the flow-matching loss.** Over-constraining degrades denoising and takes
  feature quality with it. Sweep `lambda` with FM loss as the stopping criterion.
* Note CryoFM2 already pretrains with `RotCube24` at p=1.0, so the octahedral part is partly
  baked in; the marginal gain lives in the generic-SO(3) term.

**Stage B — alignment (paired, ~3,700 clusters).** Freeze the Stage-A vision tower and ESM-C;
train only two small projection heads. LiT/ProteinCLIP regime. This is where the data is scarce,
so this is where the parameter count must be small.

## 4.2 Fixed vision configuration

| choice | value | basis |
|---|---|---|
| tap | `up_blocks[1]` (256 ch) or `up_blocks[2]` (128 ch) | measured: beat raw voxels on SS at 2-4x fewer dims |
| timestep | coupled t~500 / decoupled t~500-900 | measured: +11.7 SS points over t=10; coupled peaks at 500 then collapses (0.640 by t=900), decoupled saturates and is insensitive across 500-900 |
| coupling | coupled (`x_t` at matched t) | measured: coupled >= decoupled at matched t |
| invariance | Stage-A trained; frame averaging as baseline/complement | pending the running K-sweep |
| readout | central 2^3 feature cells | flip-symmetric; a single centre cell misregisters by half a cell under flips |

## 4.3 The pairing scheme

**Positives are multi-positive, not one-to-one.** A map of a complex contains several chains, and
the logged dataset defect (14 of 27 entries with `n_unique_seq > 1`) is exactly this. Use
SupCon-style multi-positive InfoNCE: every chain of a map is a positive for that map. The sibling
project already implements multi-positive InfoNCE (`spatial_contrastive_loss`), so there is
precedent to copy.

**Granularity: run both heads, weight them by what P0 says.**
* *Global* head: pooled map descriptor <-> pooled ESM-C. Enabled only if P0 clears it.
* *Dense* head: per-residue density feature <-> per-residue ESM-C. Alive independent of P0 (the
  per-residue target has 0.447 sequence-tracked variance and 0.187 already achieved).
DINO.txt trains image-level and pixel-level jointly; the same combined loss applies here.

## 4.4 Batch construction — where the anti-triviality work actually goes

**Make the trivial solution unavailable during TRAINING, not merely measured at evaluation.**

* **Length-matched batches.** Sample each batch from a narrow band of sequence length, so
  within-batch length is uninformative and the model cannot use size to separate positives from
  negatives. This is the single most important mechanism in the design; without it the model will
  learn size, because contrastive objectives reward anything that separates instances.
* **Cluster-aware negatives.** Never place two members of one 30%-identity cluster in a batch —
  they are false negatives.
* **Composition-matched hard negatives** (stretch): sample a fraction of negatives with similar
  AA composition, closing the other trivial channel.

**Loss: SigLIP (pairwise sigmoid), not softmax InfoNCE.** The softmax normalisation in CLIP-style
InfoNCE needs large batches to give informative negatives; SigLIP's pairwise sigmoid does not, and
our corpus caps batch diversity at a few thousand clusters. This is a scale-appropriate choice,
not a stylistic one.

## 4.5 Heads

Each tower: `LayerNorm -> Linear -> GELU -> Linear -> L2-normalise`, output 256-d, ~0.3-0.5 M
params. Learned temperature (or SigLIP's learned bias/scale). Deliberately small: an 8 M-param
head on 1,128 chains overfit at epoch 4 in this project's Stage 1 experiment.

Sequence side: ESM-C frozen, with the ESMFold2-style learned scalar mix over layers (measured
+0.017 over the best single layer, and layer 32 > final).

## 4.6 Resolution as a first-class axis (new)

The niche for this model is the regime where **model building fails** — roughly 4-10 A — because
below that ModelAngelo already identifies proteins from maps and beats human experts. But
CryoFM2 is trained at 1.5 A/voxel, so that niche is exactly where the backbone is weakest.

Concrete handling: **resolution augmentation.** Low-pass filter training maps to a sampled target
resolution, train across the range, and report retrieval *as a function of resolution*. Then test
on genuinely low-resolution EMDB entries. This turns the plan's central tension into a measured
curve instead of an assumption, and it is nearly free (a Fourier filter).

## 4.7 Evaluation protocol

Primary: bidirectional retrieval (map->sequence, sequence->map), top-1 / top-10, on held-out
30%-identity clusters, **within length-matched pools**, reported against resolution.

Mandatory arms, all decided in advance:
1. **trivial tower** = `[N, Rg, AA composition]` only;
2. **random-weight CryoFM tower** (an untrained UNet already reached pooled retrieval top-1
   0.58-0.68 against chance 0.10);
3. **fold-then-simulate**: ESMFold2 -> simulate density -> same features, zero training;
4. **retrieval after partialling out** (N, Rg, composition);
5. **ModelAngelo** where it applies (<=4 A), to locate the niche honestly.

## 4.8 Compute budget

| stage | cost |
|---|---|
| A. CleanDIFT + rotation consistency | ~1-2 GPU-days (400-2000 steps; 4-8 forwards/step) |
| B. Dense feature extraction, ~7k maps, cached once | **dominant**; ~1-2 GPU-weeks at 24-frame, ~1 GPU-day at 1-frame |
| C. Head training on cached features | minutes-hours |
| D. Evaluation + controls | ~1 GPU-day |

The dominant term is feature extraction, it is one-time, and its size depends on the invariance
mechanism — which is what the running K-sweep decides. If K=4 suffices, extraction drops 6x.

## 4.9 Gates, in order

* **P0 (~4 h GPU) — re-run the pooled homolog diagnostic at t~500, `up_blocks[1]`.** Decides
  whether the global objective is alive. Highest value per hour in the whole plan, because the
  existing "pooled is dead" verdict is a t=10 artifact until shown otherwise.
* **P1 (hours, CPU) — trivial-tower retrieval.** If `[N, Rg, comp]` alone retrieves near ceiling
  on the intended pool, the global objective is void regardless of P0.
* **P2 (running) — frame-averaging K-sweep.** Sets the invariance cost and the extraction budget.
* **P3 (~1 GPU-day) — Phase 0d, simulated-vs-experimental gap.** If small, fold-then-simulate
  dominates and no training is justified.

Do not start Stage A until P0, P1 and P3 have reported.

---

# Part 5 — REVISION: drop the per-structure descriptor, go voxel-level (2026-08-30)

Direction from the user: not interested in the per-structure descriptor; instead sample **voxels**
with probability related to their distance to the nearest residue, with something analogous to
multi-positive assignment. This is the right call and it changes the design substantially.

## 5.1 It removes the two severest objections

* **R1 (trivial covariates) largely dissolves.** `[N, Rg, AA composition]` are properties of a
  *whole structure*. A voxel has no size. The channel through which the global objective was
  going to cheat simply is not present at voxel level. (Residual care needed only in the
  aggregation step — see 5.4.)
* **R3 (sample size) dissolves.** ~3,700 paired clusters becomes ~10^5 feature cells per map. At
  `up_blocks[1]` (stride 2) a padded 128^3 map yields ~2.6x10^5 cells, maybe 10-20% on protein.
  Even discounting heavily for correlation between neighbouring cells (4-5 A half-decay vs 3 A
  per cell, so adjacent cells are roughly half-redundant), effective N is ~10^6-10^7 rather than
  10^3. Contrastive learning becomes scale-appropriate for the first time in this plan.
* **It also removes the atomic-model dependency at inference.** The fitted model is needed only
  to build training labels (distance to nearest residue). At deployment you embed voxels of an
  unknown map with no model — which is the circularity that made the local-frame per-residue
  features undeployable.

## 5.2 Voxel sampling

Sample voxel `v` with probability `p(v) ∝ K(d(v))`, `d` = distance to the nearest residue (Ca or
any heavy atom). Properties that matter:

* `d` is an **isometry-invariant scalar**, so sampling by it does not bias orientation — the same
  reason the density-percentile criterion was safe in the O4 spectrum probe.
* Include a controlled fraction of solvent/boundary voxels as an explicit **background class**.
  They carry no sequence information, and a model that cannot recognise them will hallucinate
  assignments in solvent.
* Budget ~2-5k voxels per map: 5k x 256 ch x fp16 x 7k maps ~ **18 GB**, cacheable. Caching all
  cells would be ~360 GB, which is not.

## 5.3 Soft multi-positive residue assignment

For voxel `v`, define a distribution over residues:

    w(v, i) ∝ exp( -d(v, i)^2 / 2 sigma^2 )    for d(v,i) <= pos_r
    positives  : residues with d <= pos_r
    EXCLUDED   : pos_r < d <= neg_r      (dead zone, in neither set)
    negatives  : d > neg_r, plus all residues of other proteins in the batch

**`sigma` is not a free parameter — set it from the measured receptive field.** Half-decay is
~4-5 A at `up_blocks[1]`, ~9-10 A at `up_blocks[0]`, ~15 A at `mid_block`. The kernel should match
the tap: a voxel feature is a statement about density within roughly that radius, so the positive
set should be the residues within roughly that radius.

**The dead zone is mandatory, and this project has already paid for learning why.** The sibling
project's `spatial_contrastive_loss` used a shared cutoff (`pos_mult = neg_inner_mult = 1.0`), so a
pair at 0.99 sigma attracted and at 1.01 sigma repelled — "an arbitrarily sharp partition at the
characteristic scale". The logged fix is exactly `neg_inner_mult > pos_mult`, which excludes the
ambiguous annulus from both masks and from the softmax denominator. Reuse that, and note the
logged tradeoff: too wide a dead zone discards informative medium-distance pairs and drops anchors
via the `pos_count>0 & neg_count>0` gate.

## 5.4 Retrieval WITHOUT a pooled descriptor: late interaction

Since there is no per-structure descriptor, score a (map, sequence) pair by **aggregating
voxel-level compatibilities** — ColBERT-style late interaction rather than comparing two pooled
vectors:

    score(map, seq) = (1/|V|) * sum_{v in V}  max_{i in seq}  sim( h(v), g(seq_i) )

* No pooled vector anywhere, so the size/composition shortcut has no place to enter.
* **Normalise by `|V|`** (and consider normalising by sequence length in the max) — otherwise the
  number of sampled voxels or residues re-introduces a size channel through the back door. This is
  the one place R1 can still bite, and it is a one-line fix.
* Cost is O(|V| x L) per candidate, which is expensive for proteome-scale search; approximate
  nearest-neighbour over the residue embeddings is the standard remedy (this is exactly how
  late-interaction retrieval is deployed).

## 5.5 Rotation becomes a different, cleaner problem

This is the most important consequence and it was not available in the global design.

A late-interaction score aggregates over an **unordered set** of voxel features. If rotating the
map moves each feature with its material point but leaves its *value* unchanged — i.e. the
features are equivariant as a **scalar field**, `f(R.x)(R.v) = f(x)(v)` — then the multiset of
features is unchanged, and the score is **rotation-invariant for free**. No frame averaging, no
canonical frame, no local backbone frame.

That is exactly the property `out_channels: 1` forces at CryoFM2's *output*, and exactly what the
Stage-A rotation-consistency term would impose at intermediate layers. So:

* Stage A acquires a **precise, measurable target**: per-voxel pose cosine at matched material
  points, under generic SO(3), in the lab frame. Currently ~**0.29** at `up_blocks[1]` (measured).
  Drive it up; the score's invariance improves monotonically with it.
* The frame-averaging K-sweep now running is still informative (it bounds what an *untrained*
  invariance mechanism buys), but it stops being the primary mechanism. Under this design,
  invariance comes from Stage A + set aggregation, not from averaging at inference — which also
  removes the 24x inference cost.

## 5.6 Risks specific to this design

1. **The information ceiling is unchanged and still unmeasured at the good operating point.** The
   per-residue sequence-tracked fraction is **0.447**, but that was measured at **t=10**. It must
   be re-measured at the fixed operating point (coupled t~500 / decoupled t~500-900). This replaces the (now deprioritised) pooled gate P0.
2. **Soft assignment is a modelling choice, not ground truth.** A voxel 5 A from a Ca is genuinely
   influenced by several residues; `sigma`, `pos_r`, `neg_r` are three coupled knobs and the
   sibling project's experience is that this loss family is sensitive to them. Budget an ablation,
   and prefer the measured receptive field over tuning.
3. **Solvent voxels.** Must be an explicit background class with its own handling, not silently
   mixed into negatives.
4. **Correlated samples inflate apparent N.** Adjacent feature cells overlap receptive fields;
   effective sample size is well below the nominal count, and residues within a chain are already
   known to behave this way (chain-batched gradients cost ~0.02 R2 versus IID residue batches in
   the Stage 1 experiment). Sample voxels sparsely in space, not densely from a few maps.
5. **What it competes with.** Dense voxel->residue matching without model building is a soft
   version of what ModelAngelo does by tracing + HMM search. The niche remains the resolution
   regime where tracing fails (~4-10 A), which is where CryoFM2 is furthest out of distribution —
   hence the resolution-augmentation axis in 4.6 stays load-bearing.

## 5.7 Revised gates

* **G1 (~4 h GPU) — per-residue homolog diagnostic at the fixed operating point (coupled t~500).** Replaces P0. Gives the
  sequence-trackable ceiling for the dense objective at the good operating point; 0.447 is a t=10
  number.
* **G2 (running) — frame-averaging K-sweep.** Reinterpreted: bounds untrained invariance, and its
  K=1 arm is the baseline Stage A must beat.
* **G3 (~1 GPU-day) — voxel-level pose consistency at matched material points**, before and after
  a short Stage-A run. This is now the key Stage-A metric, and it is what makes the set-aggregation
  invariance argument true rather than aspirational.
* **G4 — Phase 0d** (simulated vs experimental gap) unchanged.

---

# Part 6 — Alternative samplers and losses (2026-08-30)

The Part-5 design (voxel anchor, distance-kernel soft positives, dead zone, multi-positive
InfoNCE) is one point in a larger space. Four alternatives are materially different, and two of
them repair defects Part 5 still carries.

## 6.1 Loss: soft-target cross-entropy instead of masked contrastive  **[RECOMMENDED]**

Part 5 turns a smooth distance kernel into **hard masks** (positive / dead-zone / negative). That
is the single most fragile component, and this project has already paid for it: the sibling
project's `spatial_contrastive_loss` used a shared cutoff, so a pair at 0.99 sigma attracted and
at 1.01 sigma repelled — logged as "an arbitrarily sharp partition at the characteristic scale",
with a dead zone as the patch and a documented tradeoff (discards informative medium-distance
pairs, drops anchors via the `pos_count>0 & neg_count>0` gate).

But the assignment `w(v,i) ∝ exp(-d^2/2 sigma^2)` is *already a distribution over residues*. So
use it as a **soft target** and minimise cross-entropy / KL against the model's predicted
distribution:

    p(i | v) = softmax_i( sim(h(v), g(seq_i)) / tau )
    L = KL( w(v, .) || p(. | v) )

* **No masks, no dead zone, no boundary discontinuity.** The annulus problem does not arise
  because nothing is thresholded. CLAUDE.md already names this as the principled fix that "needs
  code": "soft continuous distance-kernel weighting ... removes the discontinuity entirely".
* Multi-positive is native — it *is* a distribution.
* False negatives are down-weighted smoothly rather than being asserted as negatives.
* Still contrastive in effect (the softmax denominator supplies repulsion), so it keeps the
  anti-collapse property that motivated moving away from attraction-only losses.

This is a strict improvement on Part 5's loss and I would make it the default.

## 6.2 Anchor side: bidirectional, not voxel-only  **[RECOMMENDED]**

Part 5 anchors on voxels. That silently **weights each protein by its volume**, so large
complexes dominate the objective and the effective corpus shrinks toward a few big entries. The
dual — anchor on residues, positives are nearby voxels — weights by residue count instead.

Neither is correct alone; CLIP's answer is both directions. Run the symmetric loss
(voxel->residue and residue->voxel) and it also gives a free diagnostic: a large asymmetry between
the two directions indicates a sampling-bias problem rather than a representation problem.

## 6.3 Granularity: region <-> sequence k-mer instead of voxel <-> residue  **[SERIOUSLY CONSIDER]**

A measured fact argues against voxel<->single-residue: **amino-acid identity from density is weak**
— best CryoFM configuration reaches **0.2475** against a 0.089 prior, and raw voxels only 0.2619.
A voxel<->residue objective is therefore trying to learn a mapping whose information content we
have measured to be low.

Aggregate instead: anchor = a local density region (~10-15 A ball, or one feature-map patch);
positive = the **contiguous sequence k-mer** whose residues occupy that region. Composition over
~10 residues is far better determined than any single residue's identity, so the signal-to-noise
is much better.

* Enumerable at inference: all k-mers of a candidate sequence, which is exactly how seed-and-extend
  threading works.
* Caveat: a spatial ball is not sequence-contiguous in packed cores (it is for helices and
  strands), so the k-mer positive is only well-defined for a subset of regions. Either restrict to
  regions with high sequence-contiguity, or accept a soft "fraction of the ball explained by this
  k-mer" target — which composes naturally with 6.1.

## 6.4 Negative-free objective (VICReg / Barlow-style)  **[FALLBACK]**

The false-negative problem here is structural, not incidental: neighbouring voxels overlap
receptive fields, every alpha-helix looks like every other alpha-helix, and homologous residues
are genuinely similar. Methods with no negatives at all (redundancy reduction across the two
modalities) sidestep it completely and are known to work at small batch size.

Worth reaching for if 6.1 still shows false-negative pathology — but it gives no retrieval metric
directly, so evaluation needs a separate probe.

## 6.5 Relational objective  **[FALLBACK, but solves rotation differently]**

Match the **voxel-voxel similarity matrix** to the **residue-residue similarity matrix** (CKA /
RSA style) rather than aligning vectors. Invariant to any orthogonal transform of either feature
space *by construction*, so it needs neither scalar-field equivariance, nor frames, nor averaging
— it is the one option whose rotation handling does not depend on Stage A succeeding.
CLAUDE.md flags exactly this: "A RELATIONAL objective ... would be invariant to Q by construction
and could drop the frame — and the atomic-model dependency with it."

Caveat measured this session: relational *preservation under rotation* is not the same as
relational *task-relevance* — kNN probes on CryoFM features were at or below their random-weight
control on both tasks. So a relational objective may be invariant and uninformative. Test the
relational signal before committing.

## 6.6 Recommendation

Keep Part 5's sampler (distance-kernel voxel selection; sigma from the measured receptive field),
but change two things and pilot a third:

1. **Loss -> soft-target cross-entropy** (6.1). Removes the boundary discontinuity that is the
   design's most fragile part and that this project has already been burned by.
2. **Anchors -> bidirectional** (6.2). Removes the volume-weighting bias and yields an asymmetry
   diagnostic for free.
3. **Pilot region<->k-mer alongside voxel<->residue** (6.3). Given per-residue identity from
   density is only ~0.25, the aggregated formulation is better-posed, and it is the form that maps
   onto threading at inference.

Keep 6.4 and 6.5 as named fallbacks for the two specific failure modes they address
(false negatives; rotation without Stage A).

---

## G2 RESULT (2026-08-30) — imposed rotation invariance is CHEAP; my R2 objection is retracted

`probes/o4_frameavg_benchmark.py`. `up_blocks[1]`, t=261 coupled, real maps, 694/700 chains,
10,353 train / 2,233 test residues, cluster split. Features averaged over K of the 24 proper cube
rotations (all det=+1, chirality verified preserved); readout = central 2^3 feature cells
(flip-symmetric). Cumulative averaging, so all K come from one set of 24 forwards.

| K | secondary structure | Δ | amino-acid identity | Δ |
|---|---|---|---|---|
| 1 (no invariance) | 0.7747 | — | 0.1791 | — |
| 4 | 0.7698 | −0.0049 | 0.1621 | −0.0170 |
| 8 | 0.7653 | −0.0094 | 0.1657 | −0.0134 |
| **24 (exact octahedral)** | **0.7689** | **−0.0058 (−0.8%)** | **0.1661** | **−0.0130 (−7.3%)** |
| *raw voxels (ref)* | *0.7219* | | *0.2051* | |
| *prior* | *0.4456* | | *0.0914* | |

**RETRACTION.** I claimed rotation augmentation would force a head to discard ~70% of per-residue
content. Exact octahedral invariance costs **0.6 points on SS and 1.3 on AA** — wrong by roughly
an order of magnitude. The error was quantifying with pointwise pose cosine (0.29), the metric
this project had already shown understates preserved information by up to 10x (cos 0.064 ->
held-out Procrustes 0.820). **A learned rotation-invariant head, or invariance imposed inside
CleanDIFT, is well-founded: there is very little signal for the constraint to destroy.**

**⚠ SCOPE CORRECTION (2026-08-30).** G2 rotated `R @ fr`, i.e. the *convention* of an already
residue-anchored backbone frame, so **every K in the table above still uses the backbone frame and
still needs an atomic model.** G2 measures convention-averaging, not frame-freeness — which is why
it looked nearly free. The frame-free measurement (`probes/o4_lab_arms.py`, GOALS.md) shows the
frame is worth **~3.2 SS / ~3.6 AA points**, of which averaging at K=4 recovers ~57%, leaving
~1.3 SS / ~1.6 AA. So the retraction above stands for *imposed invariance over a residue frame*,
but "invariance is cheap" must not be read as "the frame is unnecessary".

Three secondary readings:
* **The cost is scale-dependent**, ~9x larger in relative terms for the FINE task (AA −7.3%) than
  the coarse one (SS −0.8%). Mechanistically consistent with the timestep sweep, where coarse
  tasks gained from smoothing and fine ones barely moved.
* **The cost does not grow with K** (K4 ≈ K8 ≈ K24): it is a one-off, not progressive. So **K=4
  buys essentially all the invariance at 1/6 the inference cost** — which cuts the dense
  extraction budget ~6x if averaging is used at all.
* Per-arm SE is 0.0088 (SS) / 0.0081 (AA) on n=2,233. The SS drops are sub-SE (not distinguishable
  from zero); the AA drop is ~1.6 SE, so small but plausibly real.

**Caveat that bounds the generalisation.** The octahedral group is exactly CryoFM2's pretraining
augmentation (`RotCube24`, p=1.0), so this is the EASY case — the same trap CLAUDE.md flags about
the original Phase 0a gate. **Generic SO(3) invariance is not bounded by this number**, and
octahedral averaging only reaches pose 0.63-0.77, not 1.0. Buying full SO(3) invariance is what
the Stage-A consistency term would have to do, and its cost remains unmeasured.

---

# Part 7 — Stage A design, given that invariance is cheap (2026-08-30)

## 7.1 First: does Stage A need to exist?

G2 showed octahedral invariance costs 0.6 (SS) / 1.3 (AA) points, and that **K=4 buys as much as
K=24**. That opens a simpler alternative: **skip Stage A entirely and just frame-average at
inference**, 4x cost, zero training. The case for Stage A over that is threefold:

1. averaging covers only the finite subgroup (pose 0.63-0.77, not 1.0), whereas training can
   target full SO(3);
2. averaging costs 4x on **every** inference forever, and the voxel-level design forwards ~10^5
   cells x ~7k maps — where 4x is a real budget line, not a rounding error;
3. CryoFM2 is *already* augmented with `RotCube24` at p=1.0, so cube-consistency has little left
   to teach it. **The marginal value of Stage A lives entirely in generic SO(3).**

**REFRAMED 2026-08-30 — cube-vs-SO(3) was the wrong axis.** Both looked cheap because both kept
the backbone frame. The axis that costs accuracy is **frame vs no frame**: frame-free is
-3.2 SS raw, -1.3 SS with K=4 averaging. So the target for Stage A is not "beat octahedral
averaging on generic SO(3)" but **"recover the ~1.3 SS / ~1.6 AA residual that averaging
leaves, without paying 4x inference"** — and, since the deficit is a linear-readout effect,
a nonlinear invariant head is the natural first attempt. If it cannot beat K=4 averaging,
average at inference and do not train at all.

## 7.2 The caveat G2 does NOT cover, and it sets the guardrail

G2 measured invariance **imposed post-hoc by averaging**, which *projects onto* the invariant
subspace and therefore preserves whatever signal lives there. **Training for invariance is not the
same operation**: a network can satisfy an invariance penalty by collapsing, and constant features
are perfectly invariant. So G2 establishes that the invariant subspace retains the task signal —
necessary, not sufficient.

Consequence: **the Stage-A guardrail must be a task probe, not just the flow-matching loss.**
Run the O1 SS/AA probe on the student at checkpoints; the harness already exists and is cheap.
Watching only the FM loss would miss a collapse that keeps denoising adequate while flattening
the features.

## 7.3 Objective

    L = L_clean + lambda_rot * L_rot        (+ the FM loss if the teacher is also updated)

**CleanDIFT part.** Student = copy of CryoFM2 consuming clean `x_0` at a single learned timestep;
teacher = frozen original on `x_t` at sampled `t`; zero-init FiLM heads conditioned on `t`; cosine
loss; heads discarded at inference.

**Deviation from the reference recipe, and it is measured:** CleanDIFT samples `t` stratified over
1..999 because for natural images all `t` carry signal. **We have measured that ours do not** —
t=10 features sit 16 points below raw voxels on SS while t=500 sits above them. Distilling
uniformly would force the student to also explain the useless low-t regime. So **truncate or
reweight the `t` distribution to the measured-useful range**; job 3482030 (t = 500/750/900, both
arms) fixes the upper end.

**Rotation part.** Target invariance at matched material points, `f(R.x)(R.v) = f(x)(v)`, which is
the property `out_channels: 1` already forces at the output.

**Use two independently-rotated views, not (original, rotated).** If one side is the un-resampled
original and the other is interpolated, the loss is asymmetric and the model is partly trained to
match a resampling artifact. Comparing `f(R1.x)` against `f(R2.x)` at matched points puts the same
interpolation burden on both sides. This is the same reasoning already logged in this project:
"rotating an already-sampled volume double-resamples and overstates error".

**Sampling `g`:** weight generic SO(3) heavily and cube rotations lightly — the opposite of what
exactness alone would suggest, because the exact group is the one the model already satisfies.
Use tricubic or Fourier-domain rotation rather than trilinear to keep the target clean.

**`lambda_rot` can be aggressive.** G2 says there is little signal for the constraint to destroy,
so the usual worry about over-constraining is weaker than it would otherwise be — subject to 7.2's
guardrail catching collapse.

## 7.4 Data, cost, schedule

* **Unpaired maps only** — 7,361 Cryo2StructData + 1,146 ours, extensible to ~37k EMDB at <=4 A.
  No sequences, so the paired-data bottleneck does not apply here. This is why Stage A is the
  right place to spend capacity.
* Reference CleanDIFT is 400 steps / batch 8 / ~30 min on one A100 for 2D. Ours is 3D at 64^3 with
  2-3 extra forwards per step for the rotation views: budget **1-3 GPU-days** for a few thousand
  steps.
* Full fine-tune (168 M is small enough; LoRA is only needed for much larger backbones).

## 7.5 Monitoring — four numbers, checkpointed

| metric | now | want |
|---|---|---|
| per-voxel pose cosine, matched material points, generic SO(3), lab frame | **~0.29** (`up_blocks[1]`) | up |
| SS probe | 0.7747 | flat (collapse guard) |
| AA probe | 0.1791 | flat (collapse guard) |
| flow-matching loss | baseline | not materially worse |

The first is the actual objective of Stage A and the thing that makes the set-aggregation
invariance argument in 5.5 true rather than aspirational. The middle two are the collapse guard
that 7.2 requires. The last is the denoising guardrail.

## 7.6 Ablations worth budgeting

1. `lambda_rot` sweep (0, small, large) against all four monitors.
2. cube-only vs SO(3)-only vs mixed `g` — tests the 7.3 claim that the marginal value is in SO(3).
3. truncated vs full `t` distribution in the distillation — tests 7.3's measured deviation.
4. Stage-A student vs **frame-averaged frozen teacher** at matched inference cost. This is the
   honest baseline and it is the one that decides whether Stage A earned its keep.

---

## G2b RESULT (2026-08-30) — GENERIC SO(3) invariance is ALSO free; Stage A loses its justification

`probes/o4_frameavg_benchmark.py --rotation so3`. `up_blocks[1]`, t=261 coupled, 395/400 chains,
5,900 train / 1,313 test residues. **One trilinear resample in every arm** (boxes cut directly in
the rotated frame `R @ fr`, never cut-then-rotated), with `R_0 = I` so K=1 is the canonical local
backbone frame — the same baseline definition as the cube run.

| K | SS | Δ | AA | Δ |
|---|---|---|---|---|
| 1 (canonical frame) | 0.7776 | — | 0.1622 | — |
| 4 | 0.7753 | −0.0023 | 0.1714 | +0.0091 |
| 8 | **0.7906** | **+0.0129** | 0.1645 | +0.0023 |
| 24 | 0.7730 | −0.0046 | 0.1637 | **+0.0015** |
| *raw voxels* | *0.7152* | | *0.1851* | |
| *prior* | *0.4372* | | *0.0914* | |

Per-arm SE ~0.011 (SS) / 0.010 (AA), so **every difference — including the two positive ones — is
within one standard error of zero.**

**This refutes the caveat I attached to G2.** I argued the octahedral result was the easy case
because the octahedral group is CryoFM2's own pretraining augmentation (`RotCube24`, p=1.0), and
that generic SO(3) would be materially harder. It is not. Combined:

| group | SS cost | AA cost |
|---|---|---|
| octahedral (24, lossless) | −0.006 | −0.013 |
| **generic SO(3) (24, interpolated)** | **−0.005** | **+0.002** |

The AA −0.013 in the cube run does not reproduce and is best read as noise on that subset.

### Consequences

1. **Stage A's invariance justification is dead.** Part 7.1 made Stage A conditional on generic
   SO(3) being harder than octahedral; it is not. Frame averaging at **K≈4-8** delivers
   SO(3)-invariant features at no measurable task cost, with no training, no `lambda` to tune, and
   no collapse risk (averaging *projects onto* the invariant subspace; training could collapse
   into it — the distinction in 7.2). **Recommendation: average, do not train.**
   Stage A's only surviving rationale is CleanDIFT's removal of the `t` hyperparameter, which is a
   much narrower claim and should be argued on its own merits, not on rotation.
2. **The local backbone frame is no longer required.** K=1 is the canonical N-CA-C frame, and
   averaging over random rotations of it washes out that choice at zero cost. So the feature needs
   residue *positions* but not *orientations* — and in the voxel-level design not even positions,
   since voxels are sampled. Cut in a lab-aligned frame, average over K rotations: **SO(3)-invariant
   voxel features with no atomic model anywhere.** That is the circularity which made per-residue
   features undeployable, dissolved.
3. **The metric lesson, twice over.** Pointwise pose cosine at `up_blocks[1]` is ~0.29, and I used
   it to predict a ~70% loss of content. Actual cost: ~0.5% on SS. Pointwise feature similarity is
   not task-relevant content — the features really do change under rotation, but the part that
   predicts the label survives averaging almost entirely.

### Honest limits of this result

* n=1,313 test residues bounds the cost at roughly **<0.02** (2 SE); it does not prove exactly zero.
* Boxes are centred at Ca positions. The voxel-level design samples arbitrary voxels, where the
  neighbourhood is not residue-centred; invariance cost there is untested, though there is no
  obvious mechanism for it to differ.
* Both probes are per-residue *classification*. A contrastive/retrieval objective may be more
  sensitive to fine structure that averaging blurs. Untested, and worth one probe before Stage B.
* Measured at 1.5-3 Å-regime maps; the low-resolution regime (4-10 Å) that motivates the niche is
  untested.

---

# Part 8 — IMPLEMENTATION PLAN (2026-08-31)

**Status of this file before Part 8: design and critique only. No DINO.txt code exists.** Verified
by enumeration of `probes/`, `teachers/`, `data/`, `scripts/`: there is no contrastive trainer, no
two-tower module, no voxel sampler, no retrieval harness, and `GOALS.md` contains zero occurrences
of "dino". Parts 1-3 are the critique, Parts 4-7 are superseded designs (4 = per-structure
descriptor, killed by R1; 7 = Stage A, killed by G2b). **Part 8 supersedes Parts 4 and 7 and is the
authoritative implementation plan.** Parts 5 and 6 survive as the design this one implements.

## 8.0 Premise check — what actually exists, as of today

| asset | state | where |
|---|---|---|
| CleanDIFT student (the encoder) | **done**, 4 arms + 2 seeds | `data/cleandift_runs/distill_t1000_s0_paperhead/best.pt` (730 MB) |
| preprocessed volumes, whole corpus | **done**, 1147/1147 maps, 70 GB | `data/cleandift_vols/{emd:04d}.npy` |
| ESM-C per-residue, 36 layers | **done**, 1500 chains, 29 GB | `data/esmc_layers/` |
| cluster split, leak-asserted | **done**, 1500 chains / 1147 maps / 672 clusters | `data/alignment_chains.csv` |
| whole-map feature stitching | **done**, patch-alignment fix in | `CryoFM2Tap.feature_volumes` |
| arbitrary-voxel feature sampling | **done** | `teachers/cryofm_tap.py:sample_at` |
| paired cluster bootstrap / MDE | **done**, unit-tested | `probes/o5_stats.py` |
| **DINO.txt alignment** | **nothing** | — |

**The encoder premise is real but narrower than "useful downstream features".** Quote it precisely:
`student_paperhead` SS 0.7908 vs best deployable clean-input timestep 0.7760 (**+1.5 pt**, CI
[+0.0087, +0.0226]) and vs raw 8³ voxels 0.7475 (+4.3). Causally attributed: both null controls
(`student_ctrl` +0.0012, `student_ctrlsampled` +0.0009) are null, so the gain requires the noisy
teacher. **On amino-acid identity there is no gain at all** (+0.0033, ns) and raw voxels still beat
every arm by ~5 points (0.2735 vs 0.2204). So the encoder is better at *coarse/backbone* structure
and not at residue identity — which constrains the objective (see 8.4 and P14).

**What CleanDIFT buys this stage, and it is bigger than the +1.5 points:** the student consumes a
**clean** input at a single **learned** timestep, so `timestep` and the coupled/decoupled choice —
two axes that dominated every earlier result and cost a full sweep to settle — are **gone**. One
deterministic forward per map. That is what makes 8.5 possible.

### 8.0.1 ★ A measurement that reshapes the design: the maps are heterogeneous complexes

Ran today over 150 randomly sampled alignment maps (`gemmi`, polymers ≥20 residues):

| quantity | median | max | fraction of maps |
|---|---|---|---|
| chains per map | **12** | 56 | 150/150 multi-chain |
| **distinct sequences** per map | **9.5** | 56 | 138/150 have ≥2, 126/150 have ≥3 |

Two consequences, pulling in opposite directions:

1. **It supplies the primary endpoint** (8.6). A within-map, ~9.5-way "which sequence owns this
   density" task is the direct analogue of dino.txt's open-vocabulary segmentation, it is
   within-map so the size/composition shortcut (R1) has no purchase, and the ground truth is the
   deposited multi-chain model we already have.
2. **It is a data defect at scale.** `alignment_chains.csv` carries **1.3 chains per map** while the
   maps contain a median of **12**. This is the previously-logged "14 of 27 entries have
   `n_unique_seq > 1`" defect, now measured properly and much larger than logged. Every existing
   per-residue number is only mildly affected (the receptive field is 4-5 Å at `up_blocks[1]`, so a
   Cα-centred feature is mostly about its own chain), but **any voxel-level or pooled objective is
   broken by it unless the target is built over all chains**: a voxel sitting on chain B is neither
   a positive for chain A's sequence nor background nor a valid negative. See P8.

## 8.1 What "dino.txt-style" means here, and the three deviations

dino.txt (Jose et al., CVPR 2025, [arXiv:2412.16334](https://arxiv.org/abs/2412.16334)) is LiT with
two changes: the aligned image vector is `g = [c' ; σ(f'_1..f'_N)]` — the global token concatenated
with the **average of the dense tokens** — and two learnable transformer blocks sit on top of the
frozen backbone, with a trainable text tower and a contrastive objective. The concat is the whole
trick: putting the dense mean inside the *globally* aligned vector makes patch-level alignment
emerge, which is how they get open-vocabulary segmentation out of an image-level loss.

Mapping, and where it must not be copied:

| dino.txt | here | deviation |
|---|---|---|
| frozen DINOv2 | frozen CleanDIFT student, tap `up_blocks[1]` (256 ch, 3 Å/cell) | none |
| [CLS] token | **does not exist** (UNet, no class token) | must be *constructed* by pooling — and pooling is exactly what we measured to destroy the signal |
| patch tokens | voxel feature cells | none |
| trainable text encoder (from scratch) | frozen ESM-C + trainable head | **LiT on both sides.** No reason to train a protein LM; ESM-C is the stronger tower |
| 2 trainable ViT blocks on vision | 2 trainable 3D residual conv blocks | self-attention over 32³ = 32,768 tokens is not affordable |
| dense alignment **emerges** from the global loss at web scale | **must be supervised explicitly** | D1 below |

**D1 — do not rely on emergence.** dino.txt's dense capability emerges from an image-level loss
over ~10⁸ pairs. We have **466 clusters** after the map-level split. Supervise the dense term
directly (Part 6.1's soft-target cross-entropy) and treat the global term as an *auxiliary* whose
weight is ablated, i.e. invert dino.txt's emphasis.

**D2 — the global slot is a liability, not a free addition.** `[N, R_g, AA composition]` scores
R²-ceiling **0.629** on the homolog diagnostic, and the pooled CryoFM descriptor has **no headroom
past it** (0.145 → **−0.022** after partialling). In dino.txt the global channel is benign; here it
is the one channel we have measured to be trivially solvable. So `λ_global` starts small and
`λ_global = 0` is a mandatory arm, not an ablation afterthought.

**D3 — no frames, and no Stage A.** G2b settled this: generic SO(3) invariance costs ≤0.02 via
K=4-8 frame averaging, with no training. Voxels are sampled in the lab frame, so the atomic model
is needed only to *build targets*, never at inference. Stage A is cancelled (see G2b consequence 1).

## 8.2 Fixed configuration

Everything here is chosen from a measured number, not by preference. Do not sweep these in v1.

* **Encoder:** `distill_t1000_s0_paperhead/best.pt`, loaded via `CryoFM2Tap.load_student` (never
  hand-rolled — the D2 embedding swap must precede `load_state_dict`; see P10).
* **Tap:** `up_blocks[1]`, 256 ch, 3 Å/cell, ~4-5 Å half-decay. It is the tap CleanDIFT was
  measured at and the tap that beat raw voxels on SS by +4.6. Secondary arm: `up_blocks[2]`
  (128 ch, 1.5 Å/cell), which is the AA-best tap (0.2475) and therefore the better bet if the
  fine-grained endpoint E3 turns out to matter.
* **Timestep:** not a parameter. The student's `LearnedTimeEmb` returns a constant, so
  `feature_volumes(timestep=...)` is **ignored**. Pass the default and assert two different
  timesteps give bitwise-identical features — a 5-second check that documents the property (P17).
* **Split:** `data/alignment_chains.csv` via `probes/cleandift_data.py:split_map_lists`, unchanged.
  **The student was trained on the train maps of this exact split, at map level, with the D11
  assertions** (no train map or train cluster in val/test). So the encoder is already split-clean
  for this stage — verified, not assumed. Do not re-derive a new split; it would invalidate that.
* **Sequence side:** `data/esmc_layers` layer **32** (best single layer, R² 0.184; the full 36-layer
  ESMFold2 mix adds only +0.017 and costs 36× the cache — see 8.9 note on the expanded chain set).
* **σ of the positive kernel:** 4 Å, from the measured `up_blocks[1]` half-decay (4-5 Å). Not tuned.

## 8.3 Architecture

```
DENSITY TOWER                                  SEQUENCE TOWER
vol [Z,Y,X] (cached, preprocessed)             ESM-C layer 32 [L, 1152] (cached, fp16)
  -> frozen student, feature_volumes             -> LayerNorm  (load-bearing: per-layer RMS
     -> FeatureVolume up_blocks[1] [256,z,y,x]      spans 1.37..108 across depth)
  -> sample_at(voxel coords) -> [V, 256]         -> MLP 1152 -> 512 -> 256
  -> 2x trainable block (see 8.5 for the           (GELU, zero-init residual branch)
     pointwise-vs-conv decision)                 -> L2 normalise -> g_i  [L, 256]
  -> linear 256 -> 256, L2 normalise -> h_v
```

Trainable parameter count is ~1-2 M. Both foundation models stay frozen. Zero-init every residual
branch and the output projection of any added block, per this project's standing rule (a 16-block
residual decoder once diverged to R² −371 without it).

Learnable temperature `τ`, init 0.07, clamped to [0.01, 1.0]. **τ interacts with feature magnitude**
— the sibling project logged that a loss weight tuned at one temperature does not transfer — so
recalibrate `λ_global` whenever `τ`'s init or the normalisation changes.

## 8.4 Objective

**Primary (dense), bidirectional soft-target cross-entropy — Part 6.1 + 6.2.** For voxel `v`, build
a target distribution over **all residues of all chains present in the map**, plus one background
token:

```
w(v, i)  ∝  exp( -d(v, i)^2 / (2 sigma^2) )        sigma = 4 A, d = distance to residue i's heavy atoms
w(v, BG) ∝  exp( -d_min(v)^2 / (2 sigma^2) ) ... complement, so solvent voxels put mass on BG
p(i | v) =  softmax_i( <h_v, g_i> / tau )
L_v2r    =  KL( w(v, .) || p(. | v) )
L_r2v    =  the dual: anchor on residue i, target over sampled voxels
L_dense  =  0.5 * (L_v2r + L_r2v)
```

* **No hard masks, no dead zone, no boundary discontinuity.** The kernel is used as a soft target
  rather than thresholded, which is the fix CLAUDE.md names as "needs code" for the sibling
  project's 0.99σ-attracts / 1.01σ-repels pathology. Nothing is thresholded, so the pathology
  cannot arise.
* **Bidirectional** because voxel-only anchoring weights each protein by its *volume*, so 56-chain
  complexes would dominate; residue-anchoring weights by residue count. Neither is right alone, and
  the asymmetry between the two directions is a free sampling-bias diagnostic (Part 6.2).
* **Background is an explicit class with its own embedding**, never silently folded into negatives.
* Negatives: all residues of all *other maps* in the batch, plus within-map other chains. **Batch
  by cluster** so no two chains from the same 30%-identity cluster co-occur (P7).

**Auxiliary (global), the dino.txt concat — ablated.** `G_map = [pool(h) ; mean(h_dense)]`,
`G_seq = [pool(g) ; mean(g_residue)]`, InfoNCE with **length-matched** negatives.
Arms: `λ_global ∈ {0, 0.1, 1.0}`. D2 says 0 may well win.

**Fallbacks, named now so they are not invented under pressure:** if false-negative pathology shows
up (every α-helix resembles every other), switch the dense term to VICReg/Barlow redundancy
reduction (Part 6.4) — no negatives at all. If rotation turns out to cost more than G2b predicts at
non-residue-centred voxels, switch to the relational objective (Part 6.5), which is invariant to any
orthogonal transform by construction. Do **not** reach for either without the measurement that
justifies it.

## 8.5 The precompute trick — why this stage is cheap, and its one catch

Both towers are frozen and the student is deterministic on clean input, so the density features are
a pure function of (map, rotation). **Precompute them once.**

* 2-5 k sampled voxels/map × 256 ch × fp16 × 1147 maps ≈ **1.5-3 GB** per rotation.
* Cache K = 8 fixed random SO(3) rotations per map → **~24 GB**, which also *is* the rotation
  augmentation. (Caching whole feature volumes instead would be ~134 MB/map fp16 → 154 GB for one
  rotation and is not worth it.)
* Training then touches no 3D UNet at all: a ~1-2 M parameter head over cached tensors. **Minutes
  per run, dozens of ablations affordable, single GPU or even CPU.** This is the LiT regime that
  dino.txt itself exploits, and it is why this stage is far cheaper than the CleanDIFT stage.

**The catch, and the resulting v1/v2 split.** dino.txt's two trainable vision blocks are
*convolutional/attentional over the token grid*, so they need spatial neighbours — which caching
individually-sampled voxels destroys. Therefore:

* **v1: pointwise trainable heads on cached sampled voxels.** No spatial mixing. Cheap, answers
  "does the alignment exist at all", supports every control arm.
* **v2, only if v1 clears G0:** convolutional vision blocks with on-the-fly `feature_volumes`
  forwards. Our own CleanDIFT result is the reason to expect this matters — `paperhead` beat
  `student` purely on head capacity (0.7908 vs 0.7862) while having *lower* feature-matching cosine.

## 8.6 The deliverable (this closes R6, which has been open since Part 2)

Three endpoints, in priority order. E1 is new and is what 8.0.1 makes possible.

**E1 (primary) — voxel → chain assignment: "which of these sequences owns this density".**
Input: a map (no atomic model) + the set of distinct sequences present (median 9.5). Output: per
voxel, a distribution over those sequences. Metric: per-voxel top-1 accuracy and mean IoU against
the deposited model, macro-averaged over chains, cluster-bootstrapped.
Why this one: it is the direct analogue of dino.txt's open-vocabulary segmentation; it is *within
map*, so R1's size/composition shortcut has no channel; it aggregates over ~10²-10³ residues per
chain, so it is not blocked by the weak per-residue AA identity (K3/P14); and chain assignment in
large complexes is a genuine model-building bottleneck that neither ModelAngelo-style tracing nor
fold-then-simulate answers cheaply.
Baselines, all mandatory: (a) **volume prior** — assign every voxel to the longest chain (this is
the size shortcut in its purest form and it will not be small); (b) per-voxel AA classifier
(measured 0.2475, 20-way) aggregated to chains by composition matching; (c) random-weight vision
tower; (d) shuffled sequence↔map pairing.

**E2 (secondary) — map → sequence retrieval**, the zero-shot-classification analogue, scored by
late interaction (Part 5.4) with `1/|V|` normalisation. **Length-matched candidate pools only**, and
the `[N, R_g, AA composition]` baseline reported in the same table. Note the logged chance
correction: for length-matched pools chance is ~1/11, not 1/M.

**E3 (diagnostic) — voxel → residue-index assignment.** Expected weak (P14). Report as a ceiling
probe, not as a headline.

**The honest competition, stated up front.** At ≤3 Å, fold-then-fit (ESMFold2 ~9.4 s + rigid fit)
and ModelAngelo (trace → HMM → search, better than human experts) are strong training-free rivals.
The niche is **4-10 Å**, where tracing fails outright (ModelAngelo 49% top-1 at 4-5 Å → **0%** at
5-10 Å) — and that is precisely where CryoFM2 is most out of distribution. Hence P11.

## 8.7 Evaluation protocol

* Val-select every hyperparameter; **read test once**.
* **All arms in one process**, on identical maps/voxels/residues, asserted equal — the measured
  cross-process wobble is ~1.1 points from nothing but a fresh RNG subsample.
* Paired **cluster** bootstrap over the 100 test clusters via `probes/o5_stats.py`; report the
  realised MDE beside every difference. Reference: the CleanDIFT SS comparison achieved MDE 0.0087
  against a projected 0.0133, so the projection was conservative — quote realised, not projected.
* **Never gate on the training similarity or feature-matching cosine.** Measured to mispredict
  downstream twice, in opposite directions: `paperhead` had *lower* cosine and *higher* SS; `t700`
  had much *higher* cosine and *equal* SS. Selecting on cosine would have shipped the wrong model.
* Report the **between-map / mismatched floor and d′ always**, never top-1 alone: an *untrained*
  UNet reaches pooled top-1 0.58-0.68 against chance 0.10.
* Collapse monitor: **raw** off-diagonal cosine and `‖x−x̄‖/‖x̄‖`. **Not effective rank** — it read
  197.7 on a provably degenerate set, because it is computed on centred data.
* Report headline numbers with CryoFM2-pretrain-contaminated entries excluded (15.1% of
  Cryo2StructData); IDs are in `data/cryofm2_pretrain_lists/` (P12).

## 8.8 Gates, cheapest first

* **G0 (hours, cached features, 1 GPU) — does the dense alignment exist?** v1 pointwise heads, E1
  top-1 vs all four baselines of 8.6. **Stop if the real arm does not clearly beat both the volume
  prior and the AA-classifier baseline.** This is the whole plan's kill switch and it is cheap.
* **G1 (minutes, CPU) — trivial-covariate partialling** on any pooled/global term. If `λ_global > 0`
  wins only before partialling, it is the R1 shortcut and the global term is dropped.
* **G2 (~2 h GPU) — invariance cost at non-residue-centred voxels.** G2b's own stated limit: it
  measured Cα-centred boxes, and the voxel design samples arbitrary positions. Cheap; must precede
  committing to frame-free deployment.
* **G3 (~1 GPU-day) — v2 convolutional vision blocks** vs v1, and `λ_global` sweep.
* **G4 — the low-resolution arm** (low-pass the cached volumes to 4/6/8/10 Å). This is where the
  niche is; if E1 collapses under low-pass, the deliverable is confined to ≤3 Å where the
  training-free rivals win, and that must be written up as such.

## 8.9 Files to write

| file | role |
|---|---|
| `data/build_map_chains.py` | **the mandatory data build.** Per map, enumerate every polymer chain ≥20 res from the deposited PDB with its sequence and heavy-atom coords; dedupe to distinct sequences; record chain↔sequence multiplicity (homo-oligomer copies). Output `data/map_chains.csv` + per-map npz. Without this, 8.0.1's defect silently poisons every target. |
| `data/extract_esmc_expanded.py` | ESM-C **layer 32 only** for the ~11 k chains the expanded inventory needs (1147 maps × ~9.5 distinct seqs), fp16 → ~6 GB. The 36-layer stack would be ~228 GB and buys +0.017; keep the existing 36-layer cache for the 1500-chain ablation only. |
| `probes/voxel_sampler.py` | `p(v) ∝ K(d(v))` with `d` = distance to nearest heavy atom (isometry-invariant, so sampling cannot bias orientation); explicit background fraction; edge margin ≥ PATCH//2 from every face (P9); sparse in space, not dense from few maps (P6). |
| `probes/build_voxel_cache.py` | frozen student → `feature_volumes` → `sample_at`, K=8 SO(3) rotations, fp16, atomic + requeue-safe (mirror `build_vol_cache.py`). |
| `probes/dinotxt_model.py` | the two towers + heads + learnable τ. Self-test asserting zero-init identity and L2 normalisation. |
| `probes/dinotxt_loss.py` | soft-target CE both directions, background class, all-chain targets, homo-oligomer multi-positives. Unit-tested on a hand-built 3-residue / 2-chain case. |
| `probes/dinotxt_train.py` | cluster-batched trainer over the cache; collapse monitors; atomic snapshots. |
| `probes/dinotxt_eval.py` | E1/E2/E3 + **all** baselines in one process, cluster bootstrap, gate evaluated in code. |
| `slurm/dinotxt*.slurm` | launchers, positional `"$@"`, mirroring `slurm/cleandift.slurm`. |

Reuse unchanged: `cleandift_data.split_map_lists`, `o5_stats`, `cryofm_tap.{load_student,
feature_volumes, sample_at, cube_rotations, rotate_volume}`, `o5_boxes.load_norm_vol`.

## 8.10 Order of work

1. `data/build_map_chains.py` and **look at the output before trusting it** — this is the step that
   fixes 8.0.1, and every number downstream depends on it being right.
2. `probes/voxel_sampler.py` + a 5-map visual/statistical sanity check (background fraction, edge
   margin, distance histogram, chains-per-voxel).
3. Student whole-map path verification (P17): stitched `feature_volumes` on the student, timestep
   invariance assertion, and reproduce a known box-path number to prove the two paths agree.
4. `probes/build_voxel_cache.py`, K=1 first on 50 maps, then all 1147 × K=8.
5. `extract_esmc_expanded.py`.
6. Model + loss + unit tests; 100-step smoke; **G0**.
7. G1, G2. Then v2/G3 only if G0 passed.
8. G4 low-resolution arm.
9. Record against the gates in `GOALS.md`; append the outcome here.

## 8.11 Pitfalls

Each one is silent if wrong, and each is either measured in this project or measured today.

**P1 — Selecting on the training loss / feature cosine.** Measured to mispredict downstream twice,
in opposite directions (8.7). Gate on E1, val-selected, test read once.

**P2 — Coordinate leakage, the trap this project has fallen into hardest.** The pairwise-contact
probe reached AUC 0.997 purely because band features average spatial neighbourhoods, so contacting
residues shared neighbourhoods *by construction* — random embeddings would have "worked". Here the
soft target `w(v,i)` is built **from coordinates**. So: no coordinate-derived quantity may enter the
inference path or the metric. E1 must be map + sequences in, assignment out. Run the random-weight
vision tower explicitly to quantify what geometry alone achieves.

**P3 — Size shortcut in any pooled term.** `[N, R_g, comp]` = 0.629; pooled CryoFM = −0.022 after
partialling. Length-matched negatives; the volume-prior baseline in every E1 table.

**P4 — Missing floors.** Mandatory arms: shuffled pairing, random-weight vision, sequence-only,
geometry-only. And judge shuffled arms **on cosine, not R²** — a working shuffled control has
*negative* R² by construction, and an earlier version of exactly this check flagged a correct
control as leakage for that reason.

**P5 — Cross-process wobble ~1.1 pt.** All arms in one process, identical samples, asserted.

**P6 — Correlated samples inflate N.** Adjacent feature cells overlap receptive fields (4-5 Å
half-decay vs 3 Å/cell → roughly half-redundant); residues within a chain are correlated
(chain-batched gradients cost ~0.02 R² vs IID). Sample sparsely in space; cluster bootstrap only.

**P7 — False negatives are structural, not incidental.** Every α-helix resembles every other;
homologous residues are genuinely similar; neighbouring voxels overlap. Batch by cluster; the soft
target down-weights smoothly rather than asserting negatives. Fallback = Part 6.4.

**P8 — ★ Multi-chain targets, the biggest correctness risk.** Median **12 chains / 9.5 distinct
sequences per map** (8.0.1). Build the target over **all** chains; treat other-chain density as
neither positive nor background nor a same-map negative. For homo-oligomers, **all n symmetry copies
are positives** — otherwise the loss punishes a correct prediction. Symmetry has already broken four
separate things in this project (per-residue assignment, PCA frames, ×2 more); treat it as a
first-class variable.

**P9 — Padding is not vacuum.** 0 in preprocessed units is raw density 0.04, which is *above*
background (raw 0 → −0.44), so an out-of-bounds sample gets a slab of mean density. Apply the same
edge margin the box path uses.

**P10 — Student checkpoint round-trip.** Keys are `time_embedding.p`; the swap must precede
`load_state_dict`, and `strict=True` is deliberate. Use `load_student`, never hand-roll.

**P11 — Resolution regime.** Everything measured at ≤3 Å; the niche is 4-10 Å where CryoFM2 is
furthest OOD. Resolution augmentation must be an axis from the start (G4), not bolted on.

**P12 — Pretrain contamination.** 15.1% of Cryo2StructData is in CryoFM2's pretrain train split; the
32-entry pretrain test set is clean but tiny. Report with contaminated entries excluded.

**P13 — Emergence needs scale we do not have.** 466 clusters vs dino.txt's ~10⁸ pairs. Supervise the
dense term; do not expect the concat trick alone to deliver voxel-level alignment.

**P14 — The target's information content is low per residue.** AA identity from density is 0.2475
(prior 0.089), and the CleanDIFT student gained *nothing* on AA. So voxel↔single-residue is
ill-posed; E1's chain-level aggregation and Part 6.3's region↔k-mer are the well-posed forms. If E3
is weak, that is the expected outcome and not a bug.

**P15 — Collapse.** Monitor raw off-diagonal cosine and `‖x−x̄‖/‖x̄‖`. Effective rank does not
detect collapse (197.7 on a degenerate set). Contrastive with heavy sample correlation is exactly
the regime where collapse hides behind a falling loss.

**P16 — τ and λ do not transfer.** Recalibrate `λ_global` after any change to τ, normalisation, or
the kernel.

**P17 — The student's whole-map path has never been exercised.** Every CleanDIFT number was measured
on the **Cα-centred box** path; this stage needs **stitched whole-map** `feature_volumes`. The
student was trained on a 50/50 mixture whose second half *is* patch-grid crops with cube rotations
(D10), so it should be in distribution — but "should be" is not a measurement. Verify explicitly in
step 3, and assert the timestep argument is genuinely ignored.

**P18 — Do not re-derive a new split.** The student was trained against the map-level split of
`alignment_chains.csv`; any new split risks showing the encoder density that contains test residues,
which is the exposure D11 exists to prevent.

## 8.12 Compute budget

| step | cost |
|---|---|
| `build_map_chains.py` | ~1 h CPU (gemmi over 1147 PDBs) |
| ESM-C layer 32, ~11 k chains | ~1 h 1×H100, ~6 GB |
| voxel cache, 1147 maps × K=8 | ~6-10 h 1×H100, ~24 GB |
| v1 training run | **minutes** on cached features (this is the point of 8.5) |
| G0 + all baselines | ~1 h |
| v2 (on-the-fly forwards) | ~1-2 GPU-days |

Total to a G0 answer: **under a day of GPU**, dominated by caching, not training.

## 8.13 Out of scope for v1

Region↔k-mer granularity (Part 6.3 — pilot after G0, since it is the better-posed form and the one
that maps onto threading at inference); VICReg and relational fallbacks (Parts 6.4/6.5 — reach for
them only on the specific measured failure that motivates each); cryo-ET transfer (no paired data,
settled); training either foundation model; Phase 0d simulated-vs-experimental.

---

## 8.14 REVISIONS FROM IMPLEMENTATION (2026-08-31, same day)

Part 8 above is the plan as designed. This section records what changed once it
was built and measured, and **supersedes the earlier text where they conflict.**
Four revisions came out of discussion, five out of the code.

### From discussion

**R1 — Voxel sampling must be decoupled from the target.** §8.9 specified
`p(v) ~ K(distance to nearest heavy atom)`, which needs the atomic model — so
§8.6's "input: a map, no atomic model" was contradicted by §8.9's own sampler,
and training and inference would have drawn from different pools. Corrected:
*which voxels we look at* and *what the target says about them* are separate
decisions. See R6 for how this then failed on contact with data.

**R2 — Voxel-anchored is primary; the residue-anchored dual is a low-weight
auxiliary.** §8.4 and Part 6.2 called for a symmetric bidirectional loss. Four
reasons to demote the dual: (a) v2r's inference shape *is* E1; (b) v2r normalises
over residues, a set the data gives, while r2v normalises over sampled voxels, a
set we chose — so r2v's loss scale is a nuisance of our own sampler; (c) with
residues indexed by (sequence, position), **v2r is single-valued under
homo-oligomer symmetry** while r2v is irreducibly n-valued; (d) background only
exists voxel-anchored. Part 6.2 reached for the dual to fix volume-weighting,
which conflates loss direction with sample weighting — the latter is the
sampler's job (`chain_quota`). Default `lambda_dual = 0.3`.

**R3 — Residues are indexed by (sequence, position), not (chain, position).**
Makes v2r single-valued under symmetry and is the only indexing the sequence
tower can consume.

**R4 — Contamination is measured, not estimated.** Overlap of the alignment set
with CryoFM2's pretrain train list: 14.7% of train maps, 18.7% of val, **12.4% of
test** (28/225 maps, 29/234 chains). At cluster level, **21 of 100 test clusters
are touched and 79 are entirely unseen**, so the unseen-only robustness split
costs ~12% wider CI (MDE 0.0087 -> ~0.0098) and stays resolvable. Weaker than the
raw number suggests: CryoFM2 trained on *half-maps* (we forward the full
deposited map) and **never saw a sequence**, so contamination can sharpen
features but cannot leak the label.

**R5 — The corpus cap is arbitrary and the sample size is not fixed.**
`build_alignment_set.py` caps at `--max-chains 2000` over a seed-0 shuffle;
**7,361 Cryo2StructData entries are available and 1,147 are used**. The
466-cluster limit that drives P13 is a budget decision, not a data limit. The same
cap explains the chains-per-map gap: the script dedupes to one row per (entry,
sequence) and takes the first 2,000 of ~70k candidates, so each map contributed
~1.3 randomly chosen chains.

### From building it

**R6 — ★ THE MODEL-FREE VOXEL CRITERION DOES NOT WORK, and this is the one that
matters.** Fraction of selected voxels actually within 5 Å of a heavy atom:

| criterion | 25414 | 0322 | 11263 | 31059 | 4873 | 27276 | 33528 | 10760 |
|---|---|---|---|---|---|---|---|---|
| raw p90 | 0.16 | 0.22 | 0.27 | 0.81 | 0.16 | 0.26 | 0.29 | 0.17 |
| blur(3σ) p99 | 0.78 | 0.05 | 0.20 | 0.99 | 1.00 | 0.51 | 0.86 | 0.31 |

Not a coordinate bug — **atoms sit at the 98–99.9th percentile of density in 7 of
8 maps**. Three causes: (a) true protein occupancy is only **2.3–17%** (median
~4%), so p90 is ~95% solvent before anything else goes wrong; (b) some maps hold
artifacts *brighter* than protein (EMD-0322: box p99 = 2.86 vs median atom
density 1.61), which is why blurring makes it **worse**; (c) EMD-11263's atoms sit
at the **61st** percentile — the model is barely in density, a per-map QC flag.
This is the same failure already logged for canonicalisation ("`vol > 0`
thresholding is indefensible on experimental maps"), now quantified for sampling.

**Consequence:** the default pool is **model-defined** (foreground within 4 Å of a
heavy atom, background beyond 12 Å, the gap a deliberate dead zone). The atomic
model is a labelling instrument here exactly as it is for the target, so this
costs nothing the target did not already cost, and G0 — which asks whether the
alignment exists at all — does not depend on it. **But "no atomic model at
inference" is NOT established, and it is now the explicit blocker on deployment.**
`pool="density"` is kept for that experiment. Measured on real maps, the model
pool puts foreground at a median 1.7–1.9 Å from an atom and background at 32–104 Å.

**R7 — `feature_volumes` ignored `stop_after` and raised.** Constructing the tap
with `stop_after=` and calling the whole-map path threw `StopForward` straight out
of the method: the two entry points disagreed and the whole-map path was unusable
with the truncation that makes it affordable. Fixed in `teachers/cryofm_tap.py`,
and **verified bitwise-equal to the full forward on GPU** before anything was
built on it (check 0 of `o6_volpath_verify`).

**R8 — Rotations in the cache are the OCTAHEDRAL group, not generic SO(3).**
`rotate_volume` implements only cube rotations (it takes `argmax|R[a]|` per row),
so a generic matrix silently snaps to the nearest axis permutation while
`rotate_coords` applies the true `R` — volume and coordinates then disagree. An
assertion now rejects any non-signed-permutation matrix. **The cache therefore
cannot measure generic-SO(3) invariance** (the octahedral group is CryoFM2's own
`RotCube24` pretraining augmentation, i.e. the easy case). For *augmentation* the
distinction looks unimportant (G2b: cube −0.006 SS vs SO(3) −0.005), but the two
claims are different and only the first is supported here.

**R9 — Small calibrations.** `n_max = 8` target residues per voxel truncates only
**0.6–0.8%** of the mass (checked against `n_max=32`), so it stays. Background's
unnormalised weight must be **well below 1** — at 1.0 it ties exactly with a
residue at zero distance, making a voxel sitting on an atom 50% background;
`bg_weight=0.1` puts the crossover at ~8.6 Å for σ=4. Poisson-disk thinning needs
a 27-neighbour check, not one-point-per-cell (which gave measured separation 1.00
for a requested 3.0), and the two class pools must be thinned against each other.

### Data as built

| artifact | measured |
|---|---|
| `data/map_chains.csv` + `data/map_chains/` | **17,498 chains / 1,147 maps**, median 9/map (max 56), median **7 distinct sequences**/map, 93% have ≥2. 280 MB, 1.6 min CPU. |
| chains already carrying ESM-C | **1,500 (8.6%)** — the other 91.4% were unlabelled in every prior target |
| maps spanning >1 chain-level split | **110/1,147**, resolved test>val>train (matches `split_map_lists`' `clean_train`; a first version wrongly asserted one split per map and died on EMD-0322) |
| `data/esmc_seq32/` | **7,186 distinct sequences** (homo-oligomer dedup), layer 32, fp16, **3.86 GiB, 4.7 min** on one H100 |

### Modules written

`data/build_map_chains.py`, `data/extract_esmc_expanded.py`,
`probes/voxel_sampler.py`, `probes/build_voxel_cache.py`,
`probes/dinotxt_data.py`, `probes/dinotxt_model.py`, `probes/dinotxt_loss.py`,
`probes/dinotxt_train.py`, `probes/dinotxt_eval.py`,
`probes/o6_volpath_verify.py`, `slurm/esmc.slurm`. The three with non-trivial
logic carry `--self-test`-style checks that run in seconds
(`voxel_sampler`, `dinotxt_model`, `dinotxt_loss`); all pass.

### 8.15 REFRAMING (2026-09-01) — the dino.txt analogy is thinner than Part 8 assumed

From discussion, and it changes a priority rather than a mechanism.

**Why dino.txt mixes tokens: it has no dense labels.** Image-text pairs supply
supervision only at the whole-input level -- there is no ground-truth
word-to-patch correspondence -- so the two learnable blocks and the
`g = [CLS ; mean-patch]` concat exist to manufacture an alignable object, with
dense capability then *emerging*. The paper's own motivation for the concat is
that plain LiT "leads to unsatisfactory results on dense tasks."

**We are in the opposite regime.** The fitted model gives EXACT residue<->voxel
correspondence, and what we lack is paired STRUCTURES:

| | dino.txt | here |
|---|---|---|
| paired samples | ~10^8 | **505 train clusters** (798 maps) |
| dense labels | none | exact, from the deposited model |
| consequence | must mix tokens, align globally, hope dense emerges | supervise dense directly; no mixing needed to *create* the signal |

So the pointwise v1 head is not a compromise forced by caching -- it is the
structure the supervision affords. Token mixing would only add spatial context,
and context is already available more cheaply and with calibration along the TAP
axis (`up_blocks[1]` 4-5 A, `up_blocks[0]` 9-10 A, `mid_block` 15 A, plus concat),
which job 3492533 measured.

**REVISION to §8.5 and §8.11: v2 is a capacity RISK, not the expected upgrade.**
Effective sample size is ~9.6 M voxels but only **505 independent clusters**, i.e.
**~19,000 heavily-correlated voxels per cluster**. A token-mixing head has many
more ways to key on map-level idiosyncrasy (specimen, instrument, reconstruction,
sharpening) than on the residue<->density relation, and the training loss would
look healthy throughout. v2 therefore stays gated on G0 AND must be run as a
capacity test with **overfitting as the pre-registered expected failure mode**,
monitored by the train-vs-val gap computed PER CLUSTER. The earlier appeal to
`paperhead > student` (+0.005 SS on head architecture alone) does not transfer:
that head sat on 279k training residues for a per-residue task, not on 505
clusters for a cross-structure one.

**NEW HIGHEST-VALUE LEVER: expand the corpus, not the head.** The binding
dimension is clusters and it was capped arbitrarily (R5: 7,361 entries available,
1,147 used, `--max-chains 2000`). The full corpus takes ~505 -> ~3,000 train
clusters -- 6x in exactly the constrained direction -- for a rerun of
`build_alignment_set.py` plus the caches, with no new design. This outranks every
architecture change in Part 8 and should be done before v2 is considered.

**Honest relabelling.** What is being built is **LiT-style two-tower alignment
with dense supervision** (frozen strong encoder per modality, light aligner,
late-interaction scoring), in the ProteinCLIP/LiT lineage. dino.txt's actual
contribution -- emergent dense alignment from a global loss -- is the part we do
not need. Keeping the "dino.txt" label invites copying machinery that solves a
problem we do not have; §8.1's D1/D2 already deviated from it on both counts, and
this is why.

### 8.16 P17 AND THE TAP SWEEP — both resolved (2026-09-01)

**★ P17 PASSES, and more strongly than the box path.** `o6_volpath_verify`, 149
chains / 2,973 residues / 20 test clusters, 68 min. Linear SS probe on
volume-path (stitched whole-map, lab-frame) features:

| arm | SS | AA (prior 0.101) |
|---|---|---|
| `vol_student` | **0.6854** | 0.0868 |
| `vol_teacher` (t=750 decoupled) | 0.5479 | 0.0759 |
| `box_student` (published config) | 0.5118 | 0.0651 |
| prior | 0.4123 | 0.1013 |

`student - teacher` on the volume path = **+0.1374**, CI [+0.0759, +0.1889],
MDE 0.0572 -> significant, and ~9x the +1.5 points the same student showed on the
box path. So the student is not merely usable on the stitched path, it is BETTER
SUITED to it than the teacher is. Plausible mechanism: the teacher is run
decoupled (clean input at t=750), which is off-manifold, and the whole-map path is
where that hurts most; the student was distilled precisely to consume clean input.
Checks 0 and 1 also passed (truncation bitwise-equal; student timestep-invariant,
teacher differs by 249.9).

**Caveats, and they are not small.** 2,240 training residues against the published
run's 279k, and 20 test clusters against 100 -- so absolute values are NOT
comparable to the published table and only the within-run contrast is meaningful.
**AA is uninformative here**: every arm scores BELOW its own prior, which is what
a 256-d 20-class logistic regression does on 2,240 samples. Do not read the AA row.

**CHECK 2 WAS MIS-DESIGNED — my error, and the number is not evidence of
anything.** I predicted box-vs-volume centred cosine >0.9 on the grounds that both
crop 64^3 and the network is convolutional. Measured: median **0.3827** (floor
-0.041). The prediction ignored that `chain_boxes` cuts in the residue's
**backbone frame**, so the two paths differ by an arbitrary ROTATION, not a
translation -- and `up_blocks[1]`'s global pose cosine is ~0.29. The measurement
is therefore consistent with this project's existing pose numbers and re-measures
rotation sensitivity rather than testing stitching registration. A valid version
would compare LAB-frame boxes (`frk = I`) against the volume path. Stitching is
separately guaranteed by the alignment assertion inside `feature_volumes` and by
check 0.

**★ TAP SWEEP: `up_blocks[1]` confirmed, decisively.** `o5_arms_multitap`
(job 3492533, 1,484 chains, 100 test clusters, all taps in one process),
`student_paperhead`:

| tap | dim | SS | AA |
|---|---|---|---|
| **`up_blocks[1]`** (3 A/token) | 256 | **0.7908** | **0.2181** |
| `concat` (all three) | 1280 | 0.7874 | 0.2024 |
| `concat_pca` (dim-matched) | 256 | 0.7753 | 0.1708 |
| `up_blocks[0]` (6 A) | 512 | 0.7152 | 0.1361 |
| `mid_block` (12 A) | 512 | 0.6760 | 0.1166 |
| *raw 8^3 voxels* | 512 | *0.7475* | *0.2735* |

Three readings:
1. **The Part 8 tap choice was right and the voxel cache does not need rebuilding**
   -- which was the live risk while this job ran.
2. **Multi-scale concatenation buys nothing.** `concat` carries 5x the dimensions
   and still loses to a single tap (0.7874 vs 0.7908); dimension-matched it is
   clearly worse (0.7753). This closes the "should the head see several scales"
   question raised in the pooling discussion: no.
3. **The script's own hypothesis is refuted.** Its docstring called `up_blocks[0]`
   "a live candidate to beat the published headline" on the strength of its pose
   invariance (0.826 vs 0.706). It is 7.6 SS points WORSE. Pose stability again
   fails to predict task usefulness -- the third time in this project.
4. **The CleanDIFT gain is TAP-SPECIFIC.** At `mid_block` and `up_blocks[0]` the
   null control `student_ctrl` (0.6797 / 0.7254) BEATS every distilled arm. The
   distillation effect exists only at `up_blocks[1]`. Any future claim about
   CleanDIFT here must name the tap.

## 8.18 ABLATION PROGRAMME — two-tower ESM-C/CryoFM alignment (2026-09-02)

Renamed at the user's suggestion: **two-tower ESM-C/CryoFM alignment (with dense
supervision)**, not "dino.txt". Accurate on architecture (two frozen encoders +
trained projection heads + shared space, LiT/ProteinCLIP lineage); the dense
per-voxel supervision is what distinguishes it from CLIP-style two-tower work.
The `dinotxt_*` filenames are legacy.

### 8.18.0 TWO MEASURED CONSTRAINTS THAT SET THE PROTOCOL

**(1) 12k steps is the converged budget; 4k is NOT.** val E1 0.2368 (800 steps)
-> 0.3218 (4k) -> **0.3485 (12k, plateaued: 0.3478/0.3481/0.3486/0.3485 over the
last four evals)**. Running a CAPACITY ablation at a budget where the baseline has
not converged systematically favours the SMALL arm, because larger models need
more steps to reach the same point. This session already inverted one conclusion
by reading an unconverged run (every arm "below the volume prior" at 800 steps;
all arms above it at 4k). **Every capacity arm runs at >=12k steps.** Confirm the
plateau per arm rather than assuming it transfers.

**(2) The sequence-side pair adapter has a QUADRATIC MEMORY WALL.** Pair rep is
[n, n, d_pair] per chain and the per-chain loop holds one per chain until
backward, so activation memory is `O(sum_c n_c^2)`, NOT `O(n_res)`. All three pair
arms died at **78 GB allocated on an 80 GB H100** (~72 chains/batch at
batch_maps=8). Fixed with per-chain gradient checkpointing (peak -> one chain's
worth, ~1.3x compute). **Consequence for this programme: sequence-side capacity
does not scale like vision-side capacity.** Any pair arm must report peak memory,
and `d_pair`/`n_tri`/`max_len` are memory knobs before they are capacity knobs.

### 8.18.1 COST TIERS — tier by whether a NEW VOXEL CACHE is needed

| tier | cost per arm | what it covers |
|---|---|---|
| **T0** | **~105 min GPU** (12k steps on the existing cache) | head architecture + capacity, objective hyperparameters, token mixing over sampled voxels, load-bearing swaps |
| **T1** | +7 min CPU/GPU extract, then T0 | ESM-C layer / layer-mix (re-extract is cheap: 36 layers took 7 min) |
| **T2** | **~14-16 h GPU cache build**, then T0 | tap, timestep, coupled/decoupled pairing, low-pass (G4), random weights |

T0 is effectively free at this scale (a 12-arm sweep is ~21 GPU-hours, runnable
in parallel). **T2 is 10x the cost of everything else and must be gated on a T0
result that motivates it.** Do not re-cache on a hunch.

### 8.18.2 T0-A — VISION ADAPTER CAPACITY (the primary ask)

Current: `depth=1`, `hidden=512`, `dim=256`, 0.89 M params, strictly per-token.

| axis | values | current | hypothesis |
|---|---|---|---|
| `depth` | 1, 2, 4 | 1 | low yield alone; depth without mixing only re-mixes channels |
| `hidden` | 512, 1024, 2048 | 512 | low-moderate |
| `dim` (shared space) | 128, 256, 512 | 256 | 256 may already bottleneck 9-way discrimination; cheap to test |

**Prior that tempers expectations:** the ESM-C -> density-feature map measured
essentially LINEAR (MLP over ridge: **+0.018**). That was a REGRESSION target, and
E1 is discriminative, so it does not transfer directly -- but it is the best
available prior and it points to modest returns from pure width/depth. Run the
axes because they are cheap, not because they are promising.

### 8.18.3 T0-B — TOKEN MIXING (the highest-value T0 arm)

This is the one real architectural gap vs dino.txt, and **it can be tested WITHOUT
a new cache**: attend over the ~384 SAMPLED voxels of a map rather than over a
dense volume. Self-attention on 384 tokens is trivially cheap. This is the plan's
"v2 conv blocks" idea at ~1% of the cost, and it does not need on-the-fly
`feature_volumes`.

Motivation is mechanistic, not analogical: `up_blocks[1]` has a **4-5 A
half-decay**, i.e. sub-neighbourhood, while a residue's structural environment is
the **~10 A contact scale** -- which is why `up_blocks[0]` (9-10 A) won the
per-residue regression. Mixing over sampled voxels lets the head rebuild
neighbourhood context from a fine tap instead of choosing a coarser one.

| arm | what |
|---|---|
| `mix=none` | baseline (current) |
| `mix=attn` | 1-2 self-attention blocks over the map's sampled voxels |
| `mix=attn+relpos` | plus a relative-position bias from voxel coordinates |
| **`mix=attn, features ABLATED`** | **MANDATORY CONTROL** |

**The mandatory control, and why.** Voxels of the same chain are spatially
clustered, so attention with position information can raise E1 by pure spatial
smoothing with no reference to feature content. Replace features with a constant
or noise, keep the mixing and the positions: whatever that arm scores is the
smoothing floor, and only the margin above it is alignment. This is the same
failure mode as the coordinate-leakage artifact that inflated the sibling
project's contact probe to AUC 0.997 -- there, band values averaged a spatial
neighbourhood, so contacting residues agreed BY CONSTRUCTION. Do not report a
mixing gain without this arm.

### 8.18.4 T0-C — SEQUENCE ADAPTER

| arm | params | status |
|---|---|---|
| `pair=none` | 0.89 M | baseline, 0.3485 @ 12k |
| `pair=relpos` | 0.95 M | **mandatory control** -- on the old target it beat every ESM-pair arm |
| `pair=esm` | 1.11 M | running (3509276) |
| `pair=esm+relpos` | 1.11 M | running (3509277) |
| `depth`/`hidden` on the sequence head | -- | same axes as T0-A |

`relpos` is the control that decides interpretation: a gain the relative-position
encoding reproduces on its own is not a pair-channel gain. Sweep `d_pair`
{16, 32, 64} and `n_tri` {1, 2, 4} only if an ESM-pair arm beats `relpos`, and
report peak memory with each.

### 8.18.5 T0-D — OBJECTIVE

| axis | values | current | note |
|---|---|---|---|
| `n_cross` | 0 (full), 200, 500, 2000 | 500 | own-map share 12.5% / ~88% / ~79% / ~52%; measured monotone so far |
| `lambda_dual` | 0, 0.3, 1.0 | 0.3 | is the r2v dual earning its place at all? |
| `sigma` (soft target) | 2, 4, 8 A | 4 | target sharpness; needs a re-derived cache field or on-the-fly recompute |
| `tau` | learnable, fixed 0.07 | learnable | it drifted 0.07 -> 0.035, so it is doing something |
| `chain_quota` | on, off | on | class balancing may be HURTING E1, which is voxel-weighted |

`chain_quota` off is a genuinely open question worth an early arm: the quota
equalises chains but E1 is scored per voxel, so the objective and the metric
disagree about weighting by construction.

### 8.18.6 T0-E — LOAD-BEARING SWAPS (most informative per GPU-hour)

These bracket the result and are the arms most likely to change the write-up.

| arm | replaces | question |
|---|---|---|
| `seq=seqwin3` | ESM-C -> +-3 one-hot (140-d) | how much of E1 needs a PLM at all? (on the old target: 0.036 vs 0.187) |
| `seq=composition` | ESM-C -> per-chain AA composition | is E1 just composition matching in disguise? (pairs with baseline (b)) |
| `vox=raw_density` | CleanDIFT features -> raw density patch | how much needs CryoFM vs the density itself? |
| `vox=random` | CleanDIFT -> untrained network | **baseline (c)**, cache building (3504892-94) |

`vox=raw_density` is the cheapest strong test of whether the whole CryoFM stack is
load-bearing, and it is a T0 arm only if the cache retains a raw-density patch
per voxel; otherwise it is T2.

### 8.18.7 T2 — GATED ON A T0 MOTIVE

| arm | cost | gate to run it |
|---|---|---|
| tap `up_blocks[0]` / `mid_block` / concat | ~15 h each | run ONLY if T0-B token mixing helps, since a mixing gain implies the tap's receptive field is the limiter -- that is the same hypothesis tested more cheaply |
| timestep / coupled pairing | ~15 h | the coupled-decoupled top1 gap was +0.071 at t=750; relevant, but this is CleanDIFT-student territory and the student is timestep-invariant by construction |
| **low-pass 4/6/8/10 A (G4)** | ~15 h | **run regardless -- this is the niche.** If E1 collapses under low-pass, the deliverable is confined to <=3 A where training-free rivals win, and that must be written up as such |

### 8.18.8 PROTOCOL, non-negotiable

1. **>=12k steps, plateau confirmed per arm** (8.18.0).
2. **Selection on val E1; the gate on test, once.** Never select on the loss --
   this project twice measured the training objective mispredicting downstream
   (`student_paperhead` lower cosine but higher SS; `student_t700` much higher
   cosine, equal SS).
3. **Collapse guard on every arm**: raw off-diagonal cosine + `||x-xbar||/||xbar||`.
   NOT effective rank (read 197.7 on a provably degenerate set).
4. **Same seed across arms**, and >=2 seeds before believing a margin under ~1.5
   points: measured run-to-run wobble is ~1.1 points from nothing but a fresh RNG.
5. **Paired cluster bootstrap** over test clusters, never a per-voxel SE: adjacent
   feature cells are ~half-redundant at 4-5 A half-decay.
6. **Report against the volume prior AND baseline (b)**, not against v1 alone. An
   arm that beats v1 while losing to an AA classifier is not progress.
7. **Report peak GPU memory** for any pair or mixing arm (8.18.0 constraint 2).
8. Headline numbers with the 15.1% CryoFM2-pretrain-contaminated entries excluded
   (IDs in `data/cryofm2_pretrain_lists/`).

### 8.18.9 RECOMMENDED ORDER

1. **Finish G0 properly** -- baseline (b) AA classifier + (c) random vision. The
   gate is UNDECIDED until (b) lands; capacity work on an undecided gate is
   premature.
2. **T0-E load-bearing swaps** (4 arms). Cheapest route to knowing whether the
   thing is even about CryoFM and ESM-C.
3. **T0-B token mixing with its ablated-feature control** (4 arms). Highest
   expected yield and it substitutes for a ~15 h tap re-cache.
4. **T0-D objective** (`chain_quota` off, `lambda_dual=0`, `n_cross` ladder).
5. **T0-A/C capacity ladders** (6-8 arms). Cheap, expected modest.
6. **T2 low-pass (G4)** regardless of the above -- it defines the niche.

Deliberately NOT in this programme: unfreezing either encoder (505 independent
clusters; the frozen-encoder + small-head regime is the one the sample size
supports), and any arm selected on a feature-similarity proxy (P1).

## 8.19 G0 PASSES, plus E2 and the contamination split (2026-09-02)

### G0 — the kill switch is cleared, on the FULL stated stop condition
`probes/dinotxt_eval.py`, converged 12k checkpoint (`v2_nc500_long`, step 11500), 225 test maps /
179,354 voxels, all arms in one process on identical voxels (asserted), paired cluster bootstrap.

| arm | E1 top-1 | trained − arm | 95% CI | MDE |
|---|---|---|---|---|
| **trained** | **0.3935** | — | — | — |
| volume_prior | 0.3114 | **+0.0822** | [+0.0602, +0.1039] | 0.0219 |
| aa_classifier | 0.2500 | +0.1435 | [+0.1271, +0.1605] | 0.0169 |
| aa_classifier_hard | 0.2029 | +0.1906 | [+0.1735, +0.2083] | 0.0170 |
| shuffled | 0.2356 | +0.1580 | [+0.1371, +0.1804] | 0.0218 |
| untrained_head | 0.2063 | +0.1872 | [+0.1676, +0.2058] | 0.0190 |

- The stop condition names **two** arms — volume prior AND AA classifier — and both are now present.
  An earlier claimed pass was on a subset and was retracted; `dinotxt_eval` now prints
  `G0 UNDECIDED` without `--aa-ckpt` so it cannot recur.
- **A gating bug of my own nearly hid this**: `res["vs"]` was built from a hardcoded three-baseline
  tuple while the gate read `must`, which includes `aa_classifier` → `KeyError` *after* every
  accuracy had been computed. Fixed to bootstrap every arm present.
- The AA baseline is a fair, not a token, rival: its own internal accuracy is 20-way AA top-1
  **0.1869** (chance 0.05). Do not quote that as an E1 number — it is a different label space. Its
  E1 score, via composition matching, is 0.2500.
- **Baseline (c), random-weight vision, is still outstanding** (head training, job 3515051). The
  gate as stated does not require it, but P4 does: an untrained UNet reached pooled top-1 0.58-0.68
  against chance 0.10 in this project.

### CONTAMINATION — the headline is NOT inflated by CryoFM2 pretrain overlap
`probes/dinotxt_contam.py`. 28/225 test maps (12.4%) are in the CryoFM2 pretrain TRAIN list.

| subset | maps | trained | volume_prior | diff | 95% CI |
|---|---|---|---|---|---|
| seen | 28 | 0.4777 | 0.3917 | +0.0859 | [−0.0033, +0.1676] ns |
| **unseen** | **197** | **0.3803** | **0.2989** | **+0.0814** | **[+0.0584, +0.1052] SIG** |

- **The MARGIN is unchanged** (+0.0814 unseen vs +0.0859 seen); what differs is difficulty — the
  seen maps have a higher prior (0.3917 vs 0.2989), i.e. more lopsided chain-size distributions.
  So the overlap makes the maps easier for *every* arm, not the model better on its training data.
- The seen row is underpowered (28 maps, MDE 0.0845) and is descriptive only; the two subsets are
  different populations, so a difference between them confounds contamination with everything else.
- **Quote 0.3803 vs 0.2989 externally.**

### E2 — map → sequence retrieval works, and its control caught a rigged first run
`probes/dinotxt_e2.py`. Late interaction, `1/|V|` normalised, nearest-K length-matched pools,
cluster-disjoint distractors, 225 test maps, pool 11.

| arm | top-1 |
|---|---|
| **retrieval** | **0.6400** |
| aa_composition | 0.1733 |
| length_only | 0.0978 |
| chance | 0.0909 |

- **The first run was VOID and the control said so: `length_only` read exactly 1.0000.** Distractors
  were drawn randomly from a ±0.25 log-length band on *observed residue count*, while the control
  ranked by *full sequence length*, where the answer sits at distance exactly 0. Length was a
  sufficient statistic. Fixed by (a) selecting the K NEAREST in length instead of sampling a loose
  band, and (b) scoring the control against a MAP-SIDE size proxy (total observed residues), never
  against the answer.
- Realised pool length spread `|log(Lmax/Lmin)|`: median **0.015**, p90 0.155. `length_only` 0.0978
  ≈ chance 0.0909 confirms the matching.
- **Retrieval barely moved across the fix (0.6323 → 0.6400)**, so it was never leaning on the loose
  band — but that could only be known *after* the control worked, not before.
- This is the same failure mode as the coordinate leakage that inflated the sibling project's
  contact probe to AUC 0.997, and it is the third time in this project a control has voided a run.
  **The controls are earning their cost; keep writing them first.**

### G4 tooling — `--lowpass` added (the niche test)
`lowpass()` in `probes/build_voxel_cache.py`: soft raised-cosine Fourier filter, then **CryoFM
re-normalisation**. Re-normalisation is not optional — filtering drops the 99.999th percentile, so
an unrenormalised volume is systematically *dimmer* and the arm would measure contrast loss as
resolution loss. Verified: HF power fraction 0.93 → 0.000000, and the filter commutes exactly with
octahedral rotations (spherically symmetric), so filtering once before the rotation loop is correct.
8 Å cache building (job 3515217, ~15 h) → `data/voxel_cache_lp8`.

### Sequence-side pair adapter — a clean negative, T0-C largely retired
12k steps, same protocol. Best val E1: `relpos` 0.3463, `esm` 0.3469, `esm+relpos` 0.3454, against
the pointwise model's **0.3486**. The pair channel does not beat no pair channel, and the ESM-fed
arm is indistinguishable from its `relpos` control (+0.0006 against ~1.1 points of measured RNG
wobble) — so what little the channel does is positional, not sequence content. Given its
O(Σ n_c²) memory wall this is a good arm to lose. Do not sweep `d_pair`/`n_tri`: §8.18.4's stated
precondition (an ESM-pair arm beating `relpos`) is not met.

## 8.20 G0 COMPLETE and G4 AT 8 Å — the niche is real (2026-09-04)

### G0 with all four baselines, including the P4 random-vision floor
`results/dinotxt_g0_full.json`, 225 test maps / 179,354 voxels, identical voxels across arms.

| arm | E1 | trained − arm | 95% CI |
|---|---|---|---|
| **trained** | **0.3938** | — | — |
| volume_prior | 0.3115 | +0.0822 | [+0.0603, +0.1045] |
| aa_classifier | 0.2504 | +0.1434 | [+0.1264, +0.1604] |
| shuffled | 0.2352 | +0.1586 | [+0.1380, +0.1807] |
| untrained_head | 0.2125 | +0.1813 | [+0.1616, +0.2012] |
| **random_vision** | **0.1971** | **+0.1967** | [+0.1791, +0.2143] |

**The random-vision floor closes the biggest Phase 0 risk.** A head TRAINED FROM SCRATCH on
random-weight features (not the real head fed random input — that measures distribution shift and
was fixed earlier) reaches only 0.1971, barely over chance ~0.18 and *below* the real head at init.
CryoFM2's pretrained weights carry +0.1967 of the result, so this is not architecture-plus-geometry
masquerading as learned features. Cf. P4, where an untrained UNet reached pooled top-1 0.58-0.68
against chance 0.10 — that hazard does not materialise on E1.

### G4 — E1 IS INSENSITIVE TO AN 8 Å LOW-PASS
`results/dinotxt_g4_lp8.json`. Independent 12k head trained on `data/voxel_cache_lp8`, and its own
AA baseline retrained on the low-pass cache (reusing the 1.5 Å classifier would have measured
distribution shift, the same error corrected for `random_vision`).

| | 1.5 Å | **8 Å low-pass** |
|---|---|---|
| val E1 | 0.3486 | **0.3449** |
| test E1 | 0.3938 | **0.3947** |
| volume_prior | 0.3115 | 0.3117 |
| **trained − prior** | **+0.0822** | **+0.0830** [+0.0608, +0.1045] |
| aa_classifier (E1) | 0.2504 | 0.2515 |
| aa_classifier internal 20-way AA top-1 | 0.1869 | **0.1660** |

- **No degradation at all** — the margin over the prior is +0.0830 vs +0.0822, well inside one MDE.
- **This lands the method inside the stated niche.** §8.6 named 4-10 Å because tracing fails there
  (ModelAngelo per-residue: 49% top-1 at 4-5 Å → **0%** at 5-10 Å). E1 is unaffected at 8 Å.
- Note the internal AA accuracy DOES degrade (0.1869 → 0.1660) while its E1 score does not
  (0.2504 → 0.2515): composition matching aggregates over ~10²-10³ residues, so per-residue AA
  fidelity is not the binding constraint for either arm. Consistent with §8.6's rationale for
  choosing a chain-level endpoint over a residue-level one.

**VERIFIED NOT A NO-OP, because "no degradation" is exactly the shape of a test whose outcome is
fixed by construction** (the lesson from the first local-frame run, which returned 1.000 on every
metric because it could not have returned anything else). Between the two caches, per-voxel
**centred** cosine is 0.40-0.75 (raw 0.15-0.66) and the mean feature-vector norm roughly DOUBLES
(e.g. 100.6 → 204.2), so the filter substantially changes what the model sees. `res_idx` is
byte-identical, so voxel geometry is not a confound. The filter itself was unit-tested: HF power
fraction 0.93 → 0.000000, exact commutation with the octahedral rotation group.

**TWO CAVEATS, both material.**
1. **A low-passed 3 Å map is not an 8 Å map.** Real low-resolution data also has worse SNR,
   different reconstruction artifacts and heavier model bias. This isolates resolution — the right
   first experiment — but it is an optimistic simulation of the niche, not the deployment condition.
2. **"No degradation" has a less flattering reading.** Either chain assignment genuinely needs only
   mesoscale information (the niche), or E1 never used fine detail, in which case the 3 Å tap is
   wasted and the task is coarser than assumed. Both predict this result; they differ in WHERE the
   curve breaks. Resolution ladder at **4 / 12 / 20 Å** building (jobs 3534545/46/47) to find out.
   If E1 is still flat at 20 Å, E1 is a coarse-shape task and that must be stated plainly.

## 8.21 T0-B TOKEN MIXING — implemented without the 19.5 h re-cache (2026-09-04)

### Coordinates were recoverable on CPU; no GPU rebuild was needed
Mixing needs each voxel's position and the cache stores none. A rebuild is ~19.5 h of H100.
`probes/recover_voxel_coords.py` gets them in **30 min of CPU across 8 shards** instead, because
`sample_voxels(pool="model")` is a pure function of (volume SHAPE, atom coordinates, seeded RNG) —
**it never reads the volume's VALUES.** That is also the explanation for something already observed
but not understood: every low-pass cache came out with byte-identical `res_idx`. The voxel geometry
does not depend on the density at all, so ONE coordinate set serves every resolution.

**Verified rather than assumed**, since silently attaching wrong positions to features would read as
a modelling result: recovered coordinates are pushed back through `soft_targets` and must reproduce
the cached `res_idx` EXACTLY, `weight` to **0.0** absolute error, and the rotation labels. All
**1,147/1,147 maps pass, 0 skipped**.

Coordinates are stored in the **ROTATED** frame, matching the features. A canonical frame would make
any spatial module rotation-invariant by construction and would not deploy — an unknown map arrives
in an arbitrary frame. The cost is that the four rotations of a map are in different frames, so a
mixing model must draw each batch from ONE rotation (`--rot-per-batch`, forced on by `--mix-depth`).

### `VoxMix` — distance-biased attention over one map's voxels
Attention logits get a learned per-head RBF function of the **pair distance**; absolute coordinates
never enter, so the module is exactly translation- and rotation-equivariant. Zero-init `out_proj`
and FFN, so step 0 *is* the pointwise baseline and mixing must earn its way in.

Four properties are asserted in `_self_test`, not argued:
1. zero at init (output bit-identical to pointwise);
2. able to move once `out_proj` is non-zero (not a dead branch);
3. **no cross-map attention** — perturbing map 1's voxels leaves map 0's outputs bit-identical, so
   batch composition cannot leak into predictions;
4. **invariance to a rigid motion** of the coordinates (random SO(3) + translation).

### The control is post-hoc smoothing, and it is mandatory
Neighbouring voxels usually share a chain, so averaging over a neighbourhood raises E1 by making
predictions locally consistent **without improving alignment** — the same shape as the coordinate
leakage that put the sibling project's contact probe at AUC 0.997. `dinotxt_eval --smooth-sigma`
Gaussian-smooths the POINTWISE model's per-sequence scores over the same neighbourhoods. **A mixing
arm must beat that, not merely beat the pointwise model.**

### Arms in flight (12k steps each, jobs 3534869/70/71)
| arm | params | note |
|---|---|---|
| `ptw_rot` | 0.89 M | pointwise + `--rot-per-batch` — the MATCHED control |
| `mix1` | 1.42 M | one mixing block |
| `mix2` | 1.95 M | two mixing blocks |

`ptw_rot` exists because the mixing arm otherwise differs from v1 in **two** ways at once (mixing,
and one-rotation-per-batch sampling). Evaluation restricts every arm to rotation 0 whenever
coordinates are in play, so all arms stay on identical voxels — the existing equality assertion
enforces it.

### 8.21.1 T0-B RESULT — token mixing is the largest single gain, and 59% of it is real
`probes/dinotxt_mixcontrol.py`, all arms in ONE process on identical voxels (asserted), 225 test
maps / 177,605 voxels, rotation 0, paired cluster bootstrap.

| arm | E1 top-1 |
|---|---|
| **mix (2 blocks)** | **0.5838** |
| smooth σ=12 Å (pointwise + post-hoc) | 0.4730 |
| smooth σ=8 | 0.4681 |
| smooth σ=20 | 0.4337 |
| smooth σ=4 | 0.4260 |
| pointwise (`ptw_rot`) | 0.3973 |
| volume_prior | 0.3084 |
| mix_shuffled | 0.2886 |

| comparison | diff | 95% CI |
|---|---|---|
| mix − pointwise | **+0.1865** | [+0.1675, +0.2060] |
| **mix − smooth12** | **+0.1109** | **[+0.0947, +0.1283]** |
| smooth12 − pointwise | +0.0756 | [+0.0650, +0.0866] |
| mix − volume_prior | +0.2754 | [+0.2405, +0.3100] |

- **The control absorbs 41% of the gain and mixing still wins by +0.1109.** Naive Gaussian smoothing
  of the POINTWISE scores at σ=12 Å recovers +0.0756 of mixing's +0.1865 — a large geometric
  component that a mixing-vs-pointwise comparison alone would have misattributed to learning. The
  remaining **59% is genuinely learned** and the CI is comfortably clear of zero.
- **Corroborating tell: the shuffled floor RISES with mixing** (0.2886 vs 0.2376 for the pointwise
  model). Local consistency helps even when the sequence pairing is wrong, which is precisely the
  geometric effect the σ sweep quantifies. Two independent measurements of the same artifact agree.
- **The σ sweep is unimodal with an interior optimum at ~12 Å** (4→0.4260, 8→0.4681, 12→0.4730,
  20→0.4337). That the best smoothing scale is ~12 Å — near the ~10 Å contact scale — supports
  §8.18.3's stated motive: our tap's ~4-5 Å half-decay is finer than the neighbourhood that defines
  a residue's environment, and mixing supplies the missing scale. It also means the smoothing
  control was tuned in the model's favour, not against it.
- Depth helps: `mix1` val 0.5344 vs `mix2` 0.6003 (pointwise 0.3422). Not swept beyond 2.
- Not collapsed: off-diagonal cos +0.405, rel-variation 1.20.
- **Headline: E1 0.3938 → 0.5838.** Largest improvement found in this stage, from ~1 M extra
  parameters and coordinates that were already recoverable for free.

**Caveat on the matched control.** `ptw_rot` (0.3973 here) is the pointwise model retrained under
one-rotation-per-batch sampling, so mixing vs pointwise differs in exactly one thing. It is scored
at rotation 0 on 177,605 voxels; the 4-rotation G0 number (0.3938 on 179,354) is a different voxel
set and the two should not be differenced across tables.

### 8.21.2 MACRO-AVERAGED CHAIN METRICS — §8.6's second metric, finally implemented (2026-09-04)
§8.6 specified "per-voxel top-1 accuracy **and mean IoU against the deposited model, macro-averaged
over chains**". Only top-1 had ever been reported. `macro_chain_metrics()` in `dinotxt_eval.py`
adds per-chain IoU and recall, macro-averaged over chains within a map then over maps, with a paired
bootstrap over maps.

| arm | top-1 | macro IoU | macro recall |
|---|---|---|---|
| **mix** | **0.5836** | **0.4023** | **0.5523** |
| smooth σ=8 | 0.4673 | 0.2923 | 0.4519 |
| smooth σ=12 | 0.4716 | 0.2831 | 0.4360 |
| smooth σ=4 | 0.4250 | 0.2656 | 0.4268 |
| smooth σ=20 | 0.4341 | 0.2266 | 0.3708 |
| pointwise | 0.3968 | 0.2440 | 0.4030 |
| mix_shuffled | 0.2894 | 0.1626 | 0.2667 |
| **volume_prior** | 0.3090 | **0.0891** | **0.1814** |

- **The volume prior collapses, exactly as intended: top-1 0.309 → macro IoU 0.089.** Confirms that
  a third of the top-1 metric was "get the biggest chain right". Macro-averaging weights a
  50-residue chain like a 2,000-residue one and removes that.
- **Mixing looks BETTER under the harder metric, not worse.** The smoothing control reproduces
  **40%** of the mixing gain on top-1 but only **31%** on macro IoU (+0.0483 of +0.1583). Smoothing
  helps mainly by tidying already-correct large regions; it cannot conjure a small chain that was
  never predicted, and macro-averaging is what exposes that.
  macro IoU: mix − pointwise **+0.1583** [+0.1399, +0.1775]; mix − best smoothing **+0.1100**
  [+0.0957, +0.1249]; mix − volume_prior **+0.3132** [+0.2849, +0.3418]. All SIG.
- **The optimal smoothing scale MOVES with the metric**: σ=12 Å by top-1, **σ=8 Å by macro IoU**, and
  σ=20 degrades far more sharply on macro (0.2266) than on top-1 (0.4341). Aggressive smoothing wins
  top-1 partly by erasing small chains — which is the artifact, visible directly.
- **Baseline ORDER changes too**: `mix_shuffled` (0.1626) now beats `volume_prior` (0.0891) on macro
  IoU, the reverse of top-1 (0.2894 vs 0.3090). A shuffled arm at least distributes predictions over
  chains; the prior never predicts a small chain at all.
- **Report both metrics from here on.** Top-1 alone systematically flatters any mechanism that
  produces spatial coherence, which is the exact failure mode T0-B was designed to guard against.
