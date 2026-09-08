# CleanDIFT distillation of CryoFM2 — implementation plan

> **Supersedes** the previous 541-line feasibility/design document (2026-08-27), archived at
> `archive/plan_cleandift_superseded_2026-08-31/` with a `WHY.md` explaining what moved where.
> Its §4.1 gate already passed; its §7 measured sweep results are preserved in `CLAUDE.md`
> ("THE TIMESTEP SWEEP"); its reference-implementation mechanics are Appendix A here and its
> P1–P10 pitfalls are Appendix B. This document is self-contained — no `§` cross-references to
> the archived file are load-bearing.

## Context

Every CryoFM2 feature in this project is read at a chosen `(tap, timestep, coupling)` operating
point. The 2026-08-30 high-t sweep (`results/o1_thigh.json`, logged in `GOALS.md`) showed the
**coupled** arm peaks at t≈500 and collapses by t=900 (SS 0.7736 → 0.6396), while the **decoupled**
arm (clean input, timestep asserted anyway) *saturates* high: SS 0.7747 / AA 0.1949 at t=750.

That reframes the CleanDIFT case. The superseded design justified it with a coupled-minus-decoupled
retrieval gap (+0.001 / +0.045 / +0.058 / +0.071 at t = 10 / 250 / 420 / 750), read as distillation
headroom. On the **classification** readout the sign **reverses** at high t — decoupled already wins
— so a student that merely beats the *coupled* teacher proves nothing. What CleanDIFT can still add
is narrow but real:

1. consolidate features across the *whole* noise schedule instead of one hand-picked `t`;
2. delete `t` as a hyperparameter (currently swept, and the sweep is corpus-dependent);
3. give a single deterministic forward pass with no noise draw.

**The prior is unfavourable and must be stated before the run, not after it.** Benefits 1 and 2
are bounded above by the spread of the *decoupled* family over `t`, and near its optimum that spread
is **0.17 SS points** (`o1_thigh.json`: 0.7747 at t=750 vs 0.7730 at t=900). The gate below asks for
**+1.0 point**, roughly 6× that. So consolidation-as-such cannot clear the bar; passing requires
CleanDIFT's *stronger* claim — that a merge over the trajectory beats every point on it, which is
what the paper reports for SD and what is untested here. Benefit 3 (determinism) is real but is
worth 0 accuracy points by construction. This plan is therefore a test of the strong claim, and the
expected outcome is the KILL branch. That is an acceptable reason to spend ~20 GPU-hours; it is not
an acceptable thing to discover halfway through.

The point of this plan is to make the result *interpretable* — this project's recurring failure mode
is a plausible-looking number produced by a metric or control that could not have come out
otherwise.

## Success gate — decided up front

All arms evaluated **in one process on identical residues** (see Verification).

**Primary endpoint, pre-registered, one number:** `acc_SS(student) − acc_SS(dec_best)` on the test
split, aligned framing, tap `up_blocks[1]`. **PASS requires all four:**

1. point estimate **≥ +0.014** (see the measured MDE below) with the 95% **cluster**-bootstrap CI
   excluding 0;
2. the null pair `cou_best` vs `cou_seed1` (identical but for the noise seed) has a CI containing 0
   **and** `|diff| < 0.005` — otherwise the measurement cannot resolve 1 point and the endpoint is
   uninterpretable;
3. `acc_SS(student) − acc_SS(student_ctrl)` **≥ +0.007**, CI excluding 0 (see D12).
4. `acc_SS(student) > acc_SS(raw)` — the student must beat **central 8³ raw voxels**.

Why 1.0 point is the bar: coupled-vs-decoupled at their respective peaks is 0.7736 vs 0.7747 (0.1
pt, nothing); the backbone frame is worth 3.2 pt; the frame-averaging residual is 1.3 pt. A
distillation pipeline buying under 1 pt is a footnote, not a deliverable.

**Caveat on that calibration, stated because this document elsewhere bans the move:** those three
literals come from *different runs and different files*. The `raw` arm alone reads 0.7241 in
`o1_thigh.json` and 0.7506 in `o1_up_blocks1.json` — a **2.6-point swing on the same nominal task**
across runs, larger than the bar itself. The 1.0-point bar is therefore a *judgement about what is
worth shipping*, not a measured scale, and the only numbers licensed to be compared against it are
the paired within-process differences from `o5_cleandift_arms.py`. Do not defend the bar with
cross-file arithmetic.

**Why condition 4 exists.** The original benchmark's own pre-registered kill criterion was "cryofm
must beat raw voxels, else the teacher is out" (`o1_tsweep.json:_meta`), and the teacher only
satisfies it at high `t` on this tap: raw beats CryoFM on SS at t=10 (0.7506 vs 0.7183,
`o1_up_blocks1.json`) and at every `t` at `up_blocks[0]` (0.7379 vs best 0.7133, `o1_tsweep.json`).
On **AA the teacher loses to raw voxels in every run ever measured** (best 0.1949 vs 0.2476-0.278).
Two honest qualifications: raw carries 512 dims against the features' 256, so the comparison flatters
it; and condition 4 is about the *teacher's* margin as much as the student's. But a pipeline whose
output cannot beat 8³ raw voxels has no deliverable, and a student that gains a point on SS while
sitting ~5 points below raw voxels on AA is a narrow one — the write-up must say so rather than leave
`raw` as a reported-but-unjudged column.

**Statistics — this is where the 1.1 pt wobble dies.** All arms share residues, so report **paired**
differences, not two independent SEs (treating 0.775 ± 0.012 vs 0.768 ± 0.012 as "overlapping"
discards most of the power). Bootstrap over the **100 test clusters**, never over the ~14k residues:
the test split is 234 chains but only **100 clusters / 225 entries**, and a residue bootstrap ignores
within-chain correlation. **MEASURED 2026-08-31 (`results/o5_power_check.json`): the design
effect is only ≈1.10, not the 3× asserted here earlier.** Within-chain correlation inflates the SE
by ~10%, so residue-level bars are mildly, not catastrophically, too narrow. The real problem is not
correlation — it is that there are few clusters and the arm-to-arm differences are genuinely noisy.
Use the cluster bootstrap anyway (it is correct and costs nothing), but do not justify it with a
factor that is not there.

**Do the power calculation FIRST, on cached data — do not discover it post hoc.** Gate condition 2
can only fail *after* the ~20 GPU-hours are spent, which is the wrong order. Every ingredient
already exists in `data/o4_lab_parts`: run `cluster_bootstrap_diff` there on an already-measured
large pair (`lab` vs `aligned`, 3.2 pt) and on a near-null pair, and read off the **minimum
detectable paired effect at 100 test clusters**. Minutes of CPU, no GPU. If a 1.0-point effect is
not detectable, the endpoint is not measurable on this split and *that* is the finding — enlarging
the test cluster count (currently capped at 100 by the split) is then the prerequisite, not the
training run.

**`--per-chain` 20 → 60 IS worth it — measured, reversing an earlier claim in this document that
it was the wrong lever.** The 10-vs-20 comparison in `o5_power_check` gives an SE ratio of **0.82 per
doubling** of residues per chain, i.e. within-cluster sampling noise is a real component and does not
sit at its asymptote at 20. Extrapolated to 60 (1.58 doublings) that is a **~27% SE reduction**, which
is worth the 3× eval GPU cost given how tight the margins are. Keep the caveat that it biases the
sample toward buried residues (the `ok` filter needs the Cα ≥ `PATCH//2` = 48 Å from every edge), so
the class balance shifts relative to every `--per-chain 20` number and only within-run paired
comparisons stay valid.

**★ MEASURED FEASIBILITY (step 0b, run 2026-08-31, `results/o5_power_check.json`) — the original
+0.010 bar was NOT resolvable and the bar has been raised to +0.014 accordingly.** On the 395-chain
cached subset (1,313 test residues, **47 test clusters**): median **MDE ≈ 0.027**, and the null pair
`lab3 − lab2` — two arbitrary fixed global orientations, so a true difference of ~0 — reads
**−0.0107**, i.e. the analogue of gate condition 2 *fails outright* at this scale. Projected to the
full split (100 test clusters) at `--per-chain 60`: **MDE ≈ 0.0133**. So:

- The endpoint is measurable on the full split only for effects **≳ 1.4 points**. The bar is now
  +0.014, set by the measurement rather than by taste.
- Resolving the original 1.0 point would need **≈ 178 test clusters** (vs the 100 available);
  resolving the 0.17-point decoupled `t`-spread that motivates benefits 1-2 is off the table by
  roughly two orders of magnitude. This sharpens the Context's prior: the experiment can only ever
  detect an effect **much larger than its own stated rationale predicts**.
- Two literals this document used for calibration do not survive at cluster level on the subset:
  `lab − aligned` reads −0.027 with a CI spanning 0 (the "backbone frame is worth 3.2 pt" claim), and
  `aligned − raw` is +0.062, CI excluding 0 — the one solidly significant comparison in the table.
- **Option, if the 1.0-point resolution is actually wanted:** rebuild the eval split at ~3× the
  cluster count. Cryo2StructData has 7,361 usable entries against the 1,500 chains currently used, so
  this is a `build_alignment_set.py` re-run, not new data. Cost is a larger box cache and a longer
  eval, not a redesign.

**KILL conditions:**
- `student ≤ dec_best + 0.005` with a tight CI → consolidation adds nothing over naive decoupling.
  Publish the negative; it retires the line.
- `student ≈ student_ctrl` → the effect is corpus adaptation, not CleanDIFT.
- **head-share** `(Δcos_bypass / Δcos_with_head) < 0.5` at the judged tap → the heads absorbed the
  learning and the *shipped* trunk barely moved (D15). Not a hyperparameter problem.
- `cos_bypass` flat from step 0 at every tap → the student cannot represent coupled features from
  clean input at all. Also a genuine finding.
- `student ≈ dec_ens` → consolidation is real but a 3-forward `t`-ensemble gets it for free, with no
  training and no new code. Report it that way; it retires the distillation, not the idea.
- per-module weight delta < 1e-3 → **not** a scientific negative, a training failure. Re-run at
  higher lr (see D13).

The decisive null control is built into the initialisation: the student *starts* as the naive
decoupled teacher at `t_init` (D2/D3), so it must beat its own starting point. Include
`decoupled_t{t_init}` as an explicit arm so this is visible, not inferred.

**The gate is deliberately a downstream task, not teacher-feature agreement.** A distilled student
is *trained* to agree with the teacher, so any feature-similarity metric is circular (Appendix B,
P8).

## Design decisions

**D1 — Taps: distil all three of `DEFAULT_TAPS`** (`mid_block`, `up_blocks[0]`, `up_blocks[1]`).
One student forward serves all three, so extra taps cost only head params. Judge on
`up_blocks[1]` — every classification number to date, and the *only* (tap, `t`) combination where
CryoFM beats raw voxels on SS (0.7747 vs 0.7241 at t=750; it loses at t=10 and loses at every `t` at
`up_blocks[0]`). `up_blocks[0]` is retained because it is the ESM-C alignment target (R² 0.230).

**D2 — Learned timestep: a free `nn.Parameter[256]`, NOT a learned scalar.** *Departs from the
superseded design and from the reference implementation's `learn_timestep`.*
`get_timestep_embedding` (`.pixi/.../diffusers/models/embeddings.py:27`) is differentiable w.r.t.
`t` — it documents fractional timesteps and `.float()` is a no-op on a float tensor — but with
`max_period=10000, half_dim=32` the top channel is ~`sin(t)`, so `d/dt` oscillates violently near
t≈420. A free 256-d vector is better conditioned and strictly more expressive.

*Implementation — a one-line attribute swap, no forward patching.* `emb` is a local in
`UNet3DModel.forward` with no hook point, but it comes from exactly one call,
`emb = self.time_embedding(t_emb)` (`third_party/cryofm/.../unet3d/unet.py:311`), and is then
threaded unchanged to all 22 `time_emb_proj` FiLM sites; `self.class_embedding is None` for this
config so the only other writer (`unet.py:307-321`) is dead. So:

```python
class LearnedTimeEmb(nn.Module):          # ignores its input entirely
    def __init__(self, init_vec): super().__init__(); self.p = nn.Parameter(init_vec.clone())
    def forward(self, t_emb): return self.p.unsqueeze(0).expand(t_emb.shape[0], -1)

student.time_embedding = LearnedTimeEmb(teacher_emb_at_t_init)
```

A useful side effect: the student's `timestep` argument becomes a no-op (pass `0`), so **none of the
three `dtype=torch.long` call sites need changing** for the student path.

**D3 — Initialise so step 0 == the baseline.** Set the parameter to the teacher's own
`time_embedding(time_proj(t_init))` and make heads identity-at-init (D4). Then step 0 is *exactly*
naive decoupled at `t_init`, and any gain is attributable to training rather than to the
initialisation. **`t_init = 750`**, not 420: the archived §3.1 converts the reference's SD 261 to FM ≈ 420, but the
2026-08-30 decoupled sweep *peaks* at 750 (SS 0.7747). A measurement beats a conversion table. Keep
`--t-init` as a flag; 420 is the ablation.

The head's FiLM conditioning uses the **teacher's** unmodified `time_proj` + `time_embedding`
(frozen) to embed the sampled `t`; only the student's copy is swapped out.

**D4 — Heads: identity + zero-init residual, conditioned on the TEACHER's `t`.**
`proj_k(f; t) = f + FFN_k(f; FiLM(t))`, `FFN_k` a 1×1×1 conv → SiLU → 1×1×1 conv with the second
conv zero-init. Two things this gets right: a *pure* zero-init head would output 0 and make cosine
undefined (NaN or dead gradient); and the head must take the **teacher's sampled `t`**, not the
student's learned embedding — that is the mechanism forcing one student representation to explain
every `t`. Heads are **discarded at inference** (keeping them reinstates the hyperparameter, since a
head-attached feature is a `t`-conditioned feature).

**Head width is a first-class decision, not a detail — and OUR CHOICE CONTRADICTS THE PAPER'S
INTENT (noted 2026-08-31, after verifying the reference architecture).** The paper uses three stacked
SwiGLU FFN blocks, FiLM per block, 45M params for SD 2.1 — i.e. it deliberately gives the head *ample*
capacity so that it absorbs the `t`-specific part and leaves the trunk free to be `t`-agnostic. The
bottleneck below was chosen for the opposite reason. Both arguments are internally coherent and they
prescribe opposite widths, so this is empirical, and **the evidence so far favours the paper**: the
D13 lr probe measured `head_share` at **0.998-0.999** (lr 1e-5, 3e-5), so the head is absorbing almost
nothing and the bottleneck is insuring against a risk that has not materialised — while plausibly
limiting the fit. `PaperHead` (`--head paper`) is now implemented and running as an arm against the
bottlenecked default. The original reasoning, retained because it is still the right thing to
*measure*:

**Bottleneck it to `C → C/4 → C`.** At full
width the head is a per-token, full-rank, `t`-conditioned MLP, and *any invertible linear channel
remap it applies is invisible to the downstream linear probe*. Nothing in the loss pressures
information into the trunk rather than the head, yet **only the trunk ships**. A full-width head can
therefore absorb the entire coupled/decoupled discrepancy and leave the shipped student exactly at
its initialisation while the training curve looks healthy — the failure mode D15's `cos_bypass`
exists to detect. Narrowing the head makes that outcome *architecturally* harder instead of merely
observable. Record the width in the results JSON; `--head-width` is the ablation.

**D5 — Loss: centred cosine over `dim=1`, mean over spatial cells, summed over taps.**
Three separate correctness points, each of which silently degrades the run if wrong:
- **Axis:** this codebase is channels-first `[B, C, D, H, W]`, so the channel axis is `dim=1`. The
  reference implementation uses `dim=-1` on channels-last tensors; copying that would silently
  optimise a *spatial-pattern* similarity instead of a feature similarity.
- **Centring is mandatory, not cosmetic:** raw cosine on high-dim activations in this project is
  ~0.99 for *any* pair from a shared common component. An uncentred loss would descend 0.990 →
  0.995 while learning nothing about the informative residual. Log raw cosine alongside as a
  diagnostic; it should sit near 0.99 and thereby demonstrate why centring was needed.
- **Mean (not sum) over spatial cells is the per-tap normalisation**, neutralising the 64× token
  imbalance (512 / 4096 / 32768 for the three taps). Write it as `mean` explicitly; "sum with token
  normalisation" is re-derivable-and-wrong later. Weight taps `(1/3,1/3,1/3)`; do **not** tune the
  weights (multiplicity, see the gate). Log the three per-tap cosines separately so a sacrificed tap
  is visible.
- **Centre with a detached EMA of the shared mean, not the in-batch mean.** At B=8, `mid_block`
  offers 8×512 tokens to estimate a 512-dim mean — noisy, and the noise is *correlated within a
  batch* since all tokens come from the same 8 boxes. Per-tap buffer initialised from step 0's
  `0.5*(P.mean(0)+T.mean(0)).detach()`, momentum 0.99, matching the convention at
  `probes/stability.py:90`. Detached, so no gradient path through the centring constant.
- **Compute the loss in fp32** (`.float()` the taps). Cosine over 256-512 channels in bf16 carries
  ~1e-2 relative error and the effects being chased are sub-1%.
- **Pre-check the token population before the primary run** (~20 min, no training). At
  `up_blocks[1]` a box is 32768 tokens and a 96 Å box around a Cα is largely solvent; an unweighted
  mean would spend most of its gradient on cosines between two near-zero-variance solvent vectors.
  Dump `‖T_centred,token‖` for one batch against distance from box centre. If solvent tokens are
  >50% of the population and low-norm, switch to a norm-weighted mean
  (`w = ‖T_c‖.detach()/mean`). Record the choice and its risk (norm weighting also up-weights
  map-edge artifacts) in the results JSON. Do not decide this by intuition.
- **★ Add an explicit CENTRE-TOKEN term — the loss and the endpoint currently read different
  things.** The eval reads `centre2`: the mean of the central **2³** feature cells. At
  `up_blocks[1]` a 64³ box yields a 32³ feature grid, so the shipped, evaluated quantity is **8 of
  32,768 tokens** and a spatially uniform mean puts ~0.02% of the gradient on it — a 4,096:1
  mismatch between what is optimised and what is measured. Norm weighting does *not* fix this; it
  up-weights high-density tokens, not central ones. Use
  `L_tap = 0.5·mean_tokens(cos) + 0.5·cos(centre 2³ mean)`, matching the eval's reduction exactly in
  the second term, and **log `cos_centre` per tap as its own instrument** — it is the only cosine
  the primary endpoint actually depends on. Keep the all-token term: it is the regulariser that
  stops the trunk from degenerating everywhere except one voxel, and it is what the `feature_volumes`
  tiling use case (D10) consumes. `--centre-weight` is the ablation.

**D6 — Teacher `t` sampling: stratified in 3 bins over `[1, 600]`, NOT `[1, 1000]`.** *Departs
from the reference, on this project's own measurements.* Use `FMScheduler.add_noise`
(`third_party/cryofm/.../scheduling_fm.py:130`) rather than the ad-hoc mix in `cryofm_tap.py:271` —
verified to be numerically identical (`noise*t/1000 + x*(1−t/1000)`), so this is hygiene at zero
risk, making the `(x_t, t)` pairing structural instead of conventional.

The departure: `t_max = 999` is inherited from SD, where high-`t` *features* behave differently.
Here, stratifying to 1000 spends **a third of the budget on the bin `[667, 1000]`, where the teacher
is close to empty** — `o1_thigh.json` puts the coupled arm at **SS 0.6396 at t=900 against an
untrained-UNet control of 0.5739**, i.e. ~1/3 of the way from random to useful. Training the student
to reproduce those targets does not buy consolidation; it spends a third of the gradient pulling the
trunk toward the statistics of noise. `t = 1000` is pure noise and carries exactly zero information
about the protein.

Two acceptable forms, and the choice is an ablation, not a conviction: (a) hard cap `t_max = 600`,
just past the coupled peak at t≈500; (b) keep the full range but weight bins by measured teacher
informativeness — the SS-accuracy-vs-`t` curve already exists, so use it as the weight. Prefer (a)
as primary for simplicity. **Record `t_max` in the results JSON**, because a student distilled over
`[1,600]` and one over `[1,1000]` are different objects and the difference is not recoverable later.

**D7 — `weight_decay = 0.0`; do NOT add a ch1 freeze.** *Revised after checking the arithmetic.*
The archived P7 says to freeze the `conv_in` ch1 slice because ch1 is all zeros. But
`dL/dW[:,1] = grad_out ⊗ input[:,1] = 0` **exactly**, so the only drift is decoupled weight decay,
and `(1 − lr·wd)^steps = (1 − 1e-7)^20000 ≈ 0.998`. A freeze hook is machinery for nothing. Set
`weight_decay = 0.0` instead — not for ch1, but because pulling a pretrained 168 M model's weights
toward zero is not something a light distillation should do at all.

**D8 — Run a REDUCED K-draw gate first — as a ceiling test, not a precondition.** *Revised.* The
superseded §7.9 called K-draw a hard prerequisite for *building* the student; that is wrong on the
mechanism, since maximising `E_{t,ε}[cos(proj_t(S(x₀)), T(x_t,ε))]` drives the student toward
`E_ε[T/‖T‖]`, i.e. the student **is** an implicit noise-marginaliser. (The `noise_draw` 0.97-0.996
figure does not bear on this either way, being a t=10 measurement.)

But K-draw still measures **the student's ceiling**, and that is worth 3 h in front of a ~20 h
commitment: if `E_ε[f(x_t,t)]` is no better a per-residue descriptor than `f(x₀,t)`, the target
carries nothing the decoupled arm already has and the training run is *guaranteed* to fail.
**Run it at t = 500, not t = 750 — an earlier version of this decision made a cross-arm selection
error.** The K-draw target is the **coupled** teacher, so its timestep must be selected on the
*coupled* sweep, where the peak is t≈500 (SS 0.7736). t=750 was inherited from the *decoupled* peak,
which is the right choice for D3's `t_init` (the student consumes clean input) and the wrong one
here. `CLAUDE.md` states the rule this violated: "the optimum is ARM-SPECIFIC and no cross-arm
comparison is licensed." The cost of the error was not cosmetic — at t=750 the coupled arm scores
0.7476 against decoupled 0.7747, so noise-averaging would have had to climb **2.7 points to reach
parity before earning the +1.0**, a ≥3.7-point ask levied at the coupled arm's worst tested point.
Sweeping t ∈ {500, 750} is affordable and preferable; if only one, use 500.

Design: t=500, `up_blocks[1]`, 300 chains, K=8 **cumulative** draws (so K=1,2,4,8 come from the same
8 forwards, free saturation curve), plus `dec_best` on the same boxes; fit accuracy vs `1/K` and
**extrapolate to `1/K = 0`** (K=8 still leaves ~35% of the single-draw noise std, so K=8 undershoots
the ceiling). **Gate: extrapolated `acc(K=∞) − acc(dec_best) ≥ +0.010`**, with the same cluster
bootstrap as the primary endpoint — a point estimate on 300 chains is not a gate. Abandon
`data/kdraw_t250/` (3 entries, wrong t) and redo — **after the `centre2` fix (order of work step
0)**, since with that bug K draws would remove the *same* field every time and flatter
noise-averaging into looking like signal.

**D8 OUTCOME (2026-08-31, job 3489041): UNRESOLVED — see `GOALS.md`.** SS extrapolated gain
**+0.0101** against the +0.010 bar (a margin ~1% of the gate's own MDE of 0.0129, CI crossing zero,
40 clusters); **AA gain is NEGATIVE (−0.0066)**; noise-averaging saturates at K=2. Combined with step
0b's resolution floor of ~+0.014, only effects above ~1.4 points are conclusive here.

**CORRECTION (same day): this gate does NOT bound the distillation, and an earlier version of this
note wrongly let it carry a stop recommendation.** D8 assumed noise-marginalisation headroom proxies
for distillation headroom. It does not. K-draw averages the coupled teacher over noise draws at a
FIXED t, measuring the *noise-marginalisation* channel; CleanDIFT's claimed gain comes from
*`t`-consolidation* — one representation explaining the teacher across the whole schedule, which the
paper reports as **exceeding the max over t**, i.e. an extrapolation outside the decoupled family's
range that neither the family's spread (the "0.17 pt" framing) nor this gate constrains. **The
expected effect size is therefore unknown and unbounded by anything measured here**, and step 0b's
~1.4-point resolution floor is the only surviving constraint — a statement about the ruler, not the
effect.

Two priors now pull in opposite directions, and both are empirical rather than bounds. FOR: the
`dec_*` arms feed CLEAN input while declaring a high `t` — off-manifold, an untrained hack — and are
nonetheless the best CryoFM arms measured on both tasks (SS 0.7747, AA 0.1949, beating every coupled
point). CleanDIFT is the principled version of exactly that hack, so student >= decoupled is
plausible. AGAINST: in Stable Diffusion clean-input features are *bad*, so DIFT needs noise and
CleanDIFT has a large gap to close; here the clean-input arm is already on top, so the mechanism
generating the paper's gain may be weaker in this setting.

**Revised decision: proceed.** Run the D13 lr probe, then the primary runs. Only an effect >~1.4
points will be conclusive; a smaller positive result is suggestive but unresolvable, and enlarging
the eval split to ~180 test clusters is the fix if that is where it lands.

**D9 — Plain distillation first; frame-averaged teacher is arm B, gated on arm A.**
The superseded §6 recommended going straight to the frame-averaged teacher. Deferring it keeps arm A
a single-variable test, so a negative result says *which* thing failed. The 2026-08-30 frame-free
numbers give arm B a precise target if arm A works: recover the ~1.3 SS-point residual that K=4
orientation averaging leaves. Cube-rotation augmentation of the input is still used in arm A
(lossless permute+flip via `probes/o4_frameavg_benchmark.py:torch_cube_rotate`).

**D10 — Training samples: a 50/50 mixture, cut from the volume cache.** *Departs from the superseded
design's pure random crops, and from an earlier version of this decision that used backbone frames
only.* The eval feeds Cα-centred, frame-rotated, trilinearly resampled boxes; a student distilled on
patch-grid crops alone is off-distribution at eval (`RotCube24` is lossless transpose+flip, whereas
`grid_sample` boxes are interpolation-blurred). So:
- **50% Cα-centred boxes with uniform random SO(3) frames** via `extract_local_boxes`. Random SO(3)
  rather than backbone frames deliberately: it covers both the `aligned` and `lab*` eval arms and
  gives orientation diversity without committing the student to the atomic-model-dependent frame
  distribution.
- **50% patch-grid 64³ crops with `RotCube24(p=1.0)`**, protecting the `feature_volumes` tiling use
  case that "deterministic feature extractor" implies downstream.

Shuffle buffer: a reservoir over the last ~6 maps (6 × 128 boxes × 64³ fp16 ≈ 400 MB CPU), refilled
one map at a time. Without it every batch of 8 comes from a single map — correlated gradients *and* a
correlated centring EMA (D5).

**D11 — Split discipline: exclude MAPS, not just clusters.** *Corrected — an earlier version of
this decision was wrong.* Training on `split == train` guarantees cluster disjointness but **not**
density disjointness: an EMDB entry can host chains in different splits. Measured on
`data/alignment_chains.csv`: **64 entries host both train and test chains, 50 host train and val.**
Training on a train chain from a shared entry means the student has seen, unsupervised, the density
that contains test residues — exposure `dec_best` and `cou_best` never had.

Build the distillation list as train-split rows whose `emd` appears in **neither** val nor test.
Measured survivors: **958 chains / 798 maps / 466 clusters / 245,633 residues** — ample (the
reference used ~3k images). Assert it in code.

Residual limitation to state rather than fix: unlisted Cryo2StructData maps could still be
homologous to test clusters, and 15.1% of Cryo2StructData is in CryoFM2's own pretrain set
(`data/cryofm2_pretrain_lists/train.csv`). Restricting to `alignment_chains.csv` train rows, where
cluster membership is known, is the cleaner choice for exactly this reason — and D12 is the control
that makes the residual harmless, since it carries identical exposure.

**D12 — `student_ctrl`: a self-distilled control, and it is not optional.** The student is trained
(unsupervised) on maps from the same corpus the eval draws from; the teacher never was. Any gain is
therefore confounded with corpus adaptation / self-distillation regularisation. `student_ctrl` is
trained *identically* — same corpus, same steps, same loss, same seed — except its teacher is fed
**clean** input at fixed `t = t_init`, so it can learn nothing about noise marginalisation. If
`student ≈ student_ctrl`, the effect is not CleanDIFT and the write-up must say so. Also cheap
(~15 min): **`student_t_only`**, trunk frozen, only the D2 embedding + heads trainable — if that
recovers most of the gain, all CleanDIFT bought was a better conditioning vector, which a sweep
gets for free.

**D13 — Budget for a FAR target: 20k steps, and probe the lr.** *Departs from the reference's 400
steps / lr 1e-5.* That budget is calibrated to SD, where the student's initialisation is already
close to its target. Here it is not: coupled and decoupled features at matched `t` have median
centred cosine **0.003 to −0.44** (near-orthogonal). Importing 400 steps at lr 1e-5 is the single
most likely way this run produces a student indistinguishable from its own initialisation — i.e.
from the naive decoupled baseline — which would then be written up as "CleanDIFT gives no gain"
when what happened is "the optimiser never moved". Plan **20k steps** as primary, treat 4k as a
smoke test, and run a **3 × 500-step lr probe** at 1e-5 / 3e-5 / 1e-4 selected on val `cos_bypass`
and on weight-delta magnitude. Do not exceed CryoFM2's own pretrain lr of 1e-4.

**D14 — Truncate the forward after the deepest tap (`stop_after`).** The tail past `up_blocks[1]`
runs at 64³ and is roughly **half the FLOPs** of the network while contributing nothing to the loss;
`up_blocks[2..3]` are pure waste for both training and evaluation. Implement by raising a
`StopForward` exception from the tap hook — safe because forward hooks fire after the module returns,
so no `torch.utils.checkpoint` frame is mid-flight and the graph for everything already computed is
intact. Guard with a startup assertion that truncated and full forwards give **identical** taps.
Related memory note: never bind the model output (`student.model(x, timestep=t)` with no
assignment), since `up_blocks[2..3]`'s saved tensors are only reachable from the returned
`UNet3DOutput.sample`.

**D15 — Instrumentation that decides whether any number means anything.** Three logs, each guarding
a specific silent failure:
- **Per-module `‖W_student − W_teacher‖ / ‖W_teacher‖`.** If < 1e-3 everywhere at the end, the
  student *is* the teacher and no evaluation number means anything regardless of its CI (D13).
- **`cos_bypass` per tap per `t`-bin on a held-out batch** — the training cosine with the head
  *skipped*. The heads are discarded at inference, so if `cos_with_head` rises while `cos_bypass`
  stays flat, the head is absorbing the whole coupled/decoupled discrepancy and the shipped student
  is unchanged. Stop the run if that happens; it is not a hyperparameter problem.
  **State the criterion relatively, not as an absolute threshold.** The KILL condition's "plateaus
  below ~0.3" is an invented number, and this project has twice paid for gating on an invented
  threshold (a 0.5 cosine bar with no mismatched floor; a top-1 ≥ 0.5 bar that was unsatisfiable by
  construction). The measurable quantity is the **share of the total cosine gain that survives head
  removal**: `(cos_bypass_final − cos_bypass_step0) / (cos_head_final − cos_head_step0)`. Require
  **≥ 0.5** — i.e. at least half the learned agreement lives in the shipped trunk. That is
  scale-free, needs no prior about attainable cosine, and is the quantity the mechanism claim
  actually rests on. Keep the absolute curve as a diagnostic, not as the gate.
- **Uncentred cosine beside the centred one.** Expect ~0.99 and flat; it is the control proving the
  centring in D5 is doing what it claims.
- Plus `nearest_timestep(teacher, student.time_embedding.p)` every 500 steps. Leaving the 1-D
  manifold of valid timestep embeddings is the honest cost of D2: "the student learned t=683" stops
  being a sentence one can say. Report the nearest `t` and the relative residual; a large residual
  is a finding ("the learned conditioning is not any timestep"), not a failure.

**D16 — Two student seeds. The student is otherwise n = 1.** Teacher-side noise gets a whole null
pair (`cou_best` / `cou_seed1`) and a validity condition, while the arm the endpoint is *about* is a
single sample. `student_ctrl` shares the seed by design (D12), so it controls for seed without
estimating its variance. Train the primary student at two seeds (~1.7 h each) and report the endpoint
as a range: if the two differ by more than the bar, the gate is not resolvable from one run and no
amount of cluster bootstrapping fixes it, because the bootstrap resamples *evaluation* residues and
not *training* randomness. This is the cheapest remaining variance estimate in the plan and it
guards the one number everything else is built to protect.

## Files

**New — `probes/build_vol_cache.py`.** One-off cache of **whole preprocessed volumes**, not boxes —
so boxes can be cut at arbitrary centres and frames later without re-paying preprocessing, which is
the dominant cost in every probe (measured: `load_map` 0.0-0.4 s, `preprocess` **1.2-2.9 s** per
map). Writes `data/cleandift_vols/<emd>.npy` + `<emd>.json` (`origin`, `voxel_size`, `shape`),
per-map incremental (`if out.exists(): continue`), never raising out of the loop. ~1,147 entries ×
~68 MB = **~78 GB** (2.5 PB free, so not a constraint).

**Use fp32, not fp16.** fp16 gives ~5e-4 relative error on preprocessed values — a confound that
cannot be falsified afterwards at the 0.1-point level being chased. 78 GB is cheap insurance
against an unfalsifiable result.

**Preprocess the whole map once, then cut boxes — never per-crop**, because `preprocess` takes a
99.999th percentile over the full volume, so per-crop normalisation is a *different transform* from
the one every existing result used. Reuse verbatim: `load_map` (`probes/stability.py`),
`to_cubic_even` / `backbone_with_resnum` / `ss_labels` (`probes/o1_cryofm_benchmark.py`),
`preprocess` (`teachers/cryofm_tap.py:135`), `extract_local_boxes`
(`probes/local_frame_stability.py:86`).

Padding subtlety to respect in training as well as eval: `extract_local_boxes` uses
`padding_mode="zeros"`, and 0 in preprocessed units is raw density 0.04 — *above* background (raw 0
maps to −0.44). An out-of-bounds box therefore gets a slab of mean-density, not vacuum. Apply the
eval's `ok` filter (Cα ≥ `PATCH//2` from every edge) during training too, or the student learns on
artifacts the eval never shows it.

**New — `probes/cleandift_train.py`.** The trainer.
- `class StudentTap` — wraps a trainable `load_cryofm2(...)` copy; non-detaching hooks; swaps
  `time_embedding` for `LearnedTimeEmb` (D2); `weight_decay=0.0`, no ch1 freeze (D7).
- `class ProjHead(nn.Module)` — D4 head; `forward(self, f, t_emb)`.
- `def distill_loss(student_feats, teacher_feats, taps) -> Tensor` — D5.
- Loop copied in style from `probes/stage1_pair_head.py:269-318`: AdamW, manual linear warmup
  (`g["lr"] = args.lr * min(1.0, step / args.warmup)`), `clip_grad_norm_(…, 1.0)`,
  `torch.autocast("cuda", torch.bfloat16)`, gradient accumulation, periodic student checkpoint
  (preemption), best-val snapshot.
- **20k steps primary, 4k smoke, lr chosen by the D13 probe** (1e-5 / 3e-5 / 1e-4), batch 4-8,
  constant-with-warmup. Do *not* import the reference's 400 steps / lr 1e-5: see D13 for why that
  budget is calibrated to a target the initialisation is already near, which is not the case here.
- Enable `enable_gradient_checkpointing()` if memory is tight (`unet.py:103` supports it).
- **Log the step-0 loss as an explicit baseline row** so a converging curve can be compared against
  "did nothing".

**New — `probes/o5_kdraw_gate.py`.** The reduced K-draw ceiling gate (D8). Runs before any training.

**New — `probes/o5_cleandift_arms.py`.** Single-process arms evaluation, structurally a copy of
`probes/o4_lab_arms.py` (which already computes every arm from the same `extract_local_boxes` output
into one per-chain `.npz`, so arms share residues exactly). 1,500 chains, `--per-chain 60`,
`stop_after` truncation on, aligned framing, tap `up_blocks[1]`:

| arm | extractor |
|---|---|
| `student` | student weights, clean input, `LearnedTimeEmb`, no `t` |
| `student_ctrl` | the D12 self-distilled control |
| `student_t_only` | D12, trunk frozen |
| `dec_best` | teacher, clean input, `t*` selected on **val** |
| `dec_alt` | teacher, clean input, t=900 — a change that shouldn't matter (scale check) |
| `dec_ens` | teacher, clean input, features **concatenated over t ∈ {261, 500, 750}** — the free rival |
| `dec_ens_pca` | `dec_ens` reduced to 256 dims, so the comparison is dimension-matched |
| `student_seed1` | the primary student at a second training seed (D16) |
| `cou_best` | teacher, coupled at its own best t, **selected on val**, `noise_seed=0` |
| `cou_seed1` | identical to `cou_best` but `noise_seed=1` — **the null pair** |
| `raw` | central 8³ raw voxels (already in the template) |
| `rand_student` | random-weight UNet, clean input — house policy (Appendix B, P8) |

**`dec_ens` is not optional — it is the zero-training competitor to the stated rationale.**
Benefit 1 is "consolidate features across the whole noise schedule"; concatenating three decoupled
extractions does exactly that for three forward passes and no training. The features are already
computed for the `dec_*` arms, so the arm costs a `np.concatenate`. If `dec_ens` matches the student,
CleanDIFT bought nothing that an ensemble does not, and that is the honest headline — the paper's own
claim is that the student beats *noise*-ensembling, and `t`-ensembling is the analogue here. Report
`dec_ens_pca` beside it because `dec_ens` carries 768 dims against the student's 256, and this
project has been burned by unmatched-dimension comparisons before.

Two changes to the template beyond the arms: **add the val split** (`o4_lab_arms.py` currently uses
only train/test and drops val, but D13's checkpoint selection and `dec_best`'s `t*` both need it —
and test is read exactly once), and add `cluster_bootstrap_diff(ok_a, ok_b, clusters, n_boot=2000)`
returning `(diff, lo95, hi95, p)` over the 100 test clusters. McNemar's exact test on discordant
pairs is reported as a secondary, flagged as anticonservative here since it assumes independent
residues.

Cache hazard specific to adding arms later: `if part.exists(): continue` will skip a chain whose
`.npz` predates a new arm, and the aggregation loop then raises `KeyError`. Validate the key set on
load and re-extract when a key is missing, rather than skipping:
`if set(keys) - set(z.files): part.unlink(); re-extract`.

Secondary table, separate job, 300 chains: `labavg4` framing for `student`, `dec_best`,
`student_ctrl` only.

**Edit — `teachers/cryofm_tap.py`.** **Four** additive changes — `detach`, `stop_after`,
`student_ckpt`, and the `resolve_tap` refactor. No dtype changes are needed anywhere (D2) and no
existing call site changes behaviour, but this file is imported by every probe in the project, so
step 3 of the order of work re-runs `o4_lab_arms` against a cached result to prove nothing moved.
The two that carry the gradient plumbing:
1. `_register` (line 179): add a `detach: bool = True` constructor flag; when false the hook stores
   `t` without `.detach().float()`. This is the whole of the gradient plumbing — a fork of this file
   would rot, and `@torch.no_grad()` on `feature_volumes`/`sample_frame_averaged` can stay, because
   the student path calls the model directly (as `local_frame_stability.py:158-165` already does)
   rather than through those wrappers.
2. `load_cryofm2` (line 112): add `trainable: bool = False` → `requires_grad_(True)`. `.eval()` is
   harmless for gradients here (dropout 0.0, GroupNorm only, no BatchNorm), so `requires_grad` is
   the only thing that genuinely needs changing.

**New — `slurm/cleandift.slurm`.** Copy `slurm/o4.slurm` (positional `"$@"`, `gpu:h100:1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Positional args are required: `sbatch --export`
silently truncates comma-containing values, which has already corrupted one sweep in this project.

## As built (2026-08-31)

Implemented; the module map differs from the sketch above where factoring helped, so this is the
authoritative list. Everything imports clean and every CLI answers `--help`.

| file | role | status |
|---|---|---|
| `probes/o5_stats.py` | paired cluster bootstrap, MDE, McNemar (secondary), between/within variance split | done, unit-tested incl. a degenerate input |
| `probes/o5_power_check.py` | **step 0b** gate + scale projection | done, **RUN — see below** |
| `probes/build_vol_cache.py` | step 1, whole preprocessed volumes, fp32, atomic + requeue-safe | done, running |
| `probes/cleandift_data.py` | `split_map_lists` (D11 assertions) + `BoxStream` (D10 50/50 mixture, shuffle buffer) | done, split verified |
| `teachers/cryofm_student.py` | `LearnedTimeEmb` (D2), `embedding_for_t`, `attach_const_time_embedding` (D3), `FiLMHead` (D4, bottlenecked), `freeze_trunk` (D12), `nearest_timestep` (D15), `--self-test` | done |
| `probes/cleandift_train.py` | the trainer: `DistillLoss` (D5 incl. the centre-token term), stratified `sample_t` (D6, `t_max=600`), `weight_delta` (D15), `--mode distill\|ctrl\|t_only` (D12), atomic snapshots | done |
| `probes/o5_boxes.py` | shared per-chain box extraction, so the gate and the arms cut *identical* boxes | done |
| `probes/o5_kdraw_gate.py` | step 2 (D8), **t=500**, cumulative K, 1/K extrapolation + cluster bootstrap | done |
| `probes/o5_cleandift_arms.py` | step 8, 12 arms one process, val split carried, gate evaluated in code | done |
| `probes/o5_verify.py` | GPU verification of every identity the plan calls load-bearing | done, queued |
| `teachers/cryofm_tap.py` | `detach`, `stop_after` + `StopForward`, `forward_taps`, `resolve_tap`, `trainable`, `load_student` | done |
| `probes/o4_frameavg_benchmark.py` | **step 0**: `centre2` generator hoisted, `--noise-seed`, `--legacy-noise` | done |
| `slurm/cleandift.slurm`, `slurm/cleandift_cpu.slurm` | GPU and CPU launchers, positional `"$@"` | done |

Deviations from the sketch, all deliberate: `StudentTap` was unnecessary once `CryoFM2Tap` gained
`detach`/`trainable`/`stop_after`, so there is no wrapper class; `ProjHead` lives in
`teachers/cryofm_student.py` as `FiLMHead` beside the other student pieces; `distill_loss` is a
`DistillLoss` class because the EMA centring buffers need to persist across steps; box sampling is
its own module (`cleandift_data.py`) because both the trainer and the val stream consume it. Volume
caching runs on the **`cpu` partition** — it is mrcfile + Fourier resample + numpy, so it should not
hold an H100 for an hour.

## Order of work

0. **Fix the `centre2` noise bug FIRST** (`probes/o4_frameavg_benchmark.py:57`; logged in
   `GOALS.md` 2026-08-31). Hoist the generator out of the batch loop, add `noise_seed: int = 0`, add
   `--legacy-noise` to reproduce old numbers byte-for-byte. Confirm `--help` shows the flag before
   launching anything. **Nothing with a coupled arm is trustworthy until this lands**, including
   D8's gate.
0b. **Power check on cached data, before any GPU spend** (minutes, CPU). Implement
   `cluster_bootstrap_diff` and run it over `data/o4_lab_parts` on a known-large paired difference
   (`lab` vs `aligned`, 3.2 pt) and a near-null one, to obtain the **minimum detectable paired effect
   at 100 test clusters** and the within- vs between-cluster variance split. This fixes
   `--per-chain` (see Statistics) and pre-tests gate condition 2. **If 1.0 pt is not detectable, stop
   — the endpoint is not measurable on this split and enlarging the test set is the prerequisite.**
1. `probes/build_vol_cache.py` — all ~1,147 entries. ~1 h, requeue-safe.
2. `probes/o5_kdraw_gate.py` — the ceiling gate (D8), at t=500. **Stop here on failure.**
3. `teachers/cryofm_tap.py` edits: `detach`, `stop_after`, `student_ckpt`, and a `resolve_tap`
   helper extracted from `_register` (pure refactor). Then the D14 truncation-equality assertion,
   and re-run `o4_lab_arms` on 20 chains against a cached result to prove no existing probe moved.
4. `teachers/cryofm_student.py`: `LearnedTimeEmb`, `embedding_for_t`,
   `attach_const_time_embedding`, `nearest_timestep`, `FiLMHead`. Then the D3 identity assertion.
5. `probes/cleandift_train.py` + 100-step smoke.
6. lr probe, 3 × 500 steps (D13).
7. Primary run at **two seeds** (D16) + `student_ctrl` + `student_t_only` (D12), same
   sampler/step count.
8. `probes/o5_cleandift_arms.py` — all arms, one process; then the 300-chain `labavg4` secondary.
9. Statistics + report: cluster bootstrap, null-pair validity check, weight-delta table,
   `nearest_timestep` readout, training curves (with-head / bypass / uncentred).
10. Record in `GOALS.md` against the gate; append the outcome to this file.
11. **Only if the gate passes: confirm on the actual deliverable.** SS/AA linear probing is a
    *proxy*; the project's product is the ESM-C → density alignment (R² 0.230). A +1-point SS gain
    need not move R² at all, and the gate cannot tell you. Re-extract per-residue targets with the
    student (P9: heads discarded, so the cached t=10 targets in `data/density_targets/` are
    unusable) keeping the 672-cluster split in `data/alignment_chains.csv`, then re-run
    `probes/align_esmc_density.py`. Budget ~3.7 h extraction + ~1 min alignment. **A pass on the
    proxy with no movement on R² is a negative result for the deliverable** and must be written up
    as one.
12. Only if arm A succeeds: arm B, frame-averaged teacher (D9).

## Verification

- **Grad plumbing:** assert `student_feats["up_blocks[1]"].requires_grad` and that a backward
  populates `.grad` on the D2 parameter and on a mid-network weight. A silent `.detach()` would
  otherwise train only the heads and still show a falling loss.
- **D3 identity (the load-bearing check):** with heads at init and the D2 parameter at `t_init`, the
  student's features must equal `centre_features(teacher, boxes, timestep=t_init,
  noise_level=0.0)` to numerical tolerance (same weights, same input, same `emb` — exact in fp32,
  within autocast noise in bf16). If they differ materially, "step 0 = baseline" is false and the
  null control is gone. This check is cheap and catches a wrong `t_init` embedding, a head that is
  not identity-at-init, and an accidentally-still-detaching hook.
- **Loss sanity:** step-0 centred cosine must be well below 1.0 (if it is ~0.99, centring is not
  working and D5's hazard has bitten); raw cosine logged beside it should be ~0.99. Note the centring
  uses the *shared* mean `0.5·(P̄ + T̄)`, matching `centred_cosine` (`probes/stability.py:90`) and
  hence every existing number in the project — with the consequence, worth logging once, that part of
  the loss can be reduced by matching means alone.
- **Centre-token term wired to the eval's reduction (D5):** assert `cos_centre` computed in the
  trainer equals `centred_cosine` between the trainer's centre-2³ pooled student and teacher vectors
  on the same batch. The endpoint depends on 8 of 32,768 tokens; a term that is supposed to target
  them must be verified to target them.
- **`dec_ens` dimension parity:** assert `dec_ens_pca` is 256 dims and that `student`, `dec_best` and
  `dec_ens_pca` all match on dimension before any comparison is printed.
- **Split assertion (map level, not cluster level):** assert no training `emd` appears in the val
  or test splits, *and* no training cluster appears in either. The cluster-only form passes on all
  **64 entries that host both train and test chains** (verified in `data/alignment_chains.csv`),
  which is exactly the leak D11 exists to close. Assert the surviving list is 958 chains / 798 maps
  / 466 clusters and fail loudly on any other count.
- **Arms parity:** `o5_cleandift_arms.py` asserts every arm has identical residue count, `aa`/`ss`
  vectors, and split masks. The ~1.1-point run-to-run wobble measured on 2026-08-30 (`aligned` read
  0.7799 vs 0.7685 across two processes on the same 395 chains, because the residue subsample is
  drawn from a fresh rng) makes cross-process comparison invalid, so this must be enforced, not
  assumed.
- **End-to-end:** `sbatch slurm/cleandift.slurm probes.o5_cleandift_arms --limit 400` reproduces
  the existing `aligned` numbers for the teacher arms (SS ≈ 0.77-0.78), confirming the harness
  matches `results/o4_lab_arms.json` before any student number is believed. Reproduce *within one
  process*: the `raw` arm alone reads 0.7241 in `o1_thigh.json` and 0.7506 in `o1_up_blocks1.json`,
  so agreement to better than ~2.6 pt is not expected across files and is not what is being checked.

## Implementation nits (each one silent if wrong)

- **Student checkpoint round-trip must apply the D2 swap BEFORE `load_state_dict`.** After the swap
  the keys are `time_embedding.p`, not `time_embedding.linear_1.*`, and `load_cryofm2` **raises** on
  any missing/unexpected key (`teachers/cryofm_tap.py:112`). Construct → swap → load, and keep the
  raise; do not paper over it with `strict=False`, which would silently drop the learned embedding
  and evaluate the student at whatever `t_init` happened to be in the constructor.
- **`LearnedTimeEmb.forward` should `.repeat`, not `.expand`.** `.expand` returns one shared buffer
  across the batch; the current resnets never write to `temb` in place, so it is safe *today*, but
  `.repeat` costs nothing at 256 floats and removes a footgun that would surface as a silent
  cross-sample coupling.
- **`StopForward` handling (D14) must re-raise anything that is not `StopForward`.** A bare
  `except Exception` around the truncated forward would convert a real CUDA or shape error into a
  silently short feature dict.
- **D2 wording:** the channel that behaves like `sin(t)` is the *lowest-index* one (highest
  frequency, period ≈ 6.3 in `t`), not the "top" channel. The conditioning argument is unaffected —
  `d/dt` does oscillate violently — but `nearest_timestep` (D15) inherits that sensitivity, so report
  the relative residual alongside the nearest `t` rather than the index alone.
- **`cou_best`'s timestep is selected on val**, like `dec_best`. Selecting it on test would leak, and
  selecting it from a literal in `o1_thigh.json` would import a cross-run number the gate section
  explicitly bans.

## Compute

Forward at 64³ is ~2.40 TFLOP/sample full, **~1.06 TFLOP/sample truncated after `up_blocks[1]`**
(D14). At a realistic 100 TFLOP/s achieved bf16 for conv3d on an H100:

| step | cost |
|---|---|
| 0 — `centre2` fix + re-verify | ~15 min |
| 0b — cluster-bootstrap power check on cached parts | **minutes, CPU** |
| 1 — volume cache, ~1,147 maps, fp32, ~78 GB | **~1 h** |
| 2 — K-draw ceiling gate, 300 chains × 40 res × 9 forwards, t=500 | **~3 h** |
| 3-4 — tap/student plumbing + assertions | ~20 min |
| 5 — trainer smoke, 100 steps | ~10 min |
| 6 — lr probe, 3 × 500 steps | **~1 h** |
| 7 — 2 seeds + ctrl + t_only, 20k steps each | **~4 × 1.7 h ≈ 7 h** |
| 8 — primary eval, 1,500 ch × 12 arms, truncated | **~3-9 h**, 1-2 requeues |
| 8b — secondary `labavg4`, 300 ch | **~2 h** |
| **total (through the gate)** | **~17-24 GPU-hours**, 9-11 slurm jobs |
| 11 — ESM-C confirmation, *only if the gate passes* | **~4 h** |

Step 8's range is set by `--per-chain`, which step 0b decides (20 → ~3 h, 60 → ~9 h); the default is
20 until the variance decomposition justifies more. `dec_ens` / `dec_ens_pca` add no forwards — they
are concatenations of the `dec_*` extractions — so 12 arms cost roughly what 9 did, provided t=261 is
among the extracted `dec_*` points.

Per-step arithmetic (B=8, both models truncated, no gradient checkpointing): teacher fwd 8.5 +
student fwd 8.5 + student bwd ~17 ≈ **34 TFLOP/step** → 0.23-0.34 s/step → 20k steps ≈ **1.3-1.9 h**.
Peak memory: two 168 M models (1.3 GB fp32) + AdamW state on the ~140 M trainable trunk (~1.7 GB) +
activations — comfortable on 80 GB. `enable_gradient_checkpointing()` is wired and safe with the taps
(block-level, `use_reentrant=False`, taps sit outside the checkpointed regions), so turn it on only
on OOM.

Disk ~78 GB for volumes + 672 MB per student checkpoint (keep 2 alternating atomic slots for
preemption; snapshot every 200 steps via temp path + `os.replace`).

**This is ~20 GPU-hours, not the "sub-hour" the superseded design claimed.** The per-step cost is
genuinely cheap; the imported 400-step budget was not defensible for a near-orthogonal target (D13).

## Out of scope / known limits

- **P3 stands** (Appendix B): distillation cannot create invariance the teacher lacks, so arm A is
  not expected to fix the ~1.3-point frame-free residual. That is arm B's job.
- **P9:** heads are discarded, so student features are not in teacher units. The 1,500 cached t=10
  targets in `data/density_targets/` **cannot** be mixed with student features; the ESM-C alignment
  needs re-extraction (keeping the 672-cluster split) before any cross-modal claim. This is no longer
  merely out of scope — it is **order of work step 11**: SS/AA probing is a proxy, and the ESM-C
  alignment is the deliverable the proxy stands in for.
- **P10:** `CompVis/cleandift` ships no LICENSE. Reimplement from the paper; do not vendor code.
  CryoFM itself is Apache-2.0, so the backbone side is clear.
- Simulated-vs-experimental (Phase 0d) is untouched; this plan trains and evaluates on deposited
  experimental maps throughout.

---

# Appendix A — What CleanDIFT is, and the reference implementation

Stracke et al., *CleanDIFT: Diffusion Features without Noise*, CVPR 2025 (oral).

Diffusion-feature extraction (DIFT and successors) needs the input **noised to a timestep t** before
the backbone produces semantically useful activations; that costs information, forces a per-task
choice of `t`, and makes features stochastic. CleanDIFT removes all three with a lightweight
unsupervised distillation:

- **Teacher**: the frozen original diffusion model, fed the *noisy* `x_t` with timestep `t`.
- **Student**: a trainable *copy of the same backbone*, fed the *clean* `x_0`.
- **Objective** (K feature maps, cosine similarity):

      L = - sum_{k=1..K} sim( proj^(k)( feat_c^(k)(x_0) ; t ),  feat^(k)(x_t ; t) )

- **Projection heads** `proj^(k)`: **VERIFIED against the paper 2026-08-31** (arXiv 2412.03439v2,
  fetched and quoted) — **"three stacked Feed Forward Networks (FFNs) that are zero-initialized such
  that initially they act as identity mappings due to their residual connections"**, with **"a FiLM
  layer in each FFN block to adaptively scale activations depending on the timestep t"** and **"the
  SwiGLU [57] gating mechanism as an activation function in each FFN block"** — **"45M additional
  trainable parameters for SD 2.1"**. One clean student representation must explain the teacher's
  features at *every* `t`, so the student is pushed to a `t`-agnostic superset while the head handles
  the `t`-specific part.
  - **Discarding is confirmed:** *"For feature extraction at inference time, we usually discard the
    projection heads and directly use the feature extraction model's internal representations."*
  - **CORRECTION — the "+0.24 PCK" figure was misattributed here.** An earlier version of this
    appendix wrote "Heads are discarded at inference (+0.24 PCK in ablation)", which reads as
    *discarding* buying 0.24. Table 3 actually compares **training WITH heads vs training WITHOUT
    heads at all**; *both* arms discard at inference. The 0.24 pp PCK_img (and 0.06 pp PCK_bbox,
    cosine) therefore argues for **having** heads in the training setup — which is what we do — not
    for keeping them at test time.
  - **Our head is ~50x smaller and 1 block, not 3** (`FiLMHead`: 0.20M at C=512 vs `PaperHead`:
    10.24M). See the D4 note.
- **t sampling**: stratified, `num_t_stratification_bins = 3`, `t_min = 1`, `t_max = 999` — i.e.
  essentially the whole trajectory; only `t = 0` is excluded.
- The student still needs *some* timestep argument (architectural requirement). The reference passes
  a learnable scalar, `self.timestep = nn.Parameter(...)`, `t_init = 261`, `learn_timestep: True`.
  261 is DIFT's empirically optimal timestep for semantic correspondence — so CleanDIFT distils
  *from* the full range but *operates* at the good part of it. **D2 departs from this**, using a free
  256-d embedding instead of a learned scalar.
- **Cost**: full fine-tune (LoRA r=64 only for SDXL/Flux), batch 8, **400 steps, ~3k unlabelled
  images, ~30 min on one A100**. lr: the paper says 2e-6, both shipped configs say **1e-5** with
  `constant_with_warmup`, 2000 warmup steps — prefer the config.
- **Ablations**: cosine > L1 > L2; FiLM > AdaRMS; task-specific training data gives no gain over
  generic data.

**Why the teacher/student asymmetry means there is no `t` downstream:**

| | input | timestep |
|---|---|---|
| **teacher** (frozen) | noisy `x_t` | sampled, stratified, 3 bins over `t_min=1` .. `t_max=999` |
| **student** (trained) | clean `x_0` | a single fixed/learned constant, present only because the UNet signature demands one |

`t` enters the *loss* solely through the FiLM-conditioned heads. At inference the heads are
discarded and the student's internal activations are read directly, so downstream there is exactly
one feature set and no `t` to choose. Note the reference's claim is stronger than "as good as the
best `t`": the distilled student reportedly beats the teacher's best single timestep *and* beats
noise-ensembling, because a merge over the trajectory is not a point on it.

**CryoFM2 specifics that make the mapping clean:** `resnet_time_scale_shift="scale_shift"`, i.e. the
conditioning is already FiLM-shaped (`Linear(256, 2C)` → `h*(1+scale)+shift`), so the heads' FiLM is
architecturally native; `time_embed_dim = 256`; 168.089 M params; `in_channels=2` (ch1 zeros),
`out_channels=1`.

---

# Appendix B — Pitfalls (P1–P10), with current status

**P1 — The premise is untested here.** *RESOLVED, then reframed.* The original gate passed, but the
2026-08-30 high-t sweep changed what "passing" means: the coupled arm peaks at t≈500 and collapses
by t=900, while decoupled saturates and already beats coupled at high t. Hence the success gate at
the top of this document is "beat best **decoupled**", not "beat the teacher".

**P2 — An inconsistent `(x_t, t)` pair.** *RESOLVED.* `cryofm_tap` takes `timestep` and
`noise_level` as independent arguments and mixes noise as `(1 − nl)·x + nl·eps`, which is exactly the
FM interpolant, so `noise_level` **is** `t/1000`. An early probe fed `x_100` while declaring `t=10`.
Any sweep must tie them (`noise_level = timestep/1000`) or measure garbage. D6 uses
`FMScheduler.add_noise` to make this structural.

**P3 — Distillation cannot create invariance the teacher lacks.** **LIVE.** A student trained to
match the teacher inherits its non-equivariance. Strong counter-evidence already in hand: CryoFM2
pretrains with `RotCube24` at `p=1.0` and *still* only reaches 0.094 global rotation consistency at
`up_blocks[0]` — augmentation buys task equivariance, not feature invariance. This is the argument
for arm B (D9), not a reason not to proceed.

**P4 — Screen the student before spending anything downstream, on the right thing.** The cheap
screen is the homolog residue diagnostic (`--rungs local frameavg`): a student should raise `local`
above 0.447 and, more importantly, the model-free `frameavg` rung above 0.255. Two cautions: (a) that
diagnostic measures how smoothly a descriptor tracks sequence similarity — a proxy, not
informativeness and not a task, so it is a screen for *killing* options, not declaring a win; (b) it
is also the metric that overrated the hand-crafted descriptors removed on 2026-08-26.

**P5 — Per-tap loss imbalance is a real 3D-specific footgun.** Token counts differ by 64× across
taps. Summing raw per-tap cosine terms lets near-output taps dominate the gradient — and those taps
behave *qualitatively differently* (global invariance is U-shaped in depth, 0.094 at `up_blocks[0]`
vs 0.620 at `up_blocks[3]`). **Addressed by D5** (mean over cells) and D1 (distil only taps we use).

**P6 — Cosine axis and centring.** **Addressed by D5.** The reference uses
`F.cosine_similarity(..., dim=-1)` on channels-last tensors; this codebase is channels-first
`[B, C, D, H, W]`, so the correct axis is `dim=1`. Getting it wrong silently optimises a
spatial-pattern similarity. Separately, every metric in this project is **centred** cosine, because
raw cosine on high-dim activations is ~0.99 for any pair.

**P7 — The two-channel input.** **Addressed by D7.** `in_channels=2` with ch1 all zeros; the
student's `conv_in` ch1 weights receive gradient from a constant-zero input — meaningless but not
harmless under weight decay. Freeze that slice.

**P8 — Evaluation traps this project has already paid for.** **LIVE — governs the gate.** Any
student number must carry: the mismatched/between-map floor (never an absolute cosine); the
random-weight control (an untrained UNet already reaches pooled top-1 0.58–0.68 against chance
0.10); generic SO(3), never cube rotations (they are CryoFM2's own augmentation group and flatter
results ~2×); and the trivial `[N, Rg, composition]` partial. And one specific to this setup:
**feature agreement with the teacher is circular** — a distilled student is trained to score well on
it. Judge only on a downstream or ceiling metric.

**P9 — Head discarding breaks target compatibility.** **LIVE.** Heads are discarded at inference, so
student features are **not** in the teacher's units. The 1,500 cached chains in
`data/density_targets/` are teacher features at t=10; a student cannot be mixed into that
regression. Re-extract, keeping the same cluster-level split (`data/alignment_chains.csv`, 672
mmseqs clusters at 30%) so numbers stay comparable.

**P10 — Licensing.** **LIVE.** The `CompVis/cleandift` repo tree contains only `README.md`,
`requirements.txt`, `train.py`, `configs/`, `src/`, `notebooks/`, `docs/` — **no LICENSE file**, and
the GitHub license API returns 404. Treat it as all-rights-reserved and **reimplement from the
paper** rather than vendoring code. CryoFM itself is Apache-2.0, so the backbone side is clear.
