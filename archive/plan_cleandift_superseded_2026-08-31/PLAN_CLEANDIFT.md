# CleanDIFT applied to CryoFM2 — feasibility, design, and pitfalls

Written 2026-08-26. Analysis only; nothing has been run. Everything about CleanDIFT below
was read from the CVPR 2025 paper HTML and `CompVis/cleandift/src/sd_feature_extraction.py`;
everything about CryoFM2 was read from `third_party/cryofm` in this checkout, not from memory.

---

## 1. What CleanDIFT actually does

Stracke et al., *CleanDIFT: Diffusion Features without Noise*, CVPR 2025 (oral).

Diffusion-feature extraction (DIFT and successors) needs the input **noised to a timestep t**
before the backbone produces semantically useful activations; that costs information, forces a
per-task choice of t, and makes features stochastic. CleanDIFT removes all three with a
lightweight unsupervised distillation:

- **Teacher**: the frozen original diffusion model, fed the *noisy* `x_t` with timestep `t`.
- **Student**: a trainable *copy of the same backbone*, fed the *clean* `x_0`.
- **Objective** (K feature maps, cosine similarity):

      L = - sum_{k=1..K} sim( proj^(k)( feat_c^(k)(x_0) ; t ),  feat^(k)(x_t ; t) )

- **Projection heads** `proj^(k)`: small FFNs, **zero-initialised**, conditioned on `t` through
  **FiLM**. They are what makes it work: one clean student representation must explain the
  teacher's features at *every* t, so the student is pushed to a t-agnostic superset while the
  head handles the t-specific part. Heads are **discarded at inference** (+0.24 PCK in ablation).
- **t sampling**: stratified, `num_t_stratification_bins = 3`, over `t_min = 1` to
  `t_max = 999` — i.e. **essentially the whole trajectory; only t = 0 is excluded**. Checked in
  `StableFeatureAligner.__init__` defaults and both shipped configs.
- The student still needs *some* timestep argument (architectural requirement). The reference
  implementation passes a learnable scalar, `self.timestep = nn.Parameter(...)`, with
  **`t_init = 261`** and `learn_timestep: True`. 261 is DIFT's empirically optimal timestep for
  semantic correspondence — so CleanDIFT distils *from* the full range but *operates* at the good
  part of it. That distinction matters for us (§3).
- **Cost**: full fine-tune (LoRA r=64 only for SDXL/Flux), batch 8, **400 steps, ~3k unlabelled
  images, ~30 min on one A100**. lr: the paper says 2e-6, both shipped configs say **1e-5** with
  `constant_with_warmup`, 2000 warmup steps — prefer the config.
- Ablations: cosine > L1 > L2; FiLM > AdaRMS; task-specific training data gives no gain over
  generic data.

---

### 1.1 How a timestep is chosen downstream: it isn't

Worth stating explicitly, because it is the usual first question and the answer is structural.

The teacher/student asymmetry is the whole mechanism:

| | input | timestep |
|---|---|---|
| **teacher** (frozen) | noisy `x_t` | sampled, stratified, 3 bins over `t_min=1` .. `t_max=999` |
| **student** (trained) | clean `x_0` | a single **fixed, optionally learned** constant (`self.timestep`, flag `learn_timestep`) — present only because the UNet signature demands one |

`t` enters the *loss* solely through the projection heads `proj^(k)(·; t)`, which are FiLM-
conditioned on the teacher's sampled `t`. The head absorbs the t-specific part; the student trunk
is forced toward a representation that can explain the teacher at **every** t at once. At
inference the heads are discarded and the student's internal activations are read directly.

So downstream there is exactly one feature set and no `t` to choose — "timestep-independent" in
the title is literal. Two consequences:

- **Keeping the heads reinstates the hyperparameter.** `proj^(k)` is a function of `t`, so a
  head-attached feature is a t-conditioned feature and you are choosing again. Discard them.
- **The claim is stronger than "as good as the best t".** The distilled student reportedly beats
  the teacher's best single timestep *and* beats noise-ensembling, because a merge over the
  trajectory is not a point on it. This is what weakens the gate in §4.1 from "find `t*`" to
  "is the trajectory informative at all".

## 2. Why the mapping onto CryoFM2 is unusually clean

Verified against source, not inferred:

| CleanDIFT assumption | CryoFM2 reality | verdict |
|---|---|---|
| backbone is a t-conditioned denoiser | `UNet3DModel(sample_size=64, in_channels=2, out_channels=1)`, v-prediction flow matching | ✅ |
| a well-defined noising operator | `FMScheduler.add_noise`: `x_t = (t/1000)·eps + (1 − t/1000)·x_0`, T = 1000 | ✅ exactly linear, so "noise level" ≡ `t/1000` |
| t enters via FiLM-style modulation | `resnet_time_scale_shift="scale_shift"` in the pretrain config | ✅ the head design is architecturally native; reuse `time_proj`/`time_embedding` |
| K feature maps at mid + decoder blocks | `mid_block`, `up_blocks[0..3]` already hooked in `teachers/cryofm_tap.py` | ✅ |
| a few thousand unlabelled samples | 7,361 Cryo2StructData maps + 1,146 local EMDB maps | ✅ |
| text/caption conditioning | **none** — CryoFM2 is unconditional, ch1 is zeros | ✅ *simpler*: CleanDIFT's "prompt still influences features" limitation disappears |
| small enough to fine-tune | 168.1 M params, 64³ patches, CryoFM2's own pretrain ran batch 12 on 1 GPU, bf16-mixed | ✅ fits on one H100 with the teacher co-resident at reduced batch |

There is also a **train/eval distribution match** available for free: the per-residue pipeline
(`probes/local_frame_stability.centre_features`) already forwards single 64³ boxes with no
tiling, which is exactly the unit CryoFM2 was pretrained on and exactly what a student would be
distilled on.

---

## 3. The finding that motivates this at all

**Every CryoFM number in `CLAUDE.md` was produced at one never-swept operating point.**
`teachers/cryofm_tap.feature_volumes` defaults to `timestep=10, noise_level=0.0`, and a grep over
`probes/` shows every probe uses that default (`--timestep` default 10 in `stability.py`; the
same in `extract_density_targets.py`). So the whole Phase 0 corpus — pose invariance, the homolog
ladder, the ESM-C alignment R² = 0.187 — is CryoFM at `t = 10`, i.e. the extreme *low*-noise end,
which in the diffusion-features literature is the regime with the *least* semantic content.

That is a genuine, cheap, unexplored axis, and it is precisely the axis CleanDIFT operates on.

---

### 3.1 …and t = 10 is below the bottom of the entire range the literature uses

Schedules are not comparable by index. SD is DDPM with a `scaled_linear` beta schedule,
`x_t = sqrt(abar_t)·x_0 + sqrt(1-abar_t)·eps`; CryoFM2 is flow matching with a *linear*
interpolant, `x_t = (1 - t/1000)·x_0 + (t/1000)·eps`. Matching by signal:noise ratio:

| SD t | sqrt(abar) : sqrt(1-abar) | FM-equivalent t | noise |
|---|---|---|---|
| 0 | 1.000 : 0.029 | **~28** | 3% |
| 100 | 0.946 : 0.325 | ~256 | 26% |
| **261** (DIFT optimum, CleanDIFT `t_init`) | 0.810 : 0.587 | **~420** | 42% |
| 500 | 0.526 : 0.851 | ~618 | 62% |
| 999 | 0.068 : 0.998 | ~936 | 94% |

**Our `timestep=10` is 1% noise — below SD's t = 0 (~3%).** Every diffusion-features result in the
literature sits at or above that floor, and the reported optimum for semantic correspondence is
~200–300 in SD units, i.e. **FM t ≈ 350–450** here. We are two orders of magnitude off it in noise
fraction and have never checked.

Corollary for the sweep grid: choose it in *noise fraction*, not by copying 261. FM
`{10, 100, 250, 500, 750}` = 1 / 10 / 25 / 50 / 75 % noise, which brackets the literature optimum;
add 420 if a finer read near the expected peak is wanted.

### 3.2 Two senses of "low-noise instability" — do not merge them

- **Feature quality**: low t is *known bad*. DIFT reports performance suboptimal at very small t
  and above ~300 (SD units), peaking ~200–261. At low noise the denoising task is near-trivial and
  the network's work shifts to fine detail, so activations are appearance/high-frequency rather
  than semantic. This is the sense in which our operating point is suspect.
- **Draw-to-draw reproducibility**: low t is the *best* case, not the worst. Less of the feature
  can depend on a specific `eps` realisation when `eps` barely enters. Phase 0a's `noise_draw`
  0.97–0.996 was measured at t = 10 and should be read as an upper bound that will degrade as t
  rises — not as evidence that the operating point is fine.

Conflating these two would repeat the "variance ≠ usefulness" error already logged twice in this
project.

### 3.3 Hypothesis worth testing in the same sweep: is our pose problem a low-t artifact?

Untested, mechanistically plausible, and cheap to check alongside the R² sweep. At low t the
features are close to a passthrough of the exact voxel realisation, and high-frequency content is
precisely what changes under rotation plus independent re-simulation. That is a candidate partial
explanation for `up_blocks[0]`'s **0.094** global rotation consistency — the project's headline
bottleneck — being a badly-chosen hyperparameter rather than an intrinsic property of the tap.

It could equally go the other way: at high t the signal itself is weaker, so consistency may fall
for a different reason. Genuinely unknown, which is why it should be measured, not argued.

`probes/pose_invariance_clean.py` already has `--timestep`. **`probes/layer_rotation_invariance.py`
does not** — it silently uses the tap default of 10, so the whole per-layer U-shape table in
`CLAUDE.md` is a t = 10 slice. Adding the argument is a one-line change and should be done before
the sweep.

## 4. Proposed design

### 4.1 The gate (do this first, no training required)

Do **not** build the distillation before establishing that a better `t` exists. Sweep it —
the grid is chosen by noise fraction (§3.1), bracketing the literature optimum at FM t ~ 350-450:

    for T in 10 100 250 420 500 750; do
      EXTRA="--timestep $T --limit 300 --outdir data/density_targets_t$T" \
        sbatch slurm/align_targets.slurm
    done
    # then, per t, the ridge probe:
    pixi run python probes/align_esmc_density.py \
      --target-dir data/density_targets_t$T --out results/align_t$T.json

Sweep **two** quantities per t, not one. The second is nearly free and tests §3.3:

    # downstream usefulness
    probes/align_esmc_density.py            -> R2(t)
    # pose stability -- does the 0.094 bottleneck move with t?
    probes/pose_invariance_clean.py --timestep $T   -> pose consistency(t)

`pose_invariance_clean.py` already took `--timestep`; `layer_rotation_invariance.py` did not and
**was given one on 2026-08-26** (it had hardcoded `10` in two `centre_features` calls and used the
tap default in two `feature_volumes` calls). The per-layer U-shape table in `CLAUDE.md` is a
t = 10 slice and should be re-read at the swept optimum.

Cost anchor: the full 1500-chain extraction took **3 h 39 m on one H100** (job 3384414), so
300 chains ≈ 45 min; five t values in parallel ≈ under an hour wall-clock. The ESM-C side is
already cached (job 3391294, 7 min) and the ridge probe is minutes.

**Decision rule — note this is weaker than "find `t*`".** CleanDIFT does not *select* a timestep;
it *merges* the trajectory (see §1.1), and its headline claim is that the merged student beats the
teacher's best single t and beats noise-ensembling. So a flat sweep does **not** by itself kill the
idea — consolidation can still add something a point estimate cannot.

- `R²(t)` peaks well above `R²(10)` → clear headroom; proceed.
- `R²(t)` flat but **healthy** → point selection is worthless, consolidation is still plausible;
  proceed only if you accept a speculative run.
- `R²(t)` flat **and low across the whole trajectory** → the backbone's features barely respond to
  the operating point, so there is little to consolidate. Drop the line.
- **Independently**: if pose consistency rises materially with t, that is a result in its own
  right regardless of the R² curve — it would mean the project's headline bottleneck is partly a
  hyperparameter, and it would change the target for the frame-averaging work too.

The sweep is cheap enough to be worth running for the shape of the curve alone, which nothing in
this project currently knows.

Consider also sweeping the homolog-residue diagnostic (`probes/homolog_diagnostic_residue.py`) at
the same t values — that measures the *ceiling*, which per this project's own methodological rule
is a better gate than an absolute threshold.

### 4.2 The distillation, if the gate passes

- **Student**: `load_cryofm2(...)` a second time, trainable; **teacher**: the existing frozen one.
- **Input**: random 64³ crops from Cryo2StructData maps through `preprocess()`, plus CryoFM2's own
  `RotCube24(p=1.0)` augmentation to stay on its pretraining distribution.
- **Teacher forward**: `x_t = FMScheduler.add_noise(x_0, eps, t)` with `t` stratified in 3 bins
  over `[t_min, 1000]`. **Keep `t_min` low (the reference uses 1)** unless the sweep says
  otherwise: the FiLM head absorbs the low-t bin, so including it costs little, and unlike 2D
  semantic correspondence our per-residue target genuinely may want fine detail — `up_blocks[0]`
  has a ~10 Å half-decay, i.e. the contact scale. Raising `t_min` is a legitimate cheap ablation,
  not a default.
- **Student forward**: clean `x_0`, timestep = a single learned scalar (`learn_timestep`).
  Initialise at the sweep's best t, **not** at 10 — the reference default is `t_init = 261` in SD
  units, i.e. **FM t ≈ 420** here (§3.1). Initialising at 10 would anchor the student to the
  operating point we suspect is bad.
- **Heads**: per tap, a zero-init 1×1×1-conv FFN with FiLM from the model's own time embedding —
  the 3D analogue of CleanDIFT's adapter.
- **Loss**: `-cosine(proj_k(student_k), teacher_k.detach())` **over `dim=1`** (see P7), averaged
  over spatial cells, summed over taps with **per-tap token-count normalisation** (see P6).
- **Optimiser**: **lr 1e-5** with `constant_with_warmup` — this is CleanDIFT's shipped config
  value and it also sits an order of magnitude under CryoFM2's own pretrain lr of 1e-4, which is
  the right relationship for a light distillation of a 168 M model. bf16-mixed, batch 4–8 with the
  teacher resident, 1–4k steps. Sub-hour job.
- **Integration**: add a `student_ckpt` argument to `CryoFM2Tap`; nothing downstream changes.

### 4.3 The variant that is actually worth more than plain CleanDIFT

Because distillation lets you *choose the target*, make the target the **frame-averaged** teacher
feature — the mean over the 24 cube rotations that `CryoFM2Tap.sample_frame_averaged` already
computes. That buys 24-frame averaging at **1× inference cost** instead of 24×.

The numbers in `CLAUDE.md` say this is where the value is: at `up_blocks[0]`, global rotation
invariance is **0.094** single-frame vs **0.634** frame-averaged, with per-residue top-1 going
0.030 → 0.130. Pose, not noise, is this project's measured bottleneck — so a distillation run that
attacks noise/timestep *and* pose in the same objective dominates one that attacks noise alone.

---

## 5. Pitfalls, in order of how likely each is to kill it

**P1 — The premise is untested here.** CleanDIFT's value is consolidating good *high-t* features
into a clean pass. Nobody has shown CryoFM2 has better features at high t for our targets. The
existing evidence (`noise_shift` 0.18–0.58) shows only that features *move* with the operating
point, not that they *improve* — the same "variance ≠ usefulness" confusion that already cost this
project a wrong conclusion in the pyramid-band work. **Hence the gate in §4.1.**

**P2 — The existing noise probe fed an inconsistent `(x_t, t)` pair.** `cryofm_tap` takes
`timestep` and `noise_level` as *independent* arguments and mixes noise as
`(1 − nl)·x + nl·eps` — which is exactly the FM interpolant, so `noise_level` **is** `t/1000`.
But `stability.py` used `--noise-level 0.1` with `--timestep 10`, i.e. it fed `x_100` and told the
model `t = 10`. Off-manifold. Any sweep must tie them (`noise_level = timestep/1000`) or it will
measure garbage and conclude "high t is bad". Worth making the tap derive one from the other and
raise on a mismatch.

**P3 — Distillation cannot create invariance the teacher lacks.** A student trained to match the
teacher inherits its non-equivariance. And note the strong counter-evidence already in hand:
CryoFM2 pretrains with `RotCube24` at `p=1.0` and *still* only reaches 0.094 global rotation
consistency at `up_blocks[0]` — augmentation demonstrably buys task equivariance, not feature
invariance. So a naive CleanDIFT student fixes nothing on the axis that matters. This is the
argument for §4.3 rather than a reason not to proceed.

**P4 — Screen the student before spending anything downstream, but screen it on the right thing.**
The cheap screen is the homolog residue diagnostic (`--rungs local frameavg`): a CleanDIFT student
should raise `local` above 0.447 and, more importantly, raise the model-free `frameavg` rung above
0.255. Two cautions. (a) That diagnostic measures how smoothly a descriptor tracks sequence
similarity — a proxy, not informativeness and not a task; it is a screen for *killing* bad options,
not for declaring a win. (b) It is also the metric that overrated the hand-crafted descriptors
removed on 2026-08-26, so do not let a good score on it substitute for the downstream
`align_esmc_density` number.

**P5 — Per-tap loss imbalance is a real 3D-specific footgun.** Token counts differ by 64×
across taps (`up_blocks[0]` 4,096 vs `up_blocks[2,3]` 262,144 each). Summing raw per-tap cosine
terms lets the two near-output taps dominate the gradient — and per
`results/layer_rotation_invariance.json` those taps behave *qualitatively differently* (global
invariance is U-shaped in depth, 0.094 at `up_blocks[0]` vs 0.620 at `up_blocks[3]`). Normalise
per tap, or distil only the taps you will actually use.

**P6 — Cosine axis and centring.** The reference implementation uses
`F.cosine_similarity(..., dim=-1)` on channels-last tensors; this codebase is **channels-first**
`[B, C, D, H, W]` (see `centre_features`), so the correct axis is `dim=1`. Getting it wrong
silently optimises a spatial-pattern similarity instead of a feature similarity. Separately: every
metric in this project is **centred** cosine, because raw cosine on high-dim activations is ~0.99
for any pair. An uncentred distillation loss will sit near 1 and put its gradient into the shared
common component. Centre the loss, or at minimum log both.

**P7 — The two-channel input.** `in_channels=2` with ch1 all zeros. The student's `conv_in`
weights on ch1 receive gradient from a constant-zero input — meaningless but not harmless if the
optimiser has weight decay. Freeze that slice.

**P8 — Evaluation traps this project has already paid for.** Any student number must carry:
the **mismatched/between-map floor** (never an absolute cosine); the **random-weight control**
(an untrained UNet already reaches pooled top-1 0.58–0.68 against chance 0.10); **generic SO(3),
never cube rotations** (they are CryoFM2's own augmentation group and flatter results ~2×); and
the **trivial `[N, Rg, composition]` partial**. And one new to this setup: *feature agreement with
the teacher is circular* — a distilled student is trained to score well on it. Judge only on a
downstream or ceiling metric.

**P9 — Head discarding breaks target compatibility.** CleanDIFT discards `proj^(k)` at inference,
which is fine when you want a good representation but means student features are **not** in the
teacher's units. The 1500 cached chains in `data/density_targets/` are teacher features at t=10;
a student cannot be mixed into that regression. Re-extract, and keep the same cluster-level split
(`data/alignment_chains.csv`, 672 mmseqs clusters at 30%) so numbers stay comparable.

**P10 — Licensing.** The `CompVis/cleandift` repo tree contains only `README.md`,
`requirements.txt`, `train.py`, `configs/`, `src/`, `notebooks/`, `docs/` — **no LICENSE file**,
and the GitHub license API returns 404. Treat it as all-rights-reserved and **reimplement from the
paper** rather than vendoring code. CryoFM itself is Apache-2.0, so the backbone side is clear.

---

## 6. Bottom line

CleanDIFT is an unusually clean architectural fit for CryoFM2 — same denoiser structure, linear
interpolant, FiLM-native time conditioning, unconditional (so the text machinery drops out), tiny
model, abundant unlabelled maps, and a sub-hour training cost. The engineering risk is low.

The **scientific** risk is that it targets a problem this project has not shown is binding —
though §3.1 sharpened this: our `timestep=10` is 1% noise, *below the bottom of SD's entire
schedule*, and roughly 40x lower in noise fraction than the literature's optimum. That is no
longer "an unexplored axis"; it is a specific reason to expect the current operating point is
wrong. The documented bottleneck is rotation non-equivariance — 0.094 at the working tap, and the model-free
`frameavg` rung retaining only 57% of the atomic-model-dependent `local` rung's signal. Noise and
timestep have never been shown to cost anything here, because they were never varied.

Recommended order:
1. **t-sweep gate** (§4.1) — under a day, no training. Kill or proceed on the result.
2. If it proceeds, build the distillation with the **frame-averaged teacher** (§4.3), not plain
   CleanDIFT, so the run attacks the measured bottleneck as well as the hypothesised one.
3. Screen the student with the **homolog-residue diagnostic** (`local` and `frameavg` rungs)
   before any downstream alignment work — as a kill-screen, not as the success metric.
4. Only then re-run `probes/align_esmc_density.py`.

---

## 7. SWEEP RESULTS (2026-08-27) — the gate passed, and the headroom is now a number

All results below are `up_blocks[0]` unless stated. `t=10` reproduces every previously logged value
exactly (pose 0.826 / 0.716 / 0.677 across the three taps; relational table within sampling noise),
so these are clean one-variable sweeps rather than a changed pipeline.

### 7.1 The pairing is a first-class variable, and the earlier arm was the wrong one alone

Pitfall P2 (§5) said any sweep must tie `noise_level` to `t`. The first launch did not — every probe
routes through `centre_features`, which forwarded a CLEAN box with `timestep=t`. Both arms were then
run. They have **opposite shapes**, so the pairing is not a detail:

| t | noise | decoupled pose | **coupled pose** | decoupled top1 | **coupled top1** |
|---|---|---|---|---|---|
| 10 | 1% | 0.826 | 0.826 | 0.819 | 0.821 |
| 100 | 10% | 0.860 | 0.864 | 0.846 | 0.858 |
| 250 | 25% | 0.905 | **0.902** | 0.862 | 0.907 |
| 420 | 42% | 0.924 | 0.900 | 0.859 | 0.917 |
| 500 | 50% | 0.928 | 0.892 | 0.857 | 0.918 |
| 750 | 75% | **0.937** | 0.851 | 0.853 | **0.924** |

- Decoupled pose rises monotonically; **coupled pose peaks at t~250 and falls to 0.851.** The
  monotone rise is an off-manifold artifact — clean input at a high conditioning signal gets
  steadily more degenerate, and its apparent stability keeps climbing while the on-manifold reality
  turns over.
- **NOT the smoothing artifact** that was hypothesised. The `spatial` guard (Spearman of feature
  similarity vs 3D distance) moves only 0.545 -> 0.526 at `up_blocks[0]` and 0.600 -> 0.518 at
  `mid_block`, i.e. 1-14%, against a 59% cut in pose residual error. Coupled top1 also *rises*,
  which smoothing would not do.
- `up_blocks[1]`'s decoupled degradation (pose 0.677 -> 0.642, p10 to -0.088) **does not survive the
  correct pairing** (0.675 -> 0.683, top1 0.528 -> 0.619, p10 -0.088 -> +0.128). It was an
  off-manifold artifact. Two mechanisms were proposed for it (smoothing; effective-rank collapse);
  both were explanations for a number produced by a broken configuration.

### 7.2 The operating point is ARM-SPECIFIC — and no cross-arm comparison is licensed

**⚠ CORRECTION (2026-08-27).** An earlier version of this section recommended t~420 by taking the
DECOUPLED fraction curve (which peaks at 750) and rejecting 750 because COUPLED pose collapses
there. That is invalid: §7.1's own finding is that coupled and decoupled features at MATCHED t are
nearly unrelated (median centred cosine between the two arms' targets 0.003 / -0.145 / 0.043 at
t=250 and -0.198 / -0.439 / -0.110 at t=750). A metric measured on one arm cannot discount a number
measured on the other. "t~420" was the answer to neither arm's question; it was an average of two.

Which arm each measurement belongs to (this is the part that was being conflated):

| measurement | probe | arm |
|---|---|---|
| pose / top1 (both columns of §7.1) | `pose_invariance_clean.py --couple` | **both**, run separately |
| captured fraction, R2, ceiling (§7.3 table) | `extract_density_targets.py` (no `--couple`) | **DECOUPLED** |
| relational Procrustes, CKA, RSA, kNN, `spatial` | `layer_relational_invariance.py` (no `--couple`) | **DECOUPLED** |

**DECOUPLED arm, judged only against itself** — this is the arm every fraction number comes from:
- captured fraction monotone up, peaks **t=750** (0.421)
- decoupled pose monotone up (0.826 -> 0.937)
- *against* 750: `spatial` erodes (`up_blocks[0]` 0.545 -> 0.526, -3.5%; `mid_block` 0.600 -> 0.518,
  -14%) and decoupled top1 peaks at 250 (0.862) then drifts to 0.853.
- **Net for a regression objective whose metric is the captured fraction: t=750**, at a cost of
  ~3.5% geometric fidelity at `up_blocks[0]` and ~1% top1. Nothing in this arm argues for 420.

**COUPLED arm, judged only against itself:**
- pose peaks **t~250** (0.902) then falls to 0.851
- top1 rises monotonically to 0.924
- captured fraction measured at **only t=250 and t=750** (two coupled extractions).

**So if the coupled arm wins, the choice is between two measured points, not an optimisation over
the range.** That is defensible — two honest coupled points beat six transferred ones — but the
decoupled curve's shape licenses no inference about where the coupled optimum sits, and the write-up
must say so rather than implying a swept optimum.

### 7.3 THE CLEANDIFT HEADROOM, quantified

A CleanDIFT student consumes clean input at one fixed timestep = the **decoupled** arm; the teacher
is sampled **coupled**. So the coupled-minus-decoupled top1 gap is exactly what a distillation
exists to recover:

| t | 10 | 100 | 250 | 420 | 500 | 750 |
|---|---|---|---|---|---|---|
| gap | **+0.001** | +0.012 | +0.045 | **+0.058** | +0.061 | +0.071 |

**The gap is zero at our current operating point and grows with t.** This is the first quantitative
argument for CleanDIFT here rather than an argument from analogy, and it also explains why the idea
would have been pointless at t=10: there was nothing to distil because clean input at t~0 already
*is* the on-manifold input. Lead with this number.

### 7.4 Relational invariance is FLAT in t — the frame-free route is not a hyperparameter problem

Held-out Procrustes, global framing, change t=10 -> 750: `down_blocks[1]` -0.004, `down_blocks[2]`
-0.007, `down_blocks[3]` -0.015, `mid_block` -0.018, `up_blocks[0]` **+0.019**. Nothing moves.

So `t` buys robustness to re-discretization *inside* an already-canonical frame and buys **nothing**
for the frame-free route. Dropping the atomic-model dependency requires a relational objective.

- **Level caveat**: measured at `n_res=60`, where held-out Procrustes badly UNDERestimates
  (an independent saturation sweep gives `up_blocks[0]` 0.564 at n=60 -> 0.742 at n=400, with
  in-sample falling 0.866 -> 0.798 from the other side). Quote the converged values
  (~0.75 `up_blocks[0]`, ~0.81 `mid_block`, ~0.83 `down_blocks[2]`), not these.
- **The flatness claim survives that**, because a bias constant in t is a level shift that cancels
  out of a comparison at fixed `n_res` and fixed population. The one way it fails is if effective
  rank varies with t; a robustness run at `n_res=240` for t=10/420/750 is testing exactly that.
- **Any Procrustes number needs `(n_res, population)` attached** — the same `n_res=120` gives 0.820
  on chains 80-400 and 0.659 on chains >=400, since more of a large chain falls outside the 96 A box.
- **Two objectives, two taps, neither wrong**: `up_blocks[0]` for vector regression (coupled pose
  0.90, top1 0.907-0.917, ~10 A half-decay = contact scale); `mid_block`/`down_blocks[2]` for a
  relational objective (0.81-0.83 converged vs 0.75).

### 7.5 `up_blocks[3]` collapses with t — a retracted claim, now doubly retracted

Held-out Procrustes 0.618 -> **0.261**, strict cos 0.645 -> **0.280**, CKA 0.417 -> 0.257: the
largest movement in the table, monotone. Its pointwise stability at t=10 was a low-t artifact —
near t=0 the task is near-identity and late decoder layers approach the co-rotating velocity output;
that stops holding once the model must infer structure from noise. `CLAUDE.md` retracted this tap as
a frame-free candidate on relational grounds; it is now retracted on a second independent axis.
The predicted depth/t interaction is therefore CONFIRMED as a mechanism, but it produces **no new
(layer, t) winner** — the relational ranking is stable at every t, which makes tap choice and
timestep choice separable decisions.

### 7.6 Controls that passed

- `conv_in` is byte-identical across all six timesteps on every metric (0.708 / 0.304 / 0.409 /
  0.096). Architecturally required — the timestep embedding enters in the resnet blocks, after
  `conv_in` — so it confirms the plumbing and that nothing else varied between runs.
- Mismatched floor stays pinned near zero (-0.018 to -0.026 at `up_blocks[0]`) while separation
  (matched - mismatched) GROWS 0.794 -> 0.895. Not degeneracy.
- Floors are comparable *within* this pose sweep (retrieval runs per structure at N~250, constant
  across points) but NOT against runs at other `n_res`; and the reported statistic is a median of
  cosines, which no sum-to-zero centring identity constrains.

### 7.7 Still open

Ceiling sweep (sequence-trackable fraction vs t) and stage 2 (downstream R2 vs t) are running.
**Report the captured FRACTION (R2 / sequence-tracked) as its own column**: if R2 and the ceiling
rise together the fraction is flat and a t-change buys the alignment nothing even though every
individual number improved.

### 7.8 COUPLED-vs-DECOUPLED TARGETS (2026-08-27) — better features, worse targets

Coupled targets extracted at t=250/750 (`extract_density_targets.py --couple`), same 300 chains and
cluster split as §7.3:

| t | arm | shuf cos | seqwin3 | esmc | **esmc+mlp** |
|---|---|---|---|---|---|
| 250 | decoupled | 0.0117 | 0.0400 | 0.1700 | **0.1890** |
| 250 | coupled | 0.0097 | 0.0240 | 0.0446 | 0.0702 |
| 750 | decoupled | 0.0073 | 0.0413 | 0.2054 | **0.2304** |
| 750 | coupled | 0.0110 | 0.0198 | 0.0548 | 0.0838 |

**Decoupled beats coupled 2.7x at both t — INVERTING the pose/top1 ordering**, where coupled won
(top1 0.907 vs 0.862 at t=250).

**Mechanism, pinned by the `seqwin3` control.** seqwin3 collapses in lockstep (0.040 -> 0.024,
0.041 -> 0.020), so the coupled target is harder for EVERY sequence-based predictor, not for ESM-C
specifically. `x_t = (1-t/1000)x_0 + (t/1000)eps` and `eps` is not a function of the structure, so a
coupled feature is `f(structure, eps)` while a decoupled feature is `f(structure)`. Both arms measure
something real and neither is wrong: **coupled features are better FEATURES and worse TARGETS**,
simultaneously and consistently.

**DECISION: extract DECOUPLED at t=750** for a regression target — it wants maximum absolute R2 AND
a target that is a deterministic function of the structure. Decoupled wins on both.

**RELATIONAL FLATNESS HOLDS ON THE COUPLED ARM TOO** (`layer_relational_invariance.py --couple`,
t=250/750; baseline is decoupled t=10, valid because the arms coincide there to <=0.0022). Held-out
Procrustes vs decoupled t=10: `down_blocks[2]` -0.024, `down_blocks[3]` -0.013, `mid_block` +0.003,
`up_blocks[0]` +0.030, `up_blocks[1]` -0.033. And `up_blocks[3]` collapses on the coupled arm as well
(0.618 -> 0.552 -> **0.314**). So the frame-free negative is arm-independent, and a relational
objective does not need re-justifying if the pairing changes.

### 7.9 NEXT EXPERIMENT — K-draw noise marginalisation (the cheap test of CleanDIFT's premise)

The coupled arm's quality advantage is not lost, only unreachable by single-draw targets. Predict
`E_eps[f(x_t, t)]` instead of `f(x_t, t)` for one `eps`: that expectation is **deterministic in the
structure AND on-manifold**, which is the combination neither single arm offers.

- **Cheap route, no training**: average coupled features over K noise draws at extraction. Cost is
  the forward, so K=8 is ~8x; 150-300 chains at K=8 is 3-6 h and answers the question. Does not need
  the full set.
- **CleanDIFT route**: a student trained on clean input against a coupled teacher sampled over noise
  **IS a learned noise-marginaliser**. This is now the strongest argument for CleanDIFT in this
  project, ahead of the top1-gap number in §7.3.
- **Either outcome is informative.** Recovers the coupled advantage -> much of what CleanDIFT would
  buy is available for the price of an extraction, which reprices the whole plan. Fails to recover ->
  the advantage was never separable from the noise realisation, which is a real result about what the
  coupled features are.
- **Run the K-draw test BEFORE building the distillation.**

### 7.10 Process lessons from this sweep

- **An assertion protects the FILE, not the WORLD.** A `count==N` guard fired correctly and still
  left two live slurm jobs depending on an unapplied patch, because the submission had already
  escaped. Put the guard at the last irreversible step, not the last edit: verify `--help` shows a
  new flag before launching anything that uses it.
- **`count==N` with an explicit N**, not "exactly one match" — the failure mode is "the number of
  matches was not what I assumed", which cuts both ways. Two independent instances this session: an
  empty-slice `replace("")` that inserted between every character, and a count of 3 asserted where
  the truth was 2. Both were assumptions about a string not looked at first.
- **No cross-arm comparison.** Coupled and decoupled features at matched t are near-orthogonal
  (median centred cosine 0.003 to -0.44), so a metric from one arm cannot discount a number from the
  other. One recommendation in this document was made that way and had to be retracted.
- **Never quote R2 without its ceiling**, and never quote a Procrustes number without
  `(n_res, population)`. Both bit this sweep.
