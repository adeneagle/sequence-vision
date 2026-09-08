# Sequence → cryo-EM density features: literature, plan, feasibility

> **2026-08-10 — canonical orientation resolved.** Whole-map canonicalisation WORKS and is
> model-free: `teachers/whole_map_canonical.py` (second-moment axes + odd-degree SH scoring to
> break the discrete sign/azimuth ambiguity). Independently-discretised volumes correlate
> **0.85–0.90** after canonicalisation vs 0.05–0.14 before; frames agree to 0.3–6° with no sign
> flips. This supersedes the earlier negative, which was about *per-residue local* frames — a
> much harder and different problem (a 10 Å neighbourhood is a generic packed-atom environment;
> structure tensor ~30°, SH dipole ~24° even optimally smoothed, with a synthetic control at 0.0°
> proving the code correct and the data at fault).
>
> **Still fails on symmetric complexes** (0.19–0.64). C_n-equivalent azimuths *should* be harmless
> since the density is identical — but deposited C_n models are not exactly symmetric, so
> equivalent-looking frames give measurably different density. Symmetry is now a first-class
> variable, not an edge case.
>
> Two bugs found en route, both by disbelieving inconsistent numbers: `eigh` can return a
> left-handed basis (all four sign candidates preserve det, so the candidate list came back
> empty); and an m=0-only score is blind to the 180°-about-e1 flip, which an `abs()`-based frame
> metric then hid. See `../CLAUDE.md` for the full account.

## Context

**Goal (as scoped with you):** a *representation-research* study answering — can ESM-C sequence
embeddings be aligned to the internal features of a cryo-EM density foundation model, and what
does the answer reveal? Target modality is **single-particle cryo-EM (≤3 Å EMDB maps with fitted
PDB models)**, not in-situ cryo-ET. Teacher choice is **open**; CryoFM is the starting candidate
but gets validated before anything is built on it.

**Original idea:** extract features from CryoFM, take ESM-C features, train an Evoformer-like
module to align them.

**Why the plan below deviates:** three things surfaced in the literature pass that change the
shape of the work (details in *Feasibility*). Briefly — CryoFM has no encoder, so "CryoFM
features" means tapping a denoiser's activations, which is unvalidated for this model; the
alignment target has a *pose-equivariance* problem that must be measured before it can be
predicted; and there is a cheap, strong baseline (fold → simulate density → run the teacher) that
the learned module must beat or the result is negative. So the plan front-loads measurement and
defers the Evoformer to an ablation rung that only gets built if a cheaper rung is beaten.

**Working directory:** `/mnt/main0/projects/et-foundation-vision-model/aden/esm/sequence_vision/`
(currently empty). Sibling project `../scripts/` has directly reusable training infrastructure.

---

## 1. Sourced literature

### The density side

| Work | What it is | Relevance |
|---|---|---|
| **CryoFM1** — *A Flow-based Foundation Model for Cryo-EM Densities*, Zhou/Li/Yuan/Gu (ByteDance Seed), ICLR 2025. [arXiv:2410.08631](https://arxiv.org/abs/2410.08631) · [OpenReview](https://openreview.net/forum?id=T4sMzjy7fO) | Rectified-flow / conditional-OT flow matching in **voxel space**. Backbone is a **3D HDiT** (hierarchical diffusion transformer), *not* a UNet. CryoFM-S: 64³ @ **1.5 Å/voxel** (96 Å box), patch 4, depths [4,8], widths [768,1536], 335 M params. CryoFM-L: 128³ @ 3.0 Å/voxel, 308 M. 3D axial RoPE, AdaRMSNorm on the timestep. | Primary candidate teacher |
| **CryoFM2** — *A Generative Foundation Model for Cryo-EM Densities*, bioRxiv [10.64898/2025.12.29.696802](https://doi.org/10.64898/2025.12.29.696802) | Plain **3D UNet** (diffusers `UNet2DModel` ported to 3D), 64³ @ 1.5 Å/voxel hard-coded, channels (64,128,256,512), attention in the two deepest blocks, ~168 M. | **Easiest teacher to actually run** (no `natten`) |
| Code / weights | [github.com/ByteDance-Seed/cryofm](https://github.com/ByteDance-Seed/cryofm) (Apache-2.0), [HF cryofm-v1](https://huggingface.co/ByteDance-Seed/cryofm-v1), [HF cryofm-v2](https://huggingface.co/ByteDance-Seed/cryofm-v2) | Fully open. Not on disk here — must be fetched. |
| **CryoLVM** — JEPA on density maps, SCUNet backbone. [arXiv:2602.02620](https://arxiv.org/abs/2602.02620) | An actual **encoder** foundation model for density. | Alternate teacher |
| **CryoSiam** — SimSiam, hierarchical **voxel-level** + subtomogram reps. [bioRxiv 2025.11.11.687379](https://www.biorxiv.org/content/10.1101/2025.11.11.687379v1) · [code](https://github.com/frosinastojanovska/cryosiam) | Encoder, cryo-ET native, voxel-level features. | Alternate teacher |
| **DRACO** — denoising-reconstruction ViT on >270k micrographs, NeurIPS 2024. [arXiv:2410.11373](https://arxiv.org/abs/2410.11373) | 2D counterpart. | Context |

### The sequence side

| Work | Key numbers |
|---|---|
| **ESM-C** ([blog](https://www.evolutionaryscale.ai/blog/esm-cambrian)) | 300M/30L/**960**, 600M/36L/**1152**, 6B/80L/**2560**. RoPE, SwiGLU, no biases. `LogitsConfig(return_hidden_states=True, ith_hidden_layer=-1)` returns all layers. `output_attentions=True` gives per-layer `[B,H,L,L]` but **is incompatible with flash attention**. |
| **ESMFold** (Lin et al., Science 2023, [10.1126/science.ade2574](https://www.science.org/doi/10.1126/science.ade2574)) | Learned **softmax over 37 scalar layer weights** → MLP → `c_s=1024`. Trunk = 48 simplified-Evoformer blocks, `c_z=128`, keeps triangle mult + triangle attention, drops the MSA stack. **The pair rep is zero-initialised** (`use_esm_attn_map=False` in the released checkpoint) — ESM-2 attention maps are *not* used. Folding head ~690 M on frozen ESM-2 3B. |
| **ESMFold2** (Biohub, May 2026, **MIT**) — [github.com/Biohub/esm](https://github.com/Biohub/esm) · [model page](https://biohub.ai/models/esmfold2) · [announcement](https://biohub.org/news/world-model-of-protein-biology/) | Frozen ESMC-6B, all layers → shared LN+Linear to `d_z=256` pair rep → **48 folding layers with no triangle attention** + recurrent weight sharing → diffusion coordinate head. **Trunk+head ≈ 189 M**. 1024 residues in 9.4 s on H100. *This is essentially the architecture you described, already trained and released.* |
| **Pairmixer** — *Triangle Multiplication Is All You Need*, [arXiv:2510.18870](https://arxiv.org/abs/2510.18870) | Drops triangle attention entirely; 4× faster at 2048 tokens, ~30% longer sequences, lDDT parity (0.78 vs 0.78). Ablation: *naive* removal costs 0.74→0.70; a retrained triangle-mult-only stack recovers it. |
| Cost reality | Triangle **attention** materialises an `L×L×L` logit tensor (2.1 GB/head at L=1024, bf16) — this, not triangle multiplication, is the bottleneck. AF3 activation memory is 96% of total. **Expect training crops of 256–384 residues per 80 GB GPU** for a 48-block trunk with triangle attention. |

### Alignment methodology

- **LiT** ([arXiv:2111.07991](https://arxiv.org/abs/2111.07991)) — freeze the strong tower, train only the other. Exactly this setting.
- **AM-RADIO** ([arXiv:2312.06709](https://arxiv.org/abs/2312.06709), [code](https://github.com/NVlabs/RADIO)) — the best-engineered public recipe for dense feature distillation from a frozen teacher: per-teacher adaptor head, cosine + smooth-L1 on **spatial** features.
- **Feature Distillation** ([arXiv:2205.14141](https://arxiv.org/abs/2205.14141)) — **whitening teacher features before regression matters a lot**.
- **SigLIP** ([arXiv:2303.15343](https://arxiv.org/abs/2303.15343)) — sigmoid contrastive; works at small batch, which matters when you have 10³–10⁴ maps, not 10⁸ pairs.
- **Platonic Representation Hypothesis** ([arXiv:2405.07987](https://arxiv.org/abs/2405.07987)) — gives mutual-kNN and CKA as *pre-training* alignability diagnostics. Used in Phase 0 below.

### Nearest prior work (and the gap)

- **ModelAngelo** (Nature 628:450, 2024) — GNN over density with **cross-attention into frozen ESM-1b sequence embeddings**, then HMM search for identification. Proof that PLM + density fuse productively — but via cross-attention inside a task model, not representation alignment.
- **EmbedOpt** ([arXiv:2602.05285](https://arxiv.org/html/2602.05285v2)) — optimises a folding model's **`s` and `z` embeddings at inference** so the structure fits a density map. Direct evidence that a folding trunk's pair rep is the right interface for density conditioning.
- **Boltz-2 test-time supervision on cryo-EM maps** ([arXiv:2605.09832](https://arxiv.org/abs/2605.09832)); **CryoBoltz**; **Cryo2Struct2** (ESM-3B + density, [Comms Chem 2025](https://www.nature.com/articles/s42004-025-01718-5)); **Cryo2StructData** ([Sci Data 2024](https://www.nature.com/articles/s41597-024-03299-9)) — a **pre-curated labelled EMDB↔PDB map dataset**, which is the dataset this project needs.
- **RMSF-net** ([Nat Commun 2024](https://www.nature.com/articles/s41467-024-49858-x)) — per-residue flexibility from map + model. Template for density-derived per-residue scalar targets.

**Gap:** no published work trains a sequence encoder to regress or contrastively match a cryo-EM
density *foundation model's learned features*. The idea sits in an empty cell. The adjacent cells
are all filled, which is both encouraging (the ingredients work) and a warning (people have been
close and gone elsewhere).

---

## 2. Feasibility assessment

### 🔴 Hard problems that must be resolved before building

**1. CryoFM is a denoiser, not an encoder.** There is no encoder, no documented feature API, no
representation benchmark in either paper. "CryoFM features" = activations of `v_Θ(x_t, t)` at a
chosen noise level `t` — the standard diffusion-features trick, but *unvalidated for this model*.
The network is a function of `(x_t, t)`, so you must pick `t` and a noise draw, and the features
inherit that stochasticity. **This is a research question, not an implementation detail.**

**2. Pose non-equivariance.** HDiT/UNet3D features are not rotation-equivariant. The teacher was
trained with random rotation augmentation, so features may be *approximately* rotation-tolerant,
but a sequence model has no pose. If the teacher feature at a residue swings with map orientation,
**no sequence model can predict it and the project is dead on arrival.** This is cheap and decisive
to measure (CryoFM2's own `RotCube24` gives the 24 cube rotations for free) and it is the very
first thing to do.

**3. Is the target reachable analytically?** Sequence → ESMFold2/Boltz structure → simulated
density → run the teacher is a cheap, fully interpretable path to the same target. If it matches
the true teacher features well, a learned sequence→feature module is a compute shortcut at best.

*The structure→density step is **analytic, not learned**.* Density is computed directly from atomic
scattering factors — `gemmi` `DensityCalculator`, `phenix.fmodel`, ChimeraX `molmap`, `e2pdb2mrc`.
No training data or learned mapping is required. Standard practice in the field (DiffModeler and
DeepTracer-ID both build AF2 model libraries and simulate density for map fitting).

*But the comparator is gated on the **simulated↔experimental domain gap**.* Simulated density has no
noise, no CTF residue, no solvent, no reconstruction anisotropy, no unmodeled ligands/partners, and
captures disorder only crudely via B-factors. These differences are **well documented in voxel
space** — it is why DeepEMhancer/EMReady train on experimental pairs and why CryoCCD
([arXiv:2505.23444](https://arxiv.org/html/2505.23444v1)) exists.

**What is *not* documented is whether that gap survives into CryoFM's feature space**, and it does
not follow from the voxel-space result. CryoFM is a denoiser: at low `t` its function is precisely
to map corrupted density onto the clean-signal manifold, so the differences separating simulated
from experimental input are close to the exact nuisance variables it was trained to remove. Its
features may therefore be substantially invariant to them. Open question, cheap to settle —
**Phase 0d** measures it directly, and the comparator's fate follows from that number:

- **Small feature-space gap** → the full fold-then-simulate comparator is valid and becomes the
  single most important experiment in the study.
- **Large feature-space gap** → the comparator is confounded; **drop it**. A weak score would mean
  "the simulation is unrealistic," not "folding is insufficient."

*One form of the comparator survives either way*, and is nearly free once the machinery exists:
**ESMFold2 structure → simulated → features** vs. **deposited structure → simulated → features**.
Both sides sit in the simulated domain, so the gap cancels exactly, isolating **folding error alone**
in teacher-feature space. Well-posed regardless of how 0d lands, and arguably the more interesting
question. Mitigation for the calibrated case either way: lowpass to the deposited map's reported
resolution, apply deposited ADPs, apply the same mask.

**4. Leakage.** This bit the sibling project repeatedly (see `../CLAUDE.md`: the pairwise-contact
probe hit AUC 0.997 purely because band features are coordinate-derived). Teacher features are
computed from a map anchored at residue coordinates, so *any* smoothed neighbourhood quantity is
predictable from geometry alone. Mandatory controls: protein-level splits, mmseqs2 30%-identity
sequence clustering, and a **random/shuffled-embedding control** to establish the leakage floor.

### 🟡 Manageable but real

- ~~**Environment split.**~~ **RESOLVED.** `natten` turns out to be a *soft* import
  (`hdit/model.py:40`, try/except) needed only for CryoFM1's neighborhood attention — not for the
  CryoFM2 UNet path. A dedicated env is still warranted because CryoFM pins `numpy<2.0` vs the main
  env's 2.2.6. Built at `sequence_vision/pixi.toml` (python 3.10, torch 2.13+cu130, numpy 1.26.4).
- **Field of view.** CryoFM-S/CryoFM2 see a 96 Å box. That is a domain, not a protein. Fine for
  per-residue-neighbourhood targets; useless for a whole-protein embedding without tiling+pooling.
- **No GPU on this node.** 96 CPU / 754 GB RAM login node; GPUs via SLURM (`h100-reserved` 300×8,
  `h200-reserved` 22×8, `l40-reserved` 25×8). Reuse `../../slurm/plm_decomp.slurm` as the template.
- **Data volume.** ~3.5k EMDB maps at ≤3 Å with half-maps (CryoFM's own curation). A few hundred GB.
  Disk is a non-issue (913 TB free on `/mnt/main0`).

### 🟢 Genuinely favourable

- Paired data is **clean and abundant** in the SPA regime: EMDB ≤3 Å entries with fitted PDB models
  give exact residue→voxel correspondence, and Cryo2StructData has already curated this.
- All weights are open (CryoFM Apache-2.0; ESM-C 6B / ESMFold2 **MIT** as of the 2026 Biohub
  release — this removes the Cambrian non-commercial constraint that applied to ESM-C 600M).
- The sibling project's training scaffolding is directly reusable and battle-tested (see §4).
- Substantial compute available and lightly used.

### Verdict

**Feasible as a research study; not yet justified as a model-building project.** The scientific
question is well-posed and the data exists. But the value of the *learned module* rests entirely on
beating fold-then-simulate, and the whole thing rests on the teacher having pose-stable, informative
features — neither of which is established. The plan therefore spends Phase 0–1 buying that
information cheaply and only builds a trunk if the measurements justify it.

Independently of how the teacher-activation track lands, **Track B (§3) produces a publishable
result either way**, which is why it runs in parallel rather than after.

---

## 3. Plan

### Phase 0 — Is there anything here? (~1 week, few GPU-hours)

Nothing here trains a model. Every step is a measurement with a kill condition.

**0a. Teacher stability triad.** Run 50 EMDB maps through `teachers/cryofm_tap.py`, tapping
`up_blocks[0]` (primary), `mid_block` (coarse) and `up_blocks[1]` (fine). For each residue's Cα,
trilinear-sample the feature volume. Measure:
- **Pose stability** — cosine similarity of the feature across the 24 cube rotations.
- **Noise stability** — across noise draws at the chosen `t`, swept over `t ∈ {1, 10, 50, 200}`.
- **Tiling stability** — same residue at different crop offsets.

> **Gate:** median pose-cosine < ~0.5 → the teacher activation track is dead as specified; try a
> deployment fix (below) and re-measure, or switch teacher.

**Deployment options if pose stability is marginal** (decide empirically in Phase 1, do not assume):
1. **Pose-averaging** over the 24 cube rotations. Well-defined, but *lossy* — averaging vectors
   whose pairwise cosine is only ~0.3–0.5 collapses toward the mean and may destroy the
   residue-specific signal that makes the feature worth predicting.
2. **Canonical frame** — orient every structure by inertial axes with a deterministic sign
   tie-break, extract once. Cheaper and preserves residue-specificity. Caveats: inertia tensors are
   near-degenerate for globular proteins so the frame *flips* under small structural change, making
   the target a discontinuous function across the dataset; and it does nothing for tiling.
3. **Single fixed orientation** (deposited frame), accepting the artifact as noise.

**What the pose number means regardless of deployment choice.** It is a *diagnostic*: low
pose-cosine says a large share of the feature's variance encodes grid placement rather than density
content. Canonicalising makes that artifact deterministic but not *learnable* (it is a
high-frequency function of structure), so the measured fraction upper-bounds achievable R² under
any of the three options.

**Two instabilities that no orientation fix addresses:**
- **Tiling.** Translating the volume at *fixed* pose already moves features by a comparable amount
  (smoke test: median 0.35–0.66). The problem is grid alignment in general, not orientation.
- **Homo-oligomer symmetry.** In a C_n complex, identical sequence positions occupy n distinct box
  locations and receive n different features from a non-equivariant network — irreducible error for
  a per-residue sequence model. Only genuine invariance fixes this.

**0b. Alignability, before any training.** CKA and mutual-kNN (Platonic Rep. Hypothesis metric)
between ESM-C 600M per-residue embeddings and pose-averaged teacher features at matched residues.
Compare against CKA(ESM-C, random-teacher) and CKA(ESM-C, raw-density-patch).

**0c. Teacher bake-off.** Same measurements for: CryoFM2 taps, CryoFM1-S `down_levels[0]`
(6 Å/token — the right granularity for per-residue) if the `natten` env cooperates, CryoSiam
voxel features, and the raw normalised density patch as a control. Pick the winner on
stability × informativeness.

**0d. Simulated↔experimental gap, in feature space.** For maps with a fitted model, simulate
density from the **deposited** structure (`gemmi DensityCalculator`, matched resolution / ADPs /
mask) and compare teacher features against those from the experimental map — same protein, same
coordinates, only the density differs. No folding in the loop. A few hundred maps, ~an hour of GPU.

This is the only reason to build the simulation machinery at all: the voxel-space gap is already
well documented, but whether it propagates into a *denoiser's* features is not, and cannot be
deduced from the voxel-space result (see Feasibility §3).

> **Gate — decides the fate of the Phase 1 comparator:**
> - *Small gap* → the fold-then-simulate comparator is valid; run it in full.
> - *Large gap* → **drop the full comparator.** Keep only the gap-immune variant: ESMFold2-simulated
>   vs deposited-simulated features, which isolates folding error with the domain gap cancelled.

### Phase 1 — The ceiling/floor sandwich (~2 weeks)

Ridge and MLP probes predicting the chosen teacher feature `T(i)` per residue, protein-level +
30%-identity-clustered splits. This is the core scientific measurement:

| Probe input | What it establishes |
|---|---|
| Shuffled / random embeddings | **Leakage floor** |
| Amino-acid one-hot | Trivial floor |
| ±15-residue sequence window one-hot | Local-sequence floor |
| **ESM-C 600M per-residue** | **The number the project is about** |
| ESM-C, best layer (sweep, à la `../scripts/layer_sweep.py`) | Whether depth matters |
| True Cα neighbourhood (coords only, no density) | "What folding would give you" ceiling |
| Teacher feature at a different noise draw | Teacher self-consistency ceiling |

**Also in Phase 1, conditional on the 0d gate — the analytic comparator.**

- *Always run (gap-immune):* **ESMFold2-simulated vs deposited-simulated** teacher features. Both
  sides in the simulated domain, so the domain gap cancels; isolates folding error alone.
- *Only if 0d showed a small gap:* the full **ESMFold2 → simulated → teacher vs experimental-map
  teacher** comparison. Report paired with the 0d number so folding error and simulation-domain
  error stay separable — unpaired, a weak score is uninterpretable.

> **Gate:** if ESM-C ≈ the local-sequence floor, the premise fails — write it up and stop.
> If ESM-C ≈ the coords ceiling, folding is unnecessary and the project has its headline.
> If the analytic comparator (where 0d licenses it) ≈ the true teacher, the learned module is a
> shortcut — reframe as a distillation/speed result rather than a representation result.

### Track B — interpretable density-derived targets (runs in parallel with Phase 1)

Same probe harness, but predicting **real, named, density-derived per-residue quantities** that a
structure prediction does not hand you:

- per-residue **local resolution** / resolvability (from half-maps)
- **Q-score** (map–model fit)
- fitted-model **B-factor / ADP**
- local density occupancy / SNR

These are meaningful whether or not the teacher-activation track works, have an existing literature
to compare against (RMSF-net), and constitute "features related to cryo-EM density" in the plainest
sense. **This is the plan's insurance policy.**

### Phase 2 — Only if Phase 1 clears the gate (~4–6 weeks)

An **ablation ladder**, escalating only when the previous rung is beaten. The sibling project's
central lesson is that raw ESM is hard to beat and a big module often adds nothing — so this is
built as a ladder, not as an Evoformer:

1. Linear ridge on ESM-C (Phase 1 result, carried forward)
2. Per-residue MLP + learned softmax over ESM-C layers (the ESMFold layer-combination trick)
3. Sequence-context transformer, no pair track (reuse `../scripts/liere_layers.py` with coords
   disabled — the sibling project found seq-context alone beat structure-conditioned attention)
4. Pair trunk, **triangle multiplication only, no triangle attention** (Pairmixer recipe)
5. **ESMFold2 trunk** (MIT, 189 M, already consumes all ESM-C layers, already emits a `d_z=256`
   pair rep, already triangle-attention-free) with a small adaptor head

> Do **not** train an Evoformer from scratch. Pairmixer cost 192 GPU-days for the *efficient*
> version, and ESMFold2 already is the architecture you sketched.

**Loss:** LiT framing (teacher frozen) + AM-RADIO-style dense regression — cosine + smooth-L1 on
**whitened** teacher features — where residue→voxel correspondence exists, plus an optional SigLIP
global term at map level.

---

## 4. Files and reuse

New code in `sequence_vision/`:

| File | Purpose |
|---|---|
| `data/build_emdb_pairs.py` | EMDB ≤3 Å + fitted PDB → paired manifest; consider bootstrapping from Cryo2StructData |
| `teachers/cryofm_tap.py` | **BUILT & VERIFIED.** Loads CryoFM2 `UNet3DModel` (168.1 M, 0 missing/0 unexpected), forward hooks, 24-rotation cube group (self-test 24/24 consistent), Fourier resampling, patch stitching, trilinear sampling at Cα. `in_channels=2` — ch1 zeros for the unconditional model |
| `teachers/{cryosiam,simulated}_tap.py` | Alternate teachers, same interface |
| `data/simulate_density.py` | Simulate density from a structure at matched resolution / ADPs / mask. Try `gemmi DensityCalculator` first (already installed); `phenix.fmodel` only if gemmi is insufficient. Used by Phase 0d and, conditionally, the Phase 1 comparator |
| `probes/stability.py` | Phase 0 pose / noise / tiling triad + the 0d sim↔experimental gap |
| `probes/alignability.py` | CKA + mutual-kNN |
| `probes/ceiling_sandwich.py` | Phase 1 probe table |
| `probes/density_targets.py` | Track B targets |
| `train_align.py`, `config.yaml` | Phase 2 only |

Reuse from `../scripts/` (do not rewrite):

- **Training scaffolding** in `lightning_decoder.py:1220-1491` — OmegaConf flat-kwargs config,
  `DECOMP_CONFIG` env override, **deterministic-hash W&B run ID with `resume="allow"`**,
  `_find_resume_checkpoint()` corruption-tolerant preemption recovery, DDP + `sqrt(num_devices)`
  LR scaling, timm cosine+warmup, run-name-scoped `ModelCheckpoint`, all-gathered R² buffers.
- **Data pattern** — ragged concat + `cumsum` offsets + `mmap_mode='r'` + padding/masking collate
  (`ESMCEmbeddingsDataset` / `collate_batch` in `preprocess_sequences.py`). Scales to arbitrary
  size with zero load time; reuse verbatim for the paired dataset.
- **ESM-C extraction** — `preprocess_sequences.py` (`ESMC.from_pretrained("esmc_600m")`,
  strip BOS/EOS) and `layer_sweep.py` for the all-layer `[36,1,L+2,1152]` tensor.
- **Probe harness idioms** — `ridge_predictability.py` (ridge-then-MLP-on-residual, which cleanly
  isolates the nonlinear gap and is regularisation-fair) and `raw_esm_probe.py`.
- **SLURM** — `../../slurm/plm_decomp.slurm` (Slack notifications, `torchrun`, pixi). Retarget from
  `l40-reserved` to `h100-reserved`.
- **Figures** — `make_summary_figures.py` conventions + the `dataviz` skill palette.

**Environment:** add `mrcfile`, `zarr`, `starfile` to the pixi env; CryoFM1 needs a **separate**
env (`natten` / torch 2.5.1 / CUDA 12.4). For density simulation, **`gemmi` 0.7.5 is already
installed** and provides `DensityCalculator` / `sfcalc` — try that before installing Phenix, which
is a heavyweight separate (academically licensed) install. Note all `/mnt/runai-*` paths in
existing configs are dead and need rewriting to
`/mnt/main0/projects/et-foundation-vision-model/`.

---

## 5. Verification

- **Phase 0a** — a residue's feature under 24 cube rotations must round-trip: sample the *same*
  physical point and report the cosine distribution. Sanity check the whole tap by confirming
  CryoFM2 denoising via the released `cfm denoise` CLI reproduces the paper's behaviour on one map
  before trusting any hook output.
- **Preprocessing** — ✅ verified on EMD-65504: 224³@0.931 Å → 140³@1.5 Å, mean −0.286 / std 0.976.
  **Correction to an earlier draft of this plan: there is NO negative clipping.** The exact
  inference chain is `v/percentile(v, 99.999)` then `(v−0.04)/0.09`
  (`sampling_helper.py:318-330`); `ClipNorm3D` belongs to CryoFM1, not CryoFM2. Resampling is
  Fourier-domain and energy-preserving (`resize_by_voxel_size`), not `scipy.zoom`.
- **Correspondence** — ✅ verified quantitatively rather than visually: density at Cα vs at random
  voxels on EMD-65504 = **+3.447 vs −0.298, contrast +3.745**. Re-run this assertion per map at
  scale; it catches origin/`nstart` handling errors, which are the likely failure mode (EMD-65504
  has origin=0 and nstart=0, so it did *not* exercise that path).
- **Every probe** — report the shuffled-embedding control alongside the real number, always.
  Splits are protein-level *and* 30%-identity-clustered; report both.
- **Phase 2** — each ladder rung must beat the rung below on a held-out protein set, or stop there.

---

## 6. Open items

- Whether CryoLVM weights are actually released (paper is very recent; only the abstract was
  reachable). CryoSiam's code is confirmed on GitHub.
- Whether to bootstrap the paired dataset from Cryo2StructData rather than curating EMDB from
  scratch — likely yes, worth an hour of checking first.
- `6Proteins_Lys_RP` (198 tomograms, 3 pick jobs) is on disk but the star files carry
  **coordinates only, no class labels** — so it is not a ready-made 6-way identification benchmark.
  Irrelevant to the SPA scope chosen here, but noting it since it looked promising.
