# Phase 0 completion: is pooled ESM-C alignable with pooled CryoFM2 density features?

## Context

The project asks whether ESM-C sequence embeddings can be aligned to the internal features of a
cryo-EM density foundation model. Scope is now fixed: **single-particle cryo-EM** (EMDB ≤3 Å with
fitted PDB models), and the success bar is **alignability at all** — a representation-science
result, not a requirement to beat any baseline or power a task.

**What changed since the original plan.** Two measured results reshaped it:

1. **Per-residue targets are capped.** CryoFM2 per-residue features are only 31–51% pose-invariant,
   which hard-bounds R² for any pose-free sequence model at ~0.5. Retrieval showed the surviving
   signal is a local-environment *class* descriptor, not a positional fingerprint (top-1 match sits
   42–46 Å away vs 49.8 Å random).
2. **Structure-level pooling fixes it.** mean+std pooling over ~1000 residues raises pose invariance
   to **0.98–0.995** (p10 0.91–0.98) and gives perfect self-retrieval including length-matched.
   The R² ceiling goes ~0.5 → ~0.98. Tiling and homo-oligomer symmetry ambiguity vanish too.

**Teacher choice is settled and needs no further search.** CryoSiam was investigated as a
pose-robust alternative and **rejected on evidence**: its released checkpoint configs show *no
rotation or flip augmentation in pretraining* (crops, zoom, noise, low/high-pass, dropout only),
its architecture is a plain MONAI ResNet+FPN with no equivariance, and its dense loss explicitly
trains for *translation* correspondence — so its per-residue features would likely be worse than
CryoFM2's. CryoLVM and APT-ViT have no released weights. No pretrained SE(3)-equivariant cryo-EM
density encoder exists. And pooling already solved pose, so there is no pose-motivated reason to
switch. **Stay with CryoFM2.**

**Intended outcome:** a defensible answer to "do pooled ESM-C and pooled cryo-EM density features
share structure beyond trivially shared covariates," with the controls that make the answer mean
something.

---

## Already validated — do not redo

| Thing | State | Where |
|---|---|---|
| Dedicated env (py3.10, torch 2.13+cu130, numpy 1.26.4) | working | `pixi.toml` |
| CryoFM2 loading (168.1 M, 0 missing/unexpected) | working | `teachers/cryofm_tap.py` |
| Exact preprocessing (Fourier resample → /p99.999 → (x−0.04)/0.09, **no clip**) | verified | same |
| 24-element cube rotation group | self-test 24/24 | same |
| Cα sampling + density-contrast assertion | +3.7 on EMD-65504 | same |
| Pooled descriptor pose invariance 0.98–0.995 | measured, 30 maps | `probes/pooled_stability.py` |
| 50 EMDB↔PDB pairs | on disk, 6.9 GB | `data/manifest.csv` |
| 1-GPU SLURM pattern (~11 min / 12 maps) | working | `slurm/*.slurm` |

---

## Work

### Step 1 — Scale and de-duplicate the dataset (blocker)

Current manifest is **36% two deposition series** (11× `9ye`, 7× `9w4`). Tolerable for measuring a
network property; **not** acceptable for a correlational claim across proteins, where correlated
entries inflate apparent evidence.

- Extend `data/build_emdb_pairs.py`: pull sequences from the fitted `.cif`, cluster with **mmseqs2
  at 30% identity**, keep one representative per cluster.
- **Restrict to maps with a single unique sequence** (homo-oligomers fine, hetero-complexes
  excluded) so map↔sequence pairing is unambiguous. Record how many are dropped.
- Target **300–500 clustered pairs**. EMDB caps near 10⁴ usable entries, so this is a sampling
  decision, not a hard limit — log what was excluded rather than silently truncating.

### Step 2 — ESM-C embeddings for the paired sequences

`esm` is absent from this env and adding it risks the `numpy<2.0` pin. **Extract in the main lab
env and cache**, rather than merging dependency trees:

- Use `../../aden/aden/pixi.toml` (has `esm 3.2.1.post1`) with the existing extraction pattern from
  `../scripts/preprocess_sequences.py` (`ESMC.from_pretrained("esmc_600m")`, strip BOS/EOS).
- Cache per-chain embeddings to `data/esmc/` as `.npy` + a manifest join key.
- Pool to a structure-level descriptor with **mean+std**, matching `pool()` in
  `probes/pooled_stability.py` so both sides are constructed identically.

### Step 3 — The measurement (`probes/cross_modal.py`)

For each paired (map, sequence), a pooled CryoFM2 descriptor **D** and a pooled ESM-C descriptor
**S**. Measure alignment three ways — they fail differently, so report all:

- **Linear CKA** between the D and S matrices.
- **Mutual k-NN** alignment (the Platonic Representation Hypothesis metric, arXiv 2405.07987).
- **Cross-modal retrieval**, both directions, top-1 and top-5, with an explicit chance line.

Use `up_blocks[1]` as primary (highest pooled pose invariance, 0.995); report `mid_block` and
`up_blocks[0]` alongside.

### Step 4 — Controls (the part that determines whether the result means anything)

With "alignability at all" as the bar, a naive positive is nearly guaranteed — both descriptors
encode protein size. Every headline number must be reported next to:

1. **Shuffled pairing** — permute the map↔sequence correspondence. The floor.
2. **Length-matched retrieval pools** — restrict competitors to similar residue count. Already
   implemented in `probes/pooled_stability.py`; reuse.
3. **Length-and-composition partialled out** — regress residue count and 20-d amino-acid
   composition out of *both* D and S, then recompute CKA / kNN / retrieval on the residuals. **This
   is the headline number**, not the raw one.
4. **Trivial-descriptor baseline** — replace D with `[n_residues, AA composition]` and rerun. If
   this scores as well as the real descriptor, CryoFM2 is contributing nothing.
5. **Random-teacher control** — random-weight CryoFM2, same pipeline. Separates "the trained
   network encodes something" from "any 3D conv over density does."

---

## Decision points

- **Partialled CKA / retrieval ≈ shuffled floor** → the modalities are not alignable beyond size
  and composition. Clean negative; write it up and stop.
- **Partialled alignment clearly above floor, above trivial-descriptor, above random-teacher** →
  the bar is met. The result stands on its own; optional follow-ons then become interesting:
  mean-pooled raw ESM-C as a comparator, a small projection head (ProteinCLIP/LiT-style — the right
  regime at ~10³ samples, not large-scale contrastive training), and the per-residue track.
- **Real but explained by the trivial descriptor** → report as such. Still a result, and a more
  useful one than a vague positive.

## Deprioritised by the "alignability at all" bar

Recorded so the reasoning isn't lost: the **fold-then-simulate comparator** (it tested whether a
*learned* module is necessary — not the bar now), **Track B** interpretable density targets, the
**per-residue track** (capped at R² ~0.5), and the **Phase 2 ablation ladder** including any
Evoformer/ESMFold2 trunk. None are dead; none are on the critical path.

---

## Files

| Path | Change |
|---|---|
| `data/build_emdb_pairs.py` | add mmseqs2 30% clustering, single-unique-sequence filter, sequence extraction; scale to 300–500 |
| `data/extract_esmc.py` | **new** — ESM-C 600m via the main lab env, cache to `data/esmc/` |
| `probes/cross_modal.py` | **new** — CKA, mutual-kNN, bidirectional retrieval, all five controls |
| `probes/pooled_stability.py` | reuse `pool()` and the length-matching logic; do not duplicate |
| `slurm/cross_modal.slurm` | copy `slurm/pooled.slurm` (1× h100) |
| `PLAN.md`, `README.md`, `../CLAUDE.md` | fold in results when they land |

## Verification

- **Pairing integrity** — assert the sequence pooled on the ESM-C side is the same chain whose Cα
  coordinates were pooled on the density side. A silent join error here fabricates or destroys
  alignment; check a handful by hand against the `.cif`.
- **Reuse the per-map Cα/bulk density contrast assertion** (skip below +0.5) — it catches
  origin/`nstart` errors, the likeliest silent failure at scale.
- **Shuffled pairing must land at chance.** If it doesn't, there is leakage in the pipeline and
  every other number is void. Run this first, not last.
- **Sanity-check the clustering** — no two retained entries above 30% identity; report cluster
  count and dropped count.
- Re-run `teachers/cryofm_tap.py` self-test (24/24) after any change to rotation or sampling code.

## Open risks

- **~10³ samples** is small for cross-modal work. Adequate for CKA/kNN/retrieval on frozen
  descriptors; inadequate for training anything sizeable. Keeps the study correlational by design.
- **mean+std pooling is a choice, not a given.** It worked for pose invariance; it may not be the
  most alignable pooling. Attention-pooling or a learned head could differ — out of scope here,
  worth noting before concluding "not alignable."
- **Ericsson et al. caution**: high pooled cosine can coexist with poor retrieval if the invariant
  subspace is low-dimensional. Our retrieval test already covers this; keep reporting both.
