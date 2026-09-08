# sequence_vision

Can ESM-C sequence embeddings be aligned to the internal features of a cryo-EM density
foundation model? A representation-research study on single-particle cryo-EM.

**Read [`PLAN.md`](PLAN.md) first** — literature, feasibility assessment, and the phased
plan with its go/no-go gates.

## Status (2026-09-08) — two-tower ESM-C/CryoFM alignment, G0 cleared

Dense **voxel <-> sequence** alignment: frozen CleanDIFT-CryoFM density features (`up_blocks[1]`,
256-d, 3 A/cell) + frozen ESM-C layer 32, with ~1-2 M trainable parameters per tower.

**E1 — per-voxel chain assignment.** Given a map (no atomic model) and the set of distinct
sequences it contains (median 7), name the owner of each voxel. 225 test maps, paired cluster
bootstrap, all arms in one process on identical voxels.

| arm | top-1 | macro IoU |
|---|---|---|
| **+ token mixing** | **0.5836** | **0.4023** |
| pointwise + score smoothing (sigma=8-12 A) | 0.4716 | 0.2923 |
| pointwise (v1) | 0.3968 | 0.2440 |
| volume prior | 0.3090 | 0.0891 |
| AA classifier + composition matching | 0.2504 | — |
| shuffled pairing | 0.2352 | — |
| untrained head | 0.2125 | — |
| random-weight vision tower | 0.1971 | — |

All four mandatory baselines beaten with CIs excluding zero. The random-weight floor (0.1971, vs
chance ~0.18) is the important one: CryoFM's pretrained weights carry +0.1967, so this is not
architecture-plus-geometry masquerading as learned features.

**E2 — map -> sequence retrieval**, 11-way length-matched cluster-disjoint pools:
**0.6978** with mixing (0.6400 pointwise), vs AA-composition 0.1733, length-only 0.0978,
chance 0.0909.

**Token mixing is the largest single gain, and 60-69% of it is not smoothing.** Distance-biased
attention over a map's voxels (translation/rotation-equivariant, zero-init). Post-hoc Gaussian
smoothing of the *pointwise* scores reproduces 40% of the gain on top-1 but only 31% on macro IoU;
mixing still wins by +0.1100 [+0.0957, +0.1249] macro IoU. A **fixed feature blur** (no learned
attention) reaches margin +0.1735 vs mixing's +0.3696, and blur+mixing (+0.3316) is *worse* than
mixing alone — the attention needs un-blurred fine features.

**Resolution (G4) — the method holds across the 4-10 A niche and breaks at ~12 A** (val margin over
the volume prior):

| low-pass | pointwise | + mixing |
|---|---|---|
| 1.5 A (native) | +0.0964 | +0.3696 |
| 4 A | +0.1322 | +0.3765 |
| 8 A | +0.1026 | +0.3640 |
| 12 A | +0.0688 | +0.2730 |
| 20 A | +0.0548 | +0.2583 |

Flat from 1.5 to 8 A, then degrading — so chain assignment is a *mesoscale* task, not a coarse-shape
one, and it survives where backbone tracing does not (ModelAngelo per-residue identification: 49%
top-1 at 4-5 A, **0%** at 5-10 A). Mixing retains ~70% of its native margin even at 20 A.

**Not yet done:** the external comparison against ModelAngelo's own chain assignment; test-set evals
for the resolution ladder (numbers above are val).

Full experimental record, including every retracted claim, in
[`PLAN_DINOTXT.md`](PLAN_DINOTXT.md) sections 8.19-8.21.

## Status (2026-08-22, superseded above)

**The cross-modal experiment ran: ESM-C predicts per-residue density features.** 1,500 chains,
672 mmseqs clusters at 30% identity, split BY CLUSTER; 279k train / 52k test residues.

| arm | dim | R² | cos |
|---|---|---|---|
| shuffled | 1152 | −0.211 | **+0.004** ✓ |
| seqwin3 (±3 one-hot) | 140 | +0.036 | +0.176 |
| **esmc + MLP** | 1152 | **+0.187** | **+0.417** |

ESM-C beats local sequence ~5×, and the map is essentially linear (an MLP adds +0.018 over ridge).
Achieved 0.187 against 0.447 sequence-tracked and a 0.849 pose ceiling — ~42% of what is
sequence-trackable. **The "alignability at all" bar is cleared, with controls.**
Judge a shuffled arm on COSINE: its R² is negative by construction.

Run it: `sbatch slurm/align_targets.slurm && sbatch slurm/align_esmc.slurm && sbatch slurm/align_run.slurm`

---

**Earlier findings — pooling was the mistake, and CryoFM is not the best teacher.** The homolog diagnostic
(`probes/homolog_diagnostic{,_residue}.py`, 300 homolog + 300 length-matched unrelated pairs) screened the
target before any training. R² ceiling = the fraction of descriptor variance any sequence model
could ever explain -- more precisely, the fraction that TRACKS SEQUENCE SIMILARITY at homolog
resolution (a smoothness measure, not a strict bound; the hard ceiling is `1 - pose_fraction`).

**POOLED (simulated density), R² ceiling before → after partialling `[N, Rg, AA comp]`:**

| descriptor | R² ceiling | − trivial |
|---|---|---|
| trivial `[N, Rg, comp]` | 0.629 | — |
| CryoFM `up_blocks[1]` | 0.145 | **−0.022** |
| CryoFM `up_blocks[0]` | 0.089 | **0.002** |

**PER-RESIDUE, full ladder (280 pairs, ~10.7k residues), after −(±3 sequence window):**

| rung | needs atomic model? | sequence-tracked | pose noise |
|---|---|---|---|
| **CryoFM, backbone frames** | yes | **0.447** | 0.151 |
| CryoFM, 24-frame averaging | no | 0.255 | 0.196 |

`up_blocks[0]` in local backbone frames is the working per-residue target. The 24-frame
model-free rung keeps ~57% of its sequence-tracked signal, which is the number to improve if
the pipeline is to run on maps with no fitted model.

**Conclusion.** *Pooling was the mistake* — the sequence-predictable content of a density
feature is LOCAL; pooling averages away the predictable part and keeps global size/shape, which
`[N, Rg, comp]` already explains. That is the third instance of this pattern in the
project lineage.

The pose gate passed in both arms, so orientation is *not* what kills the pooled version.
**OOD is not the explanation either** — on raw experimental maps CryoFM is relatively worse, though
that arm is noisy — every density-derived descriptor collapsed on raw maps while `trivial` was
unchanged at 0.645, so the degradation is instrument nuisance + neighbouring chains, not CryoFM).

Target **per-residue local-frame features**. The pooled target is out: after partialling
`[N, Rg, AA comp]` it has no headroom left.
See `../CLAUDE.md` for the full account — it is the authoritative log.

**Data: do not re-download.** A complete curated corpus is already on this filesystem —
`/mnt/main0/projects/hypernetworks-for-cryo-em/cryo2structdata-3A/full/` — **7,361 entries, all with
raw map + fitted PDB + FASTA + per-voxel labels**. Read in place. Our own `data/maps/` (1,146 recent
maps, 164 GB) covers the post-2023 gap that corpus lacks.

| Phase | What | State |
|---|---|---|
| 0 — homolog diagnostic | is the target sequence-predictable at all? | ✅ **yes per-residue (0.447); pooled is dead** |
| 0 — env + tap | CryoFM2 loading, preprocessing, feature tap | ✅ verified |
| 0a — stability triad | pose / noise / tiling | ✅ done — but **cube rotations**, see below |
| 0 — pose invariance | which targets are well-defined | ✅ resolved (table below) |
| 0 — data | paired manifest | ✅ 300+300 pairs from Cryo2StructData |
| 0d — sim↔experimental gap | decides the project's utility | ❌ not started |
| 1 — cross-modal test | **the actual question** | ✅ ran 2026-08-22 — ESM-C→CryoFM R² 0.187 (see `../CLAUDE.md`) |

### Pose-invariance question: resolved

Invariance is measured under **generic SO(3)** where marked. The old cube-rotation numbers flattered
the model: CryoFM2 pretrains with `RotCube24` at `p=1.0`, so the octahedral group is exactly what it
was augmented to be invariant to.

| target | invariance | needs atomic model | granularity |
|---|---|---|---|
| raw per-residue CryoFM, SO(3) | 0.28–0.43 (top-1 2–7%) | no | per-residue |
| **+ 24-frame octahedral averaging** | **0.63–0.77** (top-1 9–28%) | **no** | per-residue |
| **pooled CryoFM** | **0.95 self / −0.24 other**, d′ 3.8–6.0 | no* | whole molecule |
| **local backbone frames** | **0.68–0.83** (top-1 0.53–0.82) | **yes — fatal for map-side use** | per-residue |
| whole-map canonicalisation | **0.0°** within-orbit, breaks across proteins | no | whole molecule |
| voxel pair-distribution | **rejected** — dominated by 32-d pooling | no | tunable |

\* samples at Cα as implemented; pool over thresholded voxels to make it genuinely model-free.

**Frame averaging is the best MODEL-FREE lever.** Lossless (transpose+flip), exactly invariant to the
group by construction, no atomic model, no canonical frame — 24× inference and nothing else.

**Tap choice is target-dependent.** `up_blocks[1]` (3 Å/token) is best pooled; **`up_blocks[0]`
(6 Å/token) is best per-residue** (0.826 vs 0.677), which restores the original granularity argument.
Don't carry one ranking across both tracks.

**The per-residue R² ceiling is ~0.83, not ~0.5.** The old cap came from global framing plus the
stitching bug. Pooling still wins on raw invariance and is model-free; local frames win decisively on
residue-specificity.

### Known-bad, fix before trusting any cross-modal number
- **Chain mismatch.** `load_ca()` defaults to *all* chains; the manifest carries *one* chain's
  sequence. 14/27 entries are hetero (`n_unique_seq > 1`), mean 5.15 chains, max 28. Pass `chain_id`
  or restrict the manifest to single-unique-sequence entries.
- **Never report an invariance cosine without its mismatched floor.** A random-weight 3D CNN scores
  raw pooled self-cosine 0.9937 against a *between-protein* 0.9949 — the statistic can be entirely
  the shared constant. Mean-centre, and report the floor.
- The logged `chance = 1/M = 0.033` **does not apply to the length-matched row** (pool ≈ 11 → ≈ 0.10).

### Next three things
1. **Correspondence-source control** — fold-then-fit (ESMFold2 → fit into the map) supplies
   residue↔voxel pairing for *any* (sequence, map) pair, which removes the deposited-model
   requirement and obviates the optimal-transport machinery. But correspondence derived from a
   structure predicted *from the sequence* makes "ESM-C predicts the feature at residue i" partly
   restate "the fold was right". Run the alignment twice — deposited vs predicted correspondence —
   and check they agree. Both exist for all 7,361 Cryo2StructData entries.
2. **CleanDIFT-style timestep consolidation** for the CryoFM tap — see `PLAN_CLEANDIFT.md`.
   Every CryoFM number in this repo was taken at `timestep=10`, an operating point never swept.
3. **Track B screen** — same harness, target swapped to per-residue B-factor / local resolvability.
   The one branch fold-then-simulate structurally cannot serve, and the OOD arm strengthened its
   premise (experimental variance swamps structural signal in every density descriptor).

## Layout

```
PLAN.md                     the study design (start here)
pixi.toml                   dedicated env: python 3.10, torch 2.13+cu130, numpy 1.26.4
teachers/cryofm_tap.py      CryoFM2 feature tap + cube rotations + CA sampling
data/build_emdb_pairs.py    EMDB API -> paired (map, fitted model) manifest
probes/stability.py         Phase 0a triad (NB: cube rotations = CryoFM2's aug group)
probes/homolog_diagnostic.py  THE SCREEN: is a descriptor sequence-predictable?
probes/patch_alignment_ab.py  isolates the stitching bug's real magnitude
probes/pose_invariance_clean.py  local backbone frames, independent grids
probes/frame_avg_check.py   24-frame octahedral averaging correctness + gain
data/build_homolog_pairs.py homolog + length-matched unrelated pairs (Cryo2StructData)
data/simulate_density.py    gemmi DensityCalculatorE, exactly isotropic grid
slurm/*.slurm               1-GPU launchers
third_party/cryofm/         upstream CryoFM (Apache-2.0), installed editable
weights/cryofm-v2/          CryoFM2 pretrain checkpoint (672 MB, 168.1 M params)
data/maps/, data/manifest.csv, results/
```

## Running

```bash
pixi install                                    # once
export PYTHONPATH=$PWD                          # modules are not a package

pixi run python teachers/cryofm_tap.py          # rotation self-test (24/24)
pixi run python data/simulate_density.py        # isotropic-spacing self-test

# the screen: run this on any new descriptor BEFORE building anything on it
pixi run python data/build_homolog_pairs.py --n-pairs 300
LIMIT=150 OUT=homolog_diagnostic sbatch slurm/homolog.slurm
```

## Facts worth not rediscovering

- **CryoFM has no encoder.** It is a velocity field `v(x_t, t)`. "Features" = hooked
  activations at a chosen noise level. Not validated by the CryoFM papers — establishing
  whether they are usable at all is what Phase 0 is for.
- **Preprocessing must match exactly**, or every feature is silently wrong:
  Fourier resample to 1.5 Å/voxel → `v / percentile(v, 99.999)` → `(v − 0.04) / 0.09`.
  There is **no negative clipping** — that is CryoFM1's `ClipNorm3D`, not CryoFM2.
- **`in_channels=2`.** Channel 0 is `x_t`, channel 1 is conditioning; pass zeros for the
  unconditional model or the forward errors.
- **`natten` is not needed.** Soft import, CryoFM1/HDiT only. The separate env exists
  because CryoFM pins `numpy<2.0`.
- **Tap granularity** at 1.5 Å/voxel: `up_blocks[0]` 4096 tok × 512 ch @ 6.0 Å/token (~1–2
  residues); `mid_block` 512 × 512 @ 12.0 Å; `up_blocks[1]` 32768 × 256 @ 3.0 Å. **Best tap depends
  on the target**: `up_blocks[1]` pooled, `up_blocks[0]` per-residue (0.826 vs 0.677).
- **Patches must be aligned to the tap stride.** `GridPatches3D` appends a final patch per axis at
  `n − 64`, which is a multiple of the stride only if `n` is — so the `a // s` stitch truncates and
  misregisters ~57% of patches by up to 10.5 Å, *pose-dependently*. `feature_volumes` now pads each
  axis to a multiple of 64 (free — same patch count) and asserts alignment. Pad with the per-map
  **median**, not zero: normalization maps raw 0 to −0.44, so a zero pad is a slab of density.
  **But the effect is near-cosmetic**: only 14–21% of *residues* land in a misregistered region
  (vs 57% of patches — protein sits mid-box), features move 0.066 cosine there, and the Phase 0a
  median shifts <0.01. `up_blocks[1]` is immune (stride 2). Keep the fix; don't expect it to matter.
- **`simulate()` returns real spacing and now forces it isotropic.** gemmi otherwise rounds each axis
  independently against a fixed cell, giving anisotropic *pose-dependent* spacing — a shear, so
  nothing built on voxel indices is equivariant. Never hardcode 1.5.
- **Always assert Cα/bulk density contrast per map.** Catches origin/`nstart` errors, the
  likeliest silent failure. EMD-65504 gives +3.7; the probe skips anything under +0.5.
- **SLURM**: `sinfo -s` is misleading — it counts a node as allocated if *any* GPU is busy.
  Check for `mix` state; ~140 h100 nodes usually have free GPUs, so 1-GPU jobs schedule fast.
