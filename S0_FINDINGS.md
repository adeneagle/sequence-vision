# Stage 0 — ESM-C x CleanDIFT/CryoFM additivity on per-residue interface (2026-09-04)

Plan: `~/.claude/plans/nifty-brewing-kay.md`. All numbers below are ≤3 Å (the anchor
rung); no resolution ladder yet. New code: `probes/s0a_clearance_audit.py`,
`probes/s0b_labels.py`, `probes/s0c_features.py`, `probes/s0c_additivity.py`.
Nothing pre-existing was modified.

## 0a — the clearance audit: GATE PASSES
`results/s0a_clearance_audit.json`, 300 chains, 72,101 residues, 0 skipped, 20 s CPU.

The 48 Å clearance filter (`probes/o5_boxes.py:60`) drops **3.4%** of residues, and those
dropped are **1.39×** enriched for interface (0.413 vs 0.296) — so the kept population's
interface rate is 0.2962 against a true 0.3002, a 1.3% relative shift. The loss is highly
concentrated: **median per-chain drop 0.000, p90 0.000**; 14/300 chains lose >20%, 4 lose all.

**The box path is valid for interface work.** Fix by excluding/re-padding those 14 outliers,
not by moving to the volume path (which would have cost whole-map canonicalisation or 24×
frame averaging). CLAUDE.md's "every surface residue discarded" was *simulated* density with a
12 Å margin against a 48 Å requirement; real EMDB maps are generously padded (e.g. 252³ voxels
= 378 Å), so the filter barely bites.

## 0b — labels, per chain INSTANCE
`data/s0b_labels.npz`, `results/s0b_labels.json`. 1,479/1,485 chains, 87,951 cached residues,
3 min CPU. Integrity: **0 coordinate mismatches, all 1,479 row-verified against the cached
`aa`** (the check also correctly rejected 5 non-aligning chains).

Pooled positive rates: `iface5_heavy` 0.368 · `iface6_ca` **0.1044** · `iface_hetero` 0.347 ·
`iface_homo` 0.024 · `iface_nuc` 0.070 · `lig5_nonion` 0.016. The 6 Å Cα–Cα rate matches
ProteinShake's independently reported 9.0% mean — external validation of the implementation.

Homo/hetero is decided by `map_chains.csv::seq_idx`, **not** observed-sequence string equality,
which under-counts copies whose observed residues differ; before the fix `iface_homo` read 0 on
every chain, after it fires on 149.

### Power pre-flight (this changed the plan)
Test split: 227 chains, **99 clusters**, 13,367 residues.

| label | test positives | verdict |
|---|---|---|
| `iface5_heavy` | **5,332** (39.9%) | well powered |
| `iface6_ca` | 1,639 | viable |
| `lig5_nonion` | **243** | **underpowered** — below the plan's 300 floor |
| `iface_homo` | 259 | descriptive only |

**P2 (pocket) cannot be read from the existing caches.** Remedy: over *all* observed residues
`lig5_nonion` has 4,978 positives (1.38%), so extracting the 311 ligand-bearing chains in full
(~45 min GPU) recovers ~880 test positives. Otherwise use the PLINDER route.

## 0c — the additivity read: ADDITIVITY IS REAL
`results/s0c_additivity.json`. 1,479 chains, 87,951 residues (train 66,424 / val 8,160 /
test 13,367), 99 test clusters, 5,332 test positives, label `iface5_heavy`, arm
`student_paperhead`, tap selected on val, paired cluster bootstrap n=1000.

**Control passes:** `shuffled` AP 0.4029 vs base rate 0.3989, AUROC 0.5007, MCC −0.0061.

| arm | dim | AP | AUROC | MCC |
|---|---|---|---|---|
| `shuffled` | 1152 | 0.4029 | 0.5007 | −0.0061 |
| `seqwin3` | 140 | 0.5003 | 0.5965 | 0.1320 |
| `raw` (8³ voxels) | 512 | 0.5240 | 0.6240 | 0.1892 |
| `dens_stats` | 12 | 0.5269 | 0.6456 | 0.2095 |
| `rand@up_blocks[1]` | 256 | 0.5400 | 0.6532 | 0.2180 |
| `own_geom` | **13** | 0.6447 | 0.7271 | 0.3236 |
| `esmc` | 1152 | 0.6731 | 0.7725 | 0.3998 |
| `dens@mid_block` | 512 | 0.6783 | 0.7614 | 0.3950 |
| `fused@up_blocks[1]` | 1408 | 0.7573 | 0.8260 | 0.4835 |
| `fused_geom` | 1177 | 0.7787 | 0.8360 | 0.5113 |
| **`fused_all@up_blocks[1]`** | 1433 | **0.8383** | **0.8823** | **0.5937** |

ΔAP, paired cluster bootstrap over the 99 test clusters:

| comparison | ΔAP | 95% CI | MDE | |
|---|---|---|---|---|
| **`fused − esmc`** | **+0.0842** | [+0.0502, +0.1222] | 0.0363 | SIG |
| `dens − raw` | +0.1507 | [+0.1129, +0.1869] | 0.0372 | SIG |
| `dens − dens_stats` | +0.1479 | [+0.1057, +0.1924] | 0.0422 | SIG |
| `dens − rand` | +0.1347 | [+0.0945, +0.1760] | 0.0404 | SIG |
| `fused_geom − esmc` | +0.1056 | [+0.0729, +0.1468] | 0.0364 | SIG |
| **`fused − fused_geom`** | **−0.0214** | [−0.0424, +0.0007] | 0.0212 | **ns** |
| **`fused_all − fused_geom`** | **+0.0596** | [+0.0382, +0.0836] | 0.0231 | SIG |

### Reading
1. **Additivity is real.** `fused − esmc` = +0.084, CI excludes zero, MDE 0.036. The density
   channel adds to ESM-C on interface.
2. **It is not merely "there is mass next door."** CryoFM beats the 12-number radial-density
   control by +0.148 and the tap-matched random control by +0.135, so it is neither trivial
   occupancy nor architectural smoothing.
3. **But CryoFM is not the efficient route to the first chunk.** ESM-C + 25 hand-crafted
   numbers (`fused_geom`) *equals or beats* ESM-C + 256-d CryoFM: point estimate **−0.021**.
4. **CryoFM does carry unique information**: +0.060 SIG on top of ESM-C *and* cheap geometry.
   That is the honest size of the foundation model's non-redundant contribution here.
5. **`own_geom` at 13 dims (AP 0.6447, MCC 0.3236) is a formidable rival** — monomer geometry
   alone nearly matches 1152-d ESM-C. Any future claim must clear it.

### My pre-registered tap prediction was WRONG
The plan predicted mesoscale interface would favour the coarse taps (`up_blocks[0]` ~10 Å,
`mid_block` ~15 Å half-decay) rather than `up_blocks[1]` (~4–5 Å), which won secondary
structure. Best tap by val is **`up_blocks[1]`** again, with `mid_block` close behind
(`fused_all` 0.8336 vs 0.8383). Tap choice does not track task spatial scale the way predicted.

## Caveats that gate the headline
- **`esmc_seqctx` was not run, and this is a positive result.** The harness states the rule:
  since sequence-context attention can only *raise* the sequence baseline (sibling: raw-ESM
  0.236 → 0.375 with context, while LieRE *with* coordinates gave 0.318), **a null here would
  have been decisive but a positive is not.** Whether +0.084 survives a context-aware sequence
  baseline is now the single most important open question.
- ≤3 Å only; no resolution rung, so nothing here tests the low-resolution claim.
- Deposited-model correspondence: no docking error, and `docked_geometry` is degenerate at
  Stage 0 (it is the label generator), hence the `own_geom` substitute.
- Both label definitions now run; the 6 Å Cα–Cα replicate is below.
- `o5_stats.mcnemar_exact` documents itself as **secondary and anticonservative**, contradicting
  the plan's suggestion to use it as primary. The cluster bootstrap is primary; McNemar is not
  gated on.

## 0c replicate — 6 Å Cα–Cα (ProteinShake's definition)
`results/s0c_additivity_iface6ca.json`. Same chains/split/arms; sparser label
(test 1,639 positives, 12.3%). **Every qualitative conclusion replicates, and additivity is
LARGER on the harder label.**

`shuffled` AP 0.1270 vs base 0.1226, AUROC 0.5074, MCC 0.0081 — control passes.

| arm | AP | MCC |
|---|---|---|
| `own_geom` (13-d) | 0.2599 | 0.2348 |
| `esmc` | 0.2639 | **0.2467** |
| `dens@up_blocks[1]` | 0.3481 | 0.2873 |
| `fused@up_blocks[1]` | 0.3761 | 0.3375 |
| `fused_geom` | 0.4048 | 0.3545 |
| **`fused_all@up_blocks[1]`** | **0.5333** | **0.4712** |

| comparison | ΔAP | 95% CI | |
|---|---|---|---|
| `fused − esmc` | **+0.1122** | [+0.0682, +0.1555] | SIG |
| `dens − rand` | +0.1603 | [+0.0957, +0.2188] | SIG |
| `dens − dens_stats` | +0.1611 | [+0.0923, +0.2306] | SIG |
| `dens − raw` | +0.1372 | [+0.0723, +0.2003] | SIG |
| `fused_geom − esmc` | +0.1409 | [+0.1000, +0.1822] | SIG |
| **`fused − fused_geom`** | **−0.0287** | [−0.0612, +0.0025] | **ns** |
| **`fused_all − fused_geom`** | **+0.1285** | [+0.0848, +0.1688] | SIG |

**★ External cross-check.** On ProteinShake's own 6 Å Cα–Cα definition, our `esmc` arm reads
**MCC 0.2467** against the sibling project's independently measured raw-ESM interface
**MCC 0.236** — different corpus, different pipeline, same label definition. That agreement is
the strongest evidence the label and probe are implemented correctly.

Best tap by val is `up_blocks[1]` again, so the failed tap prediction replicates too.

## Next
1. `esmc_seqctx` — the decisive baseline (reuse `scripts/interface_finetune.py::train_arm`).
2. Resolution ladder on this exact label/split: ≤3 Å anchor (done) → 8 Å (`voxel_cache_lp8`)
   → one of 4/6 Å.
3. P2 pocket: re-extract the 311 ligand-bearing chains in full, or go PLINDER.
