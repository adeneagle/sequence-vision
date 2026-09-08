# How ESMFold2 extracts features from ESM-C — and what it implies for our regression

Source read: `escalante-bio/esmjfold2` (Apache-2.0 JAX/Equinox translation of the Biohub
reference), cloned and read in full — `language_model.py`, `lm_features.py`, `trunk.py`,
`triangle.py`, `inputs.py`, `confidence.py`, `model.py`, `esmc.py` — plus the released
`config.json` for `biohub/ESMFold2` and `biohub/ESMFold2-Fast`.

Our own note in CLAUDE.md (2026-08-22) covered *one step* of this pathway, the ELMo-style
layer mix, and stopped there. The layer mix turns out to be the cheapest and least
consequential quarter of it.

---

## 1. The pathway, end to end

```
ESMC-6B  (80 layers, d_model = 2560)         [lm_d_model=2560, lm_num_layers=80]
  chain-aware tokenization: [BOS] chain1 [EOS BOS] chain2 ... [EOS]
  + sequence_id  (cumsum of BOS markers, PAD -> -1)  ->  chain-aware attention
  dedup tokens by (asym_id, residue_index); scatter back via expand_map
        |
        v
hidden_states  [B, L, 81, 2560]        <- 80 PRE-BLOCK states + final POST-NORM state
        |
        |  (1) LanguageModelShim.base_z_linear = Sequential(LayerNorm(2560), Linear(2560->256))
        |      SHARED across all 81 layers; only 81 extra scalars of layer-specific capacity
        v
   [B, L, 81, d_z=256]
        |
        |  (2) weights = softmax(base_z_combine)          # one learned scalar per layer
        |      lm_z = einsum("blnd,n->bld", lm_z, weights)  # mix over the LAYER axis
        v
   SINGLE  [B, L, d_z]
        |
        |  (3) base_z_mlp = Sequential(SingleToPair, LayerNorm(256))
        |      SingleToPair:  x     = downproject(x)                     # Linear(256->256)
        |                     outer = concat([ x_i * x_j , x_i - x_j ])  # 256 + 256 -> 512
        |                     out   = output_mlp(outer)                  # Linear(512->256),
        |                                                                # GELU, Linear(256->256)
        v
   PAIR  [B, L, L, d_pair=256]          <-- this is the ONLY form in which ESM-C enters
        |
        |  (4) lm_encoder: 4 x PairUpdateBlock          [lm_encoder.n_layers=4]
        |      PairUpdateBlock = TriMul(outgoing) + TriMul(incoming) + Transition, all residual
        |      TriMul: proj_bundle Linear(256->1024) = [signal 512 | gate 512], split L/R at 256
        |              -> einsum contraction -> norm_mix -> proj_emit(256->256) * sigmoid(gate)
        |      Transition: SwiGLU, w12 256->2048, w3 1024->256   (4x expansion)
        v
   refined lm_z
        |
        |  (5) per-loop dropout p=0.25, then injected EVERY refinement loop into a
        |      diagonal linear recurrence ("parcae"):   z <- a*z + injected @ b.T
        |      with a = exp(-delta * exp(log_a)),  b = delta * b_cont
        |      num_loops = 3 at inference; randomized Poisson(3) in [1,6] during training
        v
   folding_trunk: 48 x PairUpdateBlock  (24 in -Fast)   ->  z   [B, L, L, 256]
        |
        +-> distogram_head( z + z^T )                       # symmetrized pair -> distance bins
        +-> DiffusionStructureHead(z_trunk=z, s_inputs=x_inputs)
        +-> ConfidenceHead: ... -> RowAttentionPooling(z) -> per-residue pLDDT / resolved
```

## 2. The five decisions that matter to us

### (0) The whole thing is astonishingly small: ~7.2 M parameters

Read directly from the local `biohub/ESMFold2` checkpoint (safetensors header,
`model.safetensors`, 1594 tensors), and **byte-identical in `ESMFold2-Fast`**:

| component | tensors | params |
|---|---|---|
| `language_model.*` (the whole ESM-C shim) | 12 | **0.924 M** |
| `lm_encoder.*` (4 pair-update blocks) | 72 | **6.30 M** |
| total ESM-C -> refined-pair machinery | | **~7.2 M** |

`language_model.base_z_combine` has shape **[81]**, confirming 80 layers + 1 state. The
entire mechanism by which a 6 B-parameter PLM is turned into geometry is under a million
parameters, and the refinement on top is six million more. This is a strong argument for
just building it on our existing cache: the architecture, not the capacity, is the content.

### (a) ESM-C is consumed **only** as a pair representation. There is no ESM-C single track.

This is the headline. The single-track conditioning handed to the diffusion head is
`ctx.x_inputs`, built by `InputsEmbedder` from **atom-encoder features + aatype one-hot +
MSA profile + deletion_mean** — `d_inputs = 451 = 384 + 33 + 33 + 1`. ESM-C appears nowhere
in it. Its entire contribution to structure is `lm_z`, a pairwise object.

Equally, `FoldingTrunk` is **pair-only**: `TriMul(out) + TriMul(in) + Transition`. There is
no single-track update and no triangle *attention* (consistent with our existing note that
triangle attention was removed) — the triangle *multiplicative* update is what they kept.

So a 189 M-parameter structure model built on a 6 B-parameter PLM treats that PLM as a
source of **relational information about residue pairs**, not as a per-residue descriptor.

### (b) The outer product keeps a symmetric **and** an antisymmetric term

`concat([x_i * x_j, x_i - x_j])`. The product is symmetric under i<->j, the difference is
antisymmetric. Most AF-style outer products use the product/outer-product-mean alone.

Note `downproject` is a misnomer in the released checkpoints: it is `Linear(256 -> 256)`,
width-preserving, so **no channel reduction happens before the outer product**. The pair
feature is built at full single width and the concat doubles it to 512 before `output_mlp`
brings it back to 256. Worth knowing if we shrink it -- the reduction is ours to choose,
not something the reference does.

Note what our own sibling project already used, in `pairwise_contact_probe.py`:
`concat(R_i * R_j, |R_i - R_j|)`. That is the same construction, and on raw ESM it gave
**long-range contact AUC 0.812 / MCC 0.463, coordinate-free and leakage-free**. We built
ESMFold2's pair feature by accident, confirmed it carries real long-range spatial signal,
and then never used it for the density regression.

### (c) Triangle multiplication is the mechanism that converts pair features into geometry

`einsum("bikd,bjkd->bijd")` (outgoing) / `einsum("bkid,bkjd->bijd")` (incoming): pair (i,j)
is updated from all k via (i,k) x (j,k). That is a transitivity / metric-consistency prior —
if i is near k and j is near k, constrain (i,j). It is precisely the inductive bias a
per-residue readout cannot express at any depth, because it is a statement about *triples*.

### (d) Pair -> per-residue readout is `RowAttentionPooling`

```python
scores  = attn_proj(z).squeeze(-1)            # [B, L, L]  learned scalar per pair
scores += mask_bias                           # mask keys
weights = softmax(scores, axis=-1)            # over j
pooled  = einsum("bnm,bnmd->bnd", weights, z) # per-residue vector
return out_proj(pooled)
```

ESMFold2 uses exactly this to produce **pLDDT** — a per-residue local-quality scalar — out
of the pair representation (`confidence.py:108`). That is structurally the same shape of
problem as ours: a per-residue quantity describing a residue's local environment, read out
of a pair rep. It is also the same shape as Track B (per-residue local resolvability).

The confidence head is worth copying wholesale as a template, because it also shows the
reverse direction — single -> pair injection before the pooling:

```python
z += s_to_z(s)[:, :, None, :] + s_to_z_transpose(s)[:, None, :, :]
z += s_to_z_prod_out( s_to_z_prod_in1(s)[:, :, None, :] * s_to_z_prod_in2(s)[:, None, :, :] )
z += folding_trunk(z)                  # 4 pair-update blocks
single = row_attention_pooling(z)      # -> per-residue
```

### (e) Repeated injection, gated state, heavy LM dropout

The LM pair features are not fed once. They are re-injected at **every** refinement loop
through a diagonal linear recurrence, under **dropout p = 0.25** on the LM channel
specifically (`lm_encoder.lm_dropout = 0.25`, `per_loop_lm_dropout = true`). The loop count
itself is randomized during training (Poisson mean 3, clamped 1..6).

A 0.25 dropout applied only to the LM pathway is a strong statement that they had to stop
the trunk over-relying on it.

## 3. Things one might assume that are false

- **ESMFold2 does not use ESM-C attention maps.** `compute_lm_hidden_states` returns hidden
  states only, and `LanguageModelShim` accepts only hidden states. The pair rep is built by
  outer product from mixed hidden states. (ESMFold v1 did feed ESM-2 attention maps into its
  trunk — stated from memory, not re-verified here — so this looks like a deliberate change.)
- **The layer mix is not layer *selection*.** Our measured weights were nearly flat
  (1.62x spread over 36) and bought +0.017; the mix is ensembling across depth. The
  architectural content is downstream of it.
- **Mixing happens at the single level, once**, before the outer product — not per-layer pair
  reps. So the mix is cheap and the pair construction is paid once.

## 4. What this predicts about our 0.187

Current state: ESM-C per-residue [1152] -> CryoFM `up_blocks[0]` local-frame feature [512],
**R2 0.187**, and the map is essentially **linear** (MLP over ridge: +0.018; best-layer
choice: +0.017; ESMFold2-style layer mix: +0.017 more). Sequence-tracked-at-homolog-
resolution reference 0.447; pose-noise ceiling 0.849.

Two facts we have already measured, which together point at one explanation:

1. `up_blocks[0]`'s effective receptive field has **half-decay ~9-10 A** — and our own note
   concluded this is *why* that tap wins per-residue: "~10 A IS the protein contact scale,
   the neighbourhood that defines a residue's structural environment."
2. Neighbourhood -> residue prediction by **geometry-blind moment pooling** is weak (ridge
   R2 0.112 at 10 A, nonlinear gap <= 0) — and the pairwise channel is where ESM's spatial
   information actually lives (contacts AUC 0.812 vs per-residue/moment probes missing it).

So the target is a function of residue i's ~10 A spatial neighbourhood, and a readout
`f(esm_i)` has to answer "who is within 10 A of me, and how are they arranged" from residue
i's own 1152-d vector. That question is intrinsically pairwise. Head nonlinearity and layer
choice have both been tested and bought ~0.02 each; the untested lever is the one ESMFold2
says is the whole point.

**Hypothesis, falsifiable:** a large part of the 0.187 -> 0.447 gap is neighbourhood identity
and arrangement, reachable only through a pairwise channel, and a shrunken ESMFold2 shim
(outer product -> triangle multiplication -> row-attention pooling) should recover part of it.

Note the construction is *legitimate* here for the same reason it is in ESMFold2: at
inference the predictor does not know which residues are spatially near i — it must infer
that from sequence. We use coordinates only to *define the target* (the local-frame
circularity caveat already logged), never as predictor input. So all-pairs + triangle
updates is the honest construction, and its O(L^3) cost is the honest price.

## 5. Staged proposal (cheap first, controls first)

**Stage 0 — oracle-neighbourhood upper bound. CPU, ~1 h. Do this before building anything.**
Predict target_i from *ground-truth* local neighbourhood (neighbour AA identities +
distances, or shell coordination counts) and see where it saturates. If ground truth
neighbourhood reaches ~0.45, the pairwise channel is where the missing variance lives and
Stage 1 is justified. If it also caps near 0.19, the residual is not neighbourhood-structured
and no pair architecture will help — and we have saved a GPU week.
This follows our own standing rule: *estimate a quantity, do not test an invented threshold.*

**Stage 1a — outer product only, no triangle stack. O(L^2), cheap.**
Layer mix (already implemented in `probes/align_layer_mix.py::Shim`) -> `SingleToPair` with
`concat[x_i*x_j, x_i-x_j]` -> `RowAttentionPooling` -> Linear -> [512]. Isolates the gain
from the pair construction alone.

**Stage 1b — add N x PairUpdateBlock (N = 2..4) between them.** Isolates the triangle
contribution. O(L^3 * d_pair) is the real cost driver: at median L = 260 and d_pair = 64
that is ~1e9 * d ops per chain, so cap L or crop.

**Stage 2 — controls, on the existing 672-cluster / 30%-identity split.**
- `shuffled` ESM-C rows — **judge on cosine, not R2** (our own logged control bug: a working
  shuffled arm has negative R2 by construction).
- `seqwin +-3` (the standing bar, 0.037).
- **`relpos-only pair`**: run the identical pair machinery with a relative-sequence-position
  encoding but *no ESM-C in the pair channel*. Mandatory — the sibling project found >50% of
  short-range predictability is backbone adjacency, so |i-j| alone in a pair rep could
  manufacture most of a positive.
- current per-residue MLP (0.187) as the arm to beat.

**Cheap wins available regardless:**
- Our layer cache omits ESM-C's final **post-norm** state (we store 36 pre-block states;
  ESMFold2's 81 = 80 pre-block + 1 post-norm). Prior measurement says this costs ~nothing
  (L35 pre-norm 0.167 vs post-norm final 0.169), so this is a completeness fix, not a gain.
- Chain-aware tokenization: ESMFold2 embeds *all chains in one forward* with BOS/EOS
  separators and `sequence_id`; `extract_esmc_chains.py` embeds one chain in isolation. This
  is exactly the multi-chain/homo-oligomer mismatch already flagged as a dataset defect
  ("14 of 27 entries have n_unique_seq > 1"). Worth aligning if we ever move off single chains.
- LM-channel dropout ~0.25 as a default regularizer on any ESM-C -> target projection.

## 6. Unknowns / things I did not verify

- ~~`SingleToPair.downproject` width and `base_z_linear`'s `d_z` are unknown.~~
  **RESOLVED** against the local checkpoint caches at
  `/mnt/main0/projects/es/rrao/.hf-cache/hub/models--biohub--ESMFold2{,-Fast}`:
  `d_z = 256` (`base_z_linear.1.weight [256, 2560]`), `downproject = [256, 256]`,
  `output_mlp.0 = [256, 512]` (confirming the 2x concat), `output_mlp.2 = [256, 256]`.
  Identical in both checkpoints.
- **Whether LM dropout is genuinely active at inference is contradictory in the sources.**
  The JAX port applies per-loop LM dropout and comments that ESMFold2 keeps it on, but the
  released config carries `force_lm_dropout_during_inference: false` alongside
  `lm_encoder.per_loop_lm_dropout: true`. Do not lean on this.
- ESMFold v1's use of ESM-2 attention maps is from memory, not re-verified.
- The claim in §4 is a hypothesis with a stated mechanism, not a measurement. Stage 0 is
  designed to kill it cheaply if it is wrong.

---

## 7. STAGE 0 RESULT (ran 2026-08-26) — the gap is DIRECTIONAL geometry, and folding beats a pair net

`probes/stage0_oracle_neighbourhood.py`, ridge, test = 52,132 residues on the **identical
cluster-level split** as the ESM-C alignment run (279,078 / 33,372 / 52,132 — matches that
run's `_meta` exactly). 1500/1500 chains, 0 skipped.

**Pipeline validation before anything else.** `seqwin3` reproduced at **0.0359** vs the
logged 0.035907 and `esmc` at **0.1694 / cos 0.3966** vs the logged 0.169379 / 0.396552 —
four decimals, so this is the same residue set and the oracle numbers are directly
comparable. Shuffled-oracle controls (features permuted against targets, refit): cosine
**+0.0009 / +0.0004 / +0.0012** for dirmom / shellcomp / full_oracle. No dead dims.

| arm | dim | R2 | cos | what it bounds |
|---|---|---|---|---|
| `seqwin3` | 140 | 0.0359 | 0.176 | reference |
| **`esmc`** | 1152 | **0.1694** | 0.397 | reference (ridge) |
| *`esmc`+MLP* | 1152 | *0.187* | *0.417* | **the arm to beat** |
| `coord` (counts only) | 9 | 0.1645 | 0.377 | burial/packing alone |
| `shellcomp` | 180 | 0.1751 | 0.390 | 1 outer product + row pool |
| `shellcomp+own` | 200 | 0.1805 | 0.397 | " + own identity |
| `shellcomp_far` | 180 | 0.1696 | 0.383 | " , |i-j|>4 only |
| `bb_only` | 16 | 0.0278 | 0.149 | backbone dihedrals / SS |
| **`dirmom`** | 81 | **0.4009** | 0.635 | frame-directional geometry |
| `dirmom_far` | 81 | 0.3842 | 0.622 | " , |i-j|>4 only |
| `bb+dirmom_far` | 97 | 0.3958 | 0.631 | |
| `shellcomp+dirmom` | 261 | 0.4073 | 0.639 | |
| **`full_oracle`** | 446 | **0.4099** | 0.641 | all ground-truth neighbourhood |
| **`esmc+shellcomp`** | 1332 | **0.2446** | 0.477 | **true Stage-1a ceiling** |
| `esmc+bb` | 1168 | 0.1791 | 0.409 | |
| `esmc+full_oracle` | 1598 | 0.4167 | 0.645 | |

### Findings

1. **Neighbour IDENTITY is worthless; geometry is everything.** `coord` — nine numbers,
   neighbour counts per radial shell — scores 0.1645, i.e. essentially all of what ESM-C's
   full 1152-d embedding delivers (0.1694). Adding 180 dims of per-shell amino-acid
   composition takes it only to 0.1751, and stacking those same 180 dims on top of `dirmom`
   buys **+0.006**. The one thing a pair channel is uniquely good at carrying — *who* my
   neighbours are — is not what this target needs.
2. **Direction beats distance-composition by 2.3x on fewer dims**: `dirmom` 0.4009 (81) vs
   `shellcomp` 0.1751 (180).
3. **It is tertiary arrangement, not secondary structure.** `bb_only` (direction+distance to
   i+-1, i+-2, i.e. the dihedrals) is **0.0278**, and `dirmom_far` (all |i-j|<=4 neighbours
   removed) retains **96%** of `dirmom`. This survives the adjacency control that halved
   short-range predictability in the sibling project.
4. **The oracle saturates at ~0.41, and ESM-C is almost entirely subsumed by it**:
   `full_oracle` 0.4099 -> `esmc+full_oracle` 0.4167, so ESM-C contributes **+0.007** once
   true local geometry is known. ESM-C is functioning as a *noisy proxy for local geometry*.
   Independent consistency check: the oracle's 0.410 sits close to the separately derived
   sequence-tracked reference 0.447.
5. **But the pairwise-reachable information IS complementary to ESM-C.** `esmc+shellcomp`
   0.2446 vs `esmc` 0.1694 — **+0.075**. The correct Stage-1a ceiling is therefore 0.245,
   not `shellcomp`'s standalone 0.175, because such a head sits *alongside* the per-residue
   readout rather than replacing it. (An earlier reading of the standalone number as
   "decisive against Stage 1a" was too strong.)

### Decision

- **Stage 1a (single outer product + row pooling): weakly justified, do not build yet.**
  Ceiling 0.245 vs current 0.187 — at most +0.058, and that ceiling assumes *perfect
  ground-truth distances*. A channel that must infer distances from sequence realises some
  fraction of it. Poor return for O(L^2) machinery and a training loop.
- **Stage 1b (triangle multiplicative updates): the headroom is real but mis-targeted.** The
  remaining variance (0.41) lives in frame-directional tertiary arrangement. Triangle updates
  are indeed the mechanism for reconstructing relative geometry from pairwise distances — but
  reconstructing per-residue relative geometry in a local backbone frame *is the folding
  problem*, and we would be rebuilding a folding trunk with 7 M parameters and no MSA.
- **=> The efficient route to the remaining variance is to FOLD, not to train a pair net.**
  `dirmom` needs only Ca coordinates and N-CA-C frames. ESMFold2 produces a structure in
  ~9.4 s for 1024 residues; computing `dirmom` from *predicted* coordinates costs milliseconds.
  CLAUDE.md already lists "ESMFold2-then-simulate-then-compute-the-feature" as the untested
  natural upper baseline; Stage 0 shows precisely why it is the right next move, since the
  features that carry the signal are exactly the ones a predicted structure supplies.

**The revised next experiment (supersedes Stage 1):** predict structures for these 1500
chains, recompute `dirmom` / `full_oracle` from the *predicted* coordinates, and measure how
much of the 0.41 survives fold error. That single run decides everything: if predicted-
structure `dirmom` lands near 0.35-0.40, the task is solved by fold-then-featurise and no
pair architecture is needed; if it collapses toward 0.19, the information is only accessible
from true geometry and *no* sequence-side model — pair net or otherwise — recovers it.

### Honest limits of Stage 0

- **`dirmom`'s strength is partly true by construction** and is not a discovery: the target is
  a CryoFM activation on a box cut in residue i's frame, so the density in that box is
  determined by where surrounding atoms sit in that frame, and `dirmom` is a coarse summary of
  exactly that. It confirms the oracle features are adequate. The decision-relevant content is
  the *contrast between arms*, not `dirmom`'s absolute value.
- The oracle uses **Ca positions + residue identity only** — no side-chain rotamers, though
  the density that generated the target used every atom. Neighbours are the observed residues
  with a complete N/CA/C backbone. Both make the oracle a **lower** bound on
  neighbourhood-determined variance, so neither can manufacture a positive.
- Ridge only. A nonlinear head on `dirmom` was not run and would likely raise 0.41 somewhat.

---

## 8. STAGE 1a / 1b RESULT (ran 2026-08-26) — the pair channel's gain is POSITIONAL, not relational

`probes/stage1_pair_head.py`, 6 arms (v1) then 5 re-run with fixed optimization (v2), one
H100 each. Head is exactly the baseline MLP with the pooled pair vector concatenated, so with
the pair channel off the model IS the 0.187 architecture; `row_pool.out_proj` and every
residual branch are zero-init, so training starts at the baseline and each `PairUpdateBlock`
is identity at init. Same metric, same 52,132 test residues.

### v2 (the run to read; `accum=48`, `warmup=50`, 200 epochs, patience 20)

| arm | pair channel | tri blocks | R2 | cos | delta vs baseline |
|---|---|---|---|---|---|
| `v2_baseline_none` | off | 0 | 0.1778 | 0.4053 | — |
| `v2_stage1a_esm` | ESM-C outer product | 0 | 0.1866 | 0.4127 | **+0.0088** |
| `v2_stage1b_esm_tri4` | ESM-C outer product | 4 | 0.1844 | 0.4080 | +0.0066 |
| `v2_stage1b_esm_relpos` | both | 4 | 0.1866 | 0.4096 | +0.0088 |
| **`v2_stage1a_relpos_ctrl`** | **\|i-j\| ONLY** | 0 | **0.2034** | 0.4276 | **+0.0256** |

v1 (`accum=16`, `warmup=200`) same ordering: baseline 0.1653 · esm 0.1751 · esm_tri4 0.1739 ·
esm_relpos 0.1770 · **relpos_ctrl 0.1968** · relpos_ctrl+tri4 0.1979.

### 1. THE PRE-REGISTERED CONFOUND CONTROL FIRES, IN BOTH REGIMES

A pair channel fed **only relative sequence position, with no ESM-C whatsoever** beats the
ESM-C outer product by ~3x (+0.026 vs +0.009 in v2; +0.032 vs +0.010 in v1). Replicated
across two optimization regimes with different baseline quality (0.165 vs 0.178).

**The mechanism is analytic, not speculative.** With the pair channel fed only relpos,
`z_ij = W . onehot(clamp(i-j))` depends solely on the clamped offset, so the pooling scores
`attn_proj(z_ij)` are content-free as well. The softmax runs over `j in [0,L)`, and the
multiset of offsets available to row i depends only on **i and L**. Hence
`pooled_i = g(i, L)` exactly — a learned function of position-in-chain and chain length,
carrying zero sequence information. Sequence content still reaches the head only via `xs`,
as in the baseline. So that arm is precisely *baseline + a learned positional/terminus
feature*.

Why it works: residues near a chain terminus have systematically fewer neighbours, and
Stage 0 showed `coord` (9 shell counts, essentially burial) alone scores 0.1645. Distance
from the terminus is a crude burial proxy, and burial is most of what this target responds to.

**Without this control the honest-looking headline would have been "+0.009 from the ESM-C pair
channel."** It is not a relational result.

### 2. Triangle multiplicative updates add NOTHING — but this is DATA-limited, not a refutation

0.1844 with 4 blocks vs 0.1866 without (and 0.1739 vs 0.1751 in v1) — slightly negative both
times. **Do not read this as "triangle updates cannot reach `dirmom`".** Every arm reaches
best val at **epoch 4 of 25** (~94 optimizer steps) and then overfits hard (train 1.04 ->
0.48 while val rises). With only **1128 training chains**, a 0.7 M-param pair channel cannot
learn geometric reasoning — and it has no geometric supervision to learn it from, since the
only signal is a 512-d feature regression. ESMFold2 trains its trunk on orders of magnitude
more structures *with an explicit distogram + structure loss*. The correct conclusion is:
**you cannot get triangle-update geometry from 1128 chains and a feature-regression
objective**, which is a statement about our data and objective, not about the architecture.

(v1 additionally had a genuine bug worth recording: `warmup=200` steps ~= 3 epochs while best
val arrived at epoch 2-4, so the LR peaked after the model was already done and the zero-init
blocks never moved. Fixed in v2; the ordering was unchanged.)

### 3. Nothing approaches the Stage 0 ceilings

Best pair arm **0.1866** vs Stage-1a ceiling **0.2446**, `dirmom` **0.4009**, `full_oracle`
**0.4099**. The ESM-C pair channel realises ~13% of the +0.067 headroom Stage 0 measured for
it. Stage 0 called Stage 1a "weakly justified"; the measurement agrees.

### 4. The one actionable win is free and is not the pair network

`v2_stage1a_relpos_ctrl` (0.2034) is the best number in the whole Stage 1 experiment and
exceeds the original 0.187 reference. Since its pooled feature is provably `g(i, L)`, the
entire effect is reproducible by feeding the per-residue head a handful of cheap positional
scalars — `i`, `L`, `i/L`, `min(i, L-i)` — with **no pair representation at all** (the
relpos pair channel is 0.01 M params and O(L^2) purely to compute a function of two integers).
**Recommended: add terminus/length features to the per-residue readout and drop the pair
machinery.** Worth ~+0.026 in-harness for essentially zero cost.

### Methodological notes

- Val and test orderings agree exactly across all five v2 arms, so model selection is clean
  and none of this is test-set fitting.
- The v1 baseline (0.1653) undershot the published 0.187 because the original `mlp_r2` samples
  4096 residues IID across ~1000 distinct proteins per batch, whereas a pair model must batch
  whole chains (16 proteins/step at `accum=16`). Raising to `accum=48` recovered most of it
  (0.1778). **A pair architecture is intrinsically handicapped on gradient diversity** at
  fixed residue budget — a real, rarely-stated cost of moving from per-residue to pairwise.

---

## 9. BETWEEN- vs WITHIN-PROTEIN (ran 2026-08-27) — 20% of the target variance is protein-level

Every R2 above uses one GLOBAL train-mean baseline, so between-protein variance sits in the
denominator and a model that only got each protein's average density character right would
score above zero without resolving any residue. `probes/stage0_within_protein.py` quantifies
it. Same 52,132 test residues / 234 test proteins; **all `global` values reproduce their
Stage 0 counterparts exactly**, which validates the (independent, ordered) loader.

| arm | dim | global | within | **centred** |
|---|---|---|---|---|
| `seqwin3` | 140 | +0.0359 | −0.2118 | +0.0444 |
| `coord` | 9 | +0.1645 | −0.0502 | +0.1191 |
| `esmc` | 1152 | +0.1694 | **−0.0441** | **+0.1434** |
| `shellcomp` | 180 | +0.1751 | −0.0369 | +0.1280 |
| **`dirmom`** | 81 | +0.4009 | **+0.2469** | **+0.4131** |
| `full_oracle` | 446 | +0.4099 | +0.2582 | +0.4201 |

`global` = vs global train mean (the headline metric). `within` = vs a **per-protein-mean**
baseline; negative means the model loses to a predictor handed each test protein's own mean
(the model never sees it, only the baseline does). `centred` = both X and Y per-protein
centred and refit: pure within-protein predictability.

**between-protein fraction = 0.2044.** Equivalently, an oracle knowing ONLY each protein's
mean target vector scores global R2 **0.2044** — *above* ESM-C's 0.1694.

### What this corrects

1. **ESM-C's headline is partly protein-level, and it loses to the trivial oracle.**
   `r2_within = −0.044`: knowing merely which protein you are looking at beats ESM-C's full
   per-residue prediction. The 0.187/0.169 figures are inflated by the 20% cross-protein
   component.
2. **But ESM-C is NOT null per-residue** — `centred` = **+0.1434**. Real resolving power; the
   headline was mislabelled, not fabricated.
3. **CORRECTION to a Stage 0 claim (§7, finding 1).** "Nine numbers of neighbour counts match
   ESM-C's entire 1152-d embedding" is true GLOBALLY (0.1645 vs 0.1694) but **false
   within-protein**: `esmc` **0.1434** vs `coord` 0.1191 vs `shellcomp` 0.1280. The apparent
   equivalence was largely the shared protein-level component, which a 9-dim burial profile
   captures as well as a 1152-dim embedding. ESM-C does carry per-residue information that
   simple counts do not; §7 understated it.
4. **The central conclusion SURVIVES and STRENGTHENS.** `dirmom` beats the per-protein-mean
   oracle decisively (**+0.247**) and its centred score (**0.4131**) EXCEEDS its global score
   (0.4009) — protein-level variance was diluting it, the opposite of ESM-C. The
   geometry-vs-ESM-C gap widens under the fairer metric: **0.413 vs 0.143 (~2.9x)** versus
   0.401 vs 0.169 (2.4x) globally. So the missing signal is genuine residue-level structure,
   which is exactly what a predicted structure supplies.
5. **`seqwin3` is the cleanest illustration** of the confound: global +0.036 but within
   **−0.212**, i.e. a +-3 sequence window is far worse than useless once protein identity is
   accounted for.

### Caveat that now propagates

The reference points quoted throughout this file — sequence-tracked **0.447** and pose ceiling
**0.849** — were computed against global means too, so they inherit the same 20% inflation and
are NOT directly comparable to `centred` numbers. Comparisons *within* a column are valid;
comparisons *across* columns are not. Re-deriving those two references per-protein has not
been done.
