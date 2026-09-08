# Candidate goals for `sequence_vision` — decision register

Written 2026-08-27, after the ESMFold2-pathway / Stage 0 / Stage 1 work
(see `ESMFOLD2_LM_PATHWAY.md`) surfaced a scoping problem.

## Why this file exists

**The stated goal has been met.** `PLAN.md`: *"can ESM-C sequence embeddings be aligned to the
internal features of a cryo-EM density foundation model."* `PLAN_PHASE0_CROSSMODAL.md`: *"the
success bar is **alignability at all** — a representation-science result, not a requirement to
beat any baseline or power a task."* That bar was cleared on 2026-08-22 (R2 0.187, controls
hold). The project has no live objective, which is why subsequent work drifted into optimising
a proxy.

**And the proxy is degenerate, not merely underdetermined.** `R2(ESM-C -> f(density))` is
monotonically improvable by making `f` LESS informative; in the limit `f = const`, `R2 = 1`.
There is no interior optimum. This has already been measured twice in this project and misread
once: `trivial [N, Rg, AA comp]` scored a homolog-diagnostic ceiling of **0.629** vs CryoFM's
**0.145**, and the hand-crafted SH descriptors won that diagnostic at **0.677** and were deleted
2026-08-26 for exactly this reason ("a low-dim hand-crafted summary can win it by being an
easier regression target"). The feature family was removed; the objective that rewards it was
kept.

**Third problem, structural.** For SIMULATED density the target is a deterministic function of
the atomic model (`coords -> DensityCalculatorE -> volume -> CryoFM -> activation`; coordinates
are the only input). So `sequence -> CryoFM feature` is strictly dominated by
`sequence -> structure -> CryoFM feature` for EVERY choice of (t, layer, frame, CleanDIFT); the
only gap is folding error. Stage 0 corroborates: coarse ground-truth geometry gives 0.410 vs
ESM-C 0.169, and ESM-C adds **+0.007** once geometry is known.

A live goal therefore needs an **external metric not of our choosing**. Then the free
transformations on BOTH sides (ESM-C layer/mix/head; CryoFM t/layer/framing/CleanDIFT) become
hyperparameters rather than objectives.

---

## O0 — Fixed-target alignment: ESM-C -> CryoFM features  **[DEPRECATED]**

What we were doing. Retire it: goal already met, objective degenerate, and on simulated density
provably redundant with folding. Retain the *findings* (they are informative about ESM-C and
about the density target) but stop treating R2 against a chosen feature configuration as a
figure of merit.

## O1 — Is CryoFM2 any good? A density-side representation benchmark  **[RECOMMENDED NEXT]**

**The logically prior question, never asked.** `PLAN.md` promised *"Teacher choice is open;
CryoFM is the starting candidate but gets validated before anything is built on it."* What was
actually validated is **stability** (pose invariance 0.83, random-weight control passes 4x).
**Usefulness was never validated** — and per CLAUDE.md, *"No feature API, no representation
benchmark in either paper — whether the activations are usable at all is the open question."*
We built an alignment target on a teacher never shown to be good for anything.

**Metric is external and standard**: classification accuracy / MCC on density-side tasks. No
sequence side needed at all, so it cannot inherit O0's degeneracy.

**Labels already exist on disk** — Cryo2StructData ships per-voxel label maps for ~7,361
entries (verified present in 300/300 sampled): `amino_*` (20-class), `sec_struc_*`, `atom_*`,
`atom_ca_*`, plus `helix/strand/coil.pdb`.

Task ladder:
| | task | label | external reference |
|---|---|---|---|
| A | per-residue secondary structure (3-class) | `sec_struc_*` | Cryo2Struct / DeepTracer / ModelAngelo |
| B | per-residue amino-acid identity (20-class) | `amino_*` | ModelAngelo ~49% top-1 at 4-5 A (our corpus is <=3 A, so its bar is higher there) |
| C | Ca / backbone voxel detection | `atom_ca_*`, `atom_*` | backbone-tracing tools |
| D | per-protein fold class (SCOP/CATH) | external annotation | this project's own SCOP probe |

**Mandatory controls** (all from this project's hard-won list):
* **raw density baseline** — identical probe on the raw normalised voxels, no CryoFM. If the
  features do not beat raw density, they are worthless. This is the analogue of "raw ESM is the
  bar to beat", which killed several earlier arms.
* **random-weight CryoFM** — re-run for classification; the stability version passed (4x) but
  that does not transfer.
* **class-prior / per-map-mean baselines** — §9 showed a per-protein-mean oracle beat ESM-C.
* protein-level (better: 30%-identity cluster-level) splits.

**This is also where the two-sided free-parameter problem gets resolved**: sweep t, tap/layer,
framing, and CleanDIFT-vs-not against a FIXED external metric. NOTE: a separate session is
already evaluating feature characteristics vs timestep (CLAUDE.md: "Do not duplicate it") —
coordinate on the t axis rather than re-running it.

**Kill criterion**: if CryoFM2 features do not beat raw density voxels on A/B/C, the teacher is
not worth aligning to, and every CryoFM-dependent option below collapses.

**Cost**: modest. Feature extraction dominates; probes are linear/MLP.

## O2 — Track B: sequence -> experimental observability  **[LOGGED]**

Predict per-residue observables a static predicted structure cannot supply: *which parts of my
protein will be resolvable by cryo-EM.* CLAUDE.md's utility analysis already identified this as
"the branch that survives" and noted its deprioritisation "was the wrong call."

Target ladder (labels verified on disk: SEQRES present, B-factors richly populated with 19,739
distinct values in the sampled entry — real ADPs, not placeholder or smuggled pLDDT):
| | target | source |
|---|---|---|
| T1 | binary: was the residue **modeled at all** | SEQRES vs observed ATOM records |
| T2 | per-residue **B-factor / ADP** | PDB B-factor column (needs per-entry normalisation; refinement protocol varies) |
| T3 | per-residue **map-model agreement** (Q-score / local CC) | map + model, both on disk |
| T4 | per-residue **local resolution** | ResMap/MonoRes-style maps, sparsely deposited |

**Why not folding**: a map is an ensemble average under an instrument, not one conformer.
Resolvability is set by conformational heterogeneity, occupancy, packing rigidity, and
instrument/processing factors; only packing is well-determined by a static structure.

**Why not pLDDT** (the incumbent, currently misused for this): pLDDT is *epistemic* (confidence
in its own prediction), resolvability is *physical*. They come apart at (i) domain-level
mobility — pLDDT is a predicted *local* lDDT so it structurally cannot express "this rigid
domain wobbles relative to that one", the largest non-redundant signal; (ii) novel folds —
well-ordered but low-pLDDT from lack of homologs; (iii) partial occupancy, which has no pLDDT
analogue.

**Baselines that must fall**, hardest first: per-map mean; **PAE-derived inter-domain mobility**
(strongest, most likely to kill it); pLDDT; burial/RSA from a predicted structure; sequence-only
disorder predictors (IUPred3, metapredict).

**Define on the WITHIN-map component.** Local resolution is dominated by the map's overall
resolution, so a model could "succeed" by predicting global map quality from protein size —
precisely the confound §9 measured (20% of the density-feature variance was protein-level).

**Risk, and it is real**: on experimental maps the pooled homolog diagnostic collapsed for every
density descriptor, and CLAUDE.md notes the experimental-only variation "is not sequence-
predictable, because it is sample- and session-specific." Track B is viable only for the
*systematic, structure-linked* share.

**Cheap gate before building**: variance-decompose T1/T2 into structure-linked /
protein-specific / session-specific. If the structure-linked-but-not-pLDDT-predictable share is
small, Track B dies for a few CPU hours.

**Note**: Track B may need **no density foundation model at all** — it could be a supervised
sequence->observable task. That is a real reframing of the project's identity and should be
chosen deliberately, not drifted into.

## O3 — Two-tower sequence<->density retrieval  **[LOGGED, weaker]**

CryoDomain-style (AAAI-25: SCOPe top-1 57.5% at 5-10 A). Metric = retrieval accuracy, external;
features learned jointly, so no degeneracy. Weaknesses: crowded (ModelAngelo already does map ID
at <=4 A; CryoDomain covers density<->structure), and homo-oligomers make sequence->density
one-to-many, so it must be contrastive rather than regression.

---

## Dependency

**O1 gates O0/O3 and the CryoFM-flavoured version of O2.** Run O1 first — it is the cheapest and
it is the question the plan promised to answer. If CryoFM2 features fail to beat raw density,
the surviving option is O2 *without* CryoFM.

---

## O1 RESULT (2026-08-27) — the alignment target loses decisively to RAW VOXELS

`probes/o1_cryofm_benchmark.py`, REAL experimental maps, 1484/1500 chains, 55,583 train /
11,332 test residues, 502/100 clusters (existing 30%-identity cluster split).
Tap `up_blocks[0]`, t=10, local N-CA-C frame — **the exact configuration the alignment project
used as its target**.

| task | prior | **raw 8^3 voxels** | cryofm | random |
|---|---|---|---|---|
| secondary structure (3-cls) | 0.4409 | **0.7506** / F1 0.7334 | 0.5882 / 0.5564 | 0.5306 / 0.4900 |
| amino-acid identity (20-cls) | 0.0880 | **0.2780** / F1 0.2189 | 0.1125 / 0.0806 | 0.0955 / 0.0665 |

* **Raw voxels beat the pretrained feature by 16 points on SS and 17 on AA**, at identical
  dimensionality (512), identical frame-aligned box, identical locality. The kill criterion
  fired.
* **On AA the feature is nearly uninformative**: +2.4 points over the majority prior, vs +19
  for raw voxels. Pretrained beats random by only 1.7 points here (5.8 on SS) — far short of
  the 4x pretrained/random gap measured for pose stability.
* **The raw baseline is sane**, so the task is learnable and CryoFM is the outlier: ModelAngelo
  reports ~49% top-1 AA at 4-5 A with a full trained network plus graph refinement; 27.8% from
  a LINEAR probe on 512 raw voxels at <=3 A is the right ballpark.

### Mechanism — the tradeoff the project optimised the wrong end of

`up_blocks[0]` was selected as the alignment target because it maximised **pose stability**
(0.826, the best of any per-residue construction). Pose stability at fine scale is bought by
**spatial smoothing** — this file's own measured half-decay for `up_blocks[0]` is ~9-10 A, while
amino-acid identity needs 1-2 A side-chain detail. The property that made the tap the best
alignment target is precisely what destroys its ability to resolve local chemistry. Usefulness
was never measured, so the tradeoff was never seen.

### Scope: decisive about the TARGET, not yet about the TEACHER

One asymmetry to keep attached: `raw` is 512 *spatially distinct* voxels over a 12 A cube at
1.5 A; `cryofm` is 512 *channels at a single point*. Dimensionality is matched, spatial
information is not, and both tasks are local geometric/chemical patterns where explicit layout
helps. So this condemns the single-`up_blocks[0]`-token target soundly, but a verdict on
"CryoFM2 features" needs the finer taps (`up_blocks[2]`, `up_blocks[3]`, `conv_in` — all
1.5 A/token) and/or multi-token patches. **Tap sweep launched against this same fixed external
metric**, which is what O1 was designed to enable.

### Does the Procrustes/rotation finding change the O1 verdict? NO — measured (2026-08-27)

CLAUDE.md's relational result (per-residue cos 0.064 -> held-out Procrustes 0.820 for
`up_blocks[0]`) says rotating the input largely applies a global ORTHOGONAL Q to feature space
rather than destroying information. Two reasons that cannot rescue O1:

1. **O1 incurs no Q.** The relational table was measured in the GLOBAL framing (rotate
   molecule, forward whole map, resample). O1 cuts each box in the residue's own N-CA-C frame,
   so every residue is canonicalised by construction and nothing is rotated.
2. **The probe absorbs any Q anyway.** L2 logistic regression is EXACTLY invariant to X -> XQ
   for orthogonal Q (take w' = Q^T w: identical loss, identical norm). Measured on the cached
   `up_blocks[0]` features with a Haar-random Q (orthogonal to 1.3e-15):

   | | as-is | @ random orthogonal Q | delta |
   |---|---|---|---|
   | with StandardScaler (what O1 does) | 0.5882 | 0.5885 | **0.00026** |
   | without scaler (pure L2 logistic) | 0.5886 | 0.5884 | **0.00018** |

   Per-dimension standardisation is formally NOT Q-invariant, so exact invariance is broken in
   the implementation -- but only at the 3rd decimal. **Procrustes alignment cannot improve a
   linear probe.**

**Where the finding DOES matter (unexploited):**
* **A relational readout is the fairer test.** The quantities preserved under rotation are all
  INTER-residue (`up_blocks[0]`: CKA 0.706, RSA 0.658, kNN 0.726). O1's probe classifies each
  residue independently from its own vector and never touches them, so it is structurally blind
  to the information those metrics certify. A kNN probe is Q-invariant by construction and
  reads exactly that metric structure.
* **It is the strongest argument for DROPPING THE LOCAL FRAME.** Local frames need a fitted
  atomic model, which is the deployment blocker (circular on an unknown map). A Q-invariant
  readout (relational, or per-protein Procrustes-aligned) permits GLOBAL-frame extraction and
  removes that dependency. Still unacted on; independent of how O1 resolves.
* It does NOT touch the structural argument above: on simulated density the target is a
  deterministic function of the atomic model, so sequence->feature stays dominated by
  fold-then-featurise however features are extracted or aligned.

### O1 TAP SWEEP (2026-08-27) — CORRECTION: the alignment target was nearly the WORST tap

Same protocol, real maps, 1484/1500 chains, 55,583 train / 11,332 test residues, 502/100
clusters. `raw` = central 8^3 raw voxels of the identical box (512 dims), constant across taps.

| tap | A/token | dim | SS cryofm | SS random | AA cryofm | AA random |
|---|---|---|---|---|---|---|
| `conv_in` | 1.5 | 64 | 0.6202 | 0.6205 | 0.1640 | 0.1644 |
| `up_blocks[3]` | 1.5 | 64 | 0.6296 | 0.6877 | 0.1631 | 0.1725 |
| **`up_blocks[2]`** | 1.5 | 128 | 0.6946 | 0.6697 | **0.2193** | 0.1724 |
| **`up_blocks[1]`** | 3.0 | 256 | **0.7183** | 0.6044 | 0.1706 | 0.1154 |
| `up_blocks[0]` **<- the alignment target** | 6.0 | 512 | 0.5882 | 0.5306 | 0.1125 | 0.0955 |
| `mid_block` | 12.0 | 512 | 0.5863 | 0.5236 | 0.1058 | 0.0935 |
| **`raw` voxels** | 1.5 | 512 | **0.7506** | — | **0.2780** | — |
| prior | | | 0.4409 | | 0.0880 | |

**CORRECTION to the previous entry.** "CryoFM2 features lose to raw voxels by 16-17 points" is
TAP-SPECIFIC, not intrinsic. `up_blocks[0]` is nearly the worst available tap on both tasks
(only `mid_block` is worse). At the best tap the deficit is **3.2 points on SS** (0.7183 vs
0.7506) and **5.9 on AA** (0.2193 vs 0.2780), with **2-4x fewer dimensions** (256 / 128 vs 512).
Revised verdict: **"no better than raw density, but more compact"** — not "worthless". The kill
criterion as written (must beat raw voxels) is still unmet, but this is a far weaker negative.

**Optimal depth tracks the PHYSICAL SCALE of the task** — a coherence check that a
mis-specified pipeline would not produce: secondary structure peaks at `up_blocks[1]`
(3 A/token; helix pitch 5.4 A), amino-acid identity at `up_blocks[2]` (1.5 A/token;
side-chain detail). And the pretrained/random gap is LARGEST at those same two taps
(SS +0.114 at up_blocks[1]; AA +0.047 at up_blocks[2]), i.e. learning helps most exactly where
the features are most useful.

**Two controls behaved exactly as theory demands, which is why the table is trustworthy:**
* `conv_in`: pretrained == random to 3-4 decimals on BOTH tasks (0.6202/0.6205, 0.1640/0.1644).
  Correct — it is the first convolution, so trained weights cannot add anything over a random
  projection of local density.
* `up_blocks[3]`: pretrained WORSE than random on both (SS 0.6296 vs 0.6877; AA 0.1631 vs
  0.1725). Independently reproduces this file's relational finding that `up_blocks[3]` is
  "pointwise stable, relationally empty" (CKA 0.420, kNN 0.283, spatial 0.23) — near the output
  the representation collapses toward the co-rotating velocity field. Two unrelated
  methodologies agreeing.

**Why the alignment project picked the worst tap.** `up_blocks[0]` was selected for maximal
per-residue POSE STABILITY (0.826). Pose stability at fine scale is bought by SPATIAL SMOOTHING
(measured half-decay ~9-10 A), which is precisely what destroys locally-decodable chemical and
fold detail. The selection criterion was in direct tension with usefulness, and the tension was
invisible because usefulness was never measured.

**Still open (the largest remaining confound):** `raw` gets 512 *spatially distinct* voxels over
a 12 A cube; the CryoFM arms get channels at a SINGLE point. Dimensionality is matched or
favours raw, but spatial information is not. A token-patch arm at `up_blocks[1]`/`up_blocks[2]`
is what would settle whether raw density genuinely wins.

### RELATIONAL (kNN) READOUT (2026-08-27) — does NOT rescue the features; it favours raw voxels MORE

Motivated by the Procrustes/relational finding: a kNN classifier is Q-invariant by construction
and reads the inter-residue metric structure that a per-residue linear decode is blind to.
k=15, centred cosine (the metric used in the relational analysis), same split.

| tap | SS raw | SS cryofm | SS random | AA raw | AA cryofm | AA random |
|---|---|---|---|---|---|---|
| `up_blocks[0]` | 0.7289 | 0.4735 | 0.4473 | 0.1691 | 0.0828 | 0.0836 |
| `up_blocks[1]` | 0.7289 | **0.6231** | 0.4992 | 0.1691 | 0.0891 | 0.0873 |
| `up_blocks[2]` | 0.7289 | 0.5778 | 0.6204 | 0.1691 | 0.1095 | 0.1119 |
| *(prior)* | | *0.4409* | | | *0.0880* | |

* **The relational readout is WORSE for CryoFM, not better.** Best-tap SS deficit vs raw widens
  from 3.2 points (linear) to **10.6** (kNN). Raw voxels hold up under kNN (0.7289 vs 0.7506
  linear), so this is not a weak probe.
* **On AA, CryoFM never separates from its own random-weight control at ANY tap**
  (0.0891/0.0873, 0.1095/0.1119, 0.0828/0.0836) — all at or near the 0.0880 prior.
* At `up_blocks[2]` pretrained is WORSE than random under kNN (0.5778 vs 0.6204) while BETTER
  under linear decode (0.6946 vs 0.6697): what the trained weights add there is
  linearly-decodable but not metric.

**The conceptual correction this forces.** This file's relational metrics (`up_blocks[0]`: CKA
0.706, kNN 0.726) measure PRESERVATION UNDER ROTATION — whether the same residues retain their
mutual relationships across two views of one structure. That is NOT the same property as the
metric structure aligning with meaningful labels. A feature space can have highly
rotation-stable relational geometry that does not encode secondary structure or amino-acid
identity, and that is exactly what these numbers show. **"Relational information is preserved"
!= "relational information is task-relevant"**, and only the second bears on utility. The
relational table should not be read as evidence of feature quality.

### TIMESTEP / COUPLING SWEEP (2026-08-27) — t matters MORE than the tap, but SCALE-DEPENDENTLY

`up_blocks[0]`, real maps, 1484 chains, `--per-chain 40` (so prior/raw differ slightly from the
tap sweep: compare WITHIN this table only). `d` = decoupled (clean input, told t — what every
prior number in this project used); `c` = coupled (x_t = (1-t/1000)x_0 + (t/1000)eps, the
trained pairing).

| setting | SS (prior 0.4339) | AA (prior 0.0890) |
|---|---|---|
| `t10d` **<- the operating point of every prior result** | 0.5960 | 0.1140 |
| `t100d` | 0.6563 | 0.1213 |
| `t261d` | 0.6832 | 0.1297 |
| **`t500d`** | **0.7133** | **0.1341** |
| `t100c` | 0.6643 | 0.1228 |
| `t261c` | 0.6992 | 0.1249 |
| `random` | 0.5070 | 0.0890 |
| **`raw` voxels** | **0.7379** | **0.2709** |

* **Monotone in t, and t=10 was a bad choice.** SS +11.7 points from t=10 -> t=500; at t=500 the
  WORST tap comes within 2.5 points of raw voxels (0.7133 vs 0.7379) versus a 14-point deficit
  at t=10. Magnitude comparable to the whole tap axis (+13 points), so the two compose.
* **But the gain is SCALE-DEPENDENT, and this vindicates the docstring warning.**
  `centre_features` warns that sweeping t decoupled "cannot distinguish a genuinely better
  representation from the network being pushed into its coarse-structure regime, which yields
  smoother features". SS (coarse; helix pitch 5.4 A) gains **+11.7**; AA (needs 1-2 A
  side-chain detail) gains only **+2.0** and stays at HALF of raw voxels (0.1341 vs 0.2709).
  So the SS gain is substantially task-scale matching, not a general representational win.
* **Coupled ~= decoupled** (slightly better for SS at matched t: 0.6643 vs 0.6563, 0.6992 vs
  0.6832). So the off-manifold extraction used throughout was NOT a significant error — one
  fewer confound.
* **Every CryoFM number in this project predates this and sits at t=10**, i.e. the worst point
  on this axis. Prior pose-stability/relational conclusions were all measured there.

---

## O4 — CryoFM2 features as an implicit steerable field (SO(3) irrep decomposition) **[LOGGED]**

Proposal (user brief, 2026-08-27): rotating the input changes per-voxel features but the change
is undone by one global orthogonal matrix in channel space, `f(R.x)(Rv) ~ Q_R f(x)(v)`. If
`R -> Q_R` is a homomorphism, channel space carries a rep of SO(3) and decomposes into irreps;
the blocks are the physically meaningful features.

**The theory is sound and SHARPER than stated.** `out_channels: 1` (verified in
`weights/cryofm-v2/cryofm2-pretrain/config.yaml`), so the velocity field is a SCALAR field.
Both endpoints are pinned to the trivial rep — input density scalar, output velocity scalar —
and only the INTERIOR is free to carry nontrivial irreps. Prediction: angular bandwidth peaks
mid-network, collapsing at both ends.
**This retro-explains the unexplained U-shape in the per-layer table** (`conv_in` spatial 0.06;
`up_blocks[3]` CKA 0.420 / kNN 0.283 "relationally empty"; `mid_block` / `up_blocks[0]`
relationally rich), and explains the O1 tap-sweep finding that `up_blocks[3]` PRETRAINED IS
WORSE THAN RANDOM on both tasks — near the output the representation is squeezed back toward
the trivial rep, discarding the higher-ell content that makes features useful.

**Existing partial evidence:** held-out Procrustes 0.820 at `up_blocks[0]` (in-sample 0.897),
i.e. ~18% of feature variance not explained by a global orthogonal Q.

### Three corrections before running it

1. **Task 1 CANNOT use the O1/alignment caches.** They are cut in local N-CA-C frames, so the
   feature distribution is deliberately canonicalised and NOT rotation-invariant; `C` would not
   commute with `Q_R` and the degeneracy test returns a FALSE NEGATIVE. Needs LAB-frame voxel
   samples (new, modest extraction). **Random-weight null is mandatory** — any 512-dim
   covariance has accidental tail near-degeneracies, and without the null (2ell+1) structure is
   not distinguishable from spectrum shape. That control has reversed conclusions twice here.
2. **Do NOT estimate generators by finite differences.** `L_a ~ (Q_{d,a} - Q_{-d,a})/2d`
   divides a small noisy difference by a small number, and each `Q` carries ~18% residual;
   errors compound through the commutator and again through the Casimir. **Instead:** hypothesise
   a budget `(m_0, m_1, ...)` with `sum m_ell (2ell+1) = C`, and fit ONE orthogonal basis `U`
   minimising `sum_R || U^T Q_R U - (+)_ell D^ell(R) ||_F^2` over many sampled `R` with known
   Wigner `D^ell`. Well-posed (Stiefel), differentiates nothing, and directly yields the
   per-budget goodness-of-fit that the multiplicity step needs.
3. **Chirality: the likely branch is that Q does NOT extend to O(3).** Mirroring gives
   D-amino-acid density, which is OOD — the model never saw it. Failure to extend IS the
   positive result (handedness encoded). Test as a discontinuous jump in homomorphism residual
   when improper elements are added, not as a clean pseudoscalar block.

### Corroboration from the O1 sweeps (2026-08-27)

* **Predicted bandwidth profile is weakly confirmed**: "low ell at high noise" matches the
  t-sweep, where the COARSE task gains +11.7 from t=10->500 while the FINE task gains +2.0.
* **All existing Procrustes/relational numbers are at `up_blocks[0]`, t=10** — now measured to be
  near-worst on BOTH axes, while `up_blocks[1]` at t=261 coupled BEATS raw voxels (0.7839 vs
  0.7379). **Start the equivariance analysis at `up_blocks[1]`, t~261, coupled**, where the
  residual may be far below 18%.

### Why it is worth doing: Task 5's local canonical frames

A frame from ell=2 (axis) + ell=1 (sign) would solve the ATOMIC-MODEL CIRCULARITY that is this
project's real deployment blocker (local frames need a fitted model, so they are circular on an
unknown map). Prior negative to respect: HAND-CRAFTED density frames fail — structure tensor
~30 deg axis error, SH dipole ~24 deg optimally smoothed, with a synthetic control at 0.0 deg
proving the code correct and the data at fault. A LEARNED ell=1/ell=2 block is a different
proposition and is the one route that could revive per-residue deployment.

### Revised order
0. Re-measure held-out Procrustes at `up_blocks[1]`/t=261c vs `up_blocks[0]`/t=10. If the
   residual does not improve at the good operating point, the foundation is weak — cheapest
   possible gate.
1. Task 1 on LAB-frame features + random-weight null.
2. Budget + basis fit (above), not generators.
3. The sharp validation: helix axis / sheet normal must be linearly decodable from ell=2 and
   NOT from ell=1 (a headless axis is rank-2). Dimension-match the probes — ell blocks differ
   in width (3m_1 vs 5m_2) and capacity would otherwise confound.
4. Then Task 5 frames.

**Scope:** O4 bears on O1 (feature quality) and on the deployment blocker. It does NOT touch the
structural argument that sank O0 — on simulated density the target is a deterministic function
of the atomic model, so sequence->feature stays dominated by fold-then-featurise however
features are decomposed.

### O4 STEP 0 RESULT (2026-08-27) — the residual does NOT improve at the good operating point

`probes/layer_relational_invariance.py` at t=10, t=261 decoupled, t=261 coupled; identical
params (24 structures x 3 rotations x 60 residues), all 10 taps. Held-out Procrustes:

| tap | t10 | t261d | t261c |
|---|---|---|---|
| `conv_in` | 0.708 | 0.708 | 0.690 |
| `down_blocks[0]` | 0.436 | 0.335 | 0.296 |
| `down_blocks[1]` | 0.666 | 0.647 | 0.643 |
| `down_blocks[2]` | 0.835 | 0.825 | 0.826 |
| **`down_blocks[3]`** | 0.852 | 0.826 | **0.857** |
| `mid_block` | 0.847 | 0.830 | 0.845 |
| `up_blocks[0]` | 0.775 | 0.777 | 0.783 |
| `up_blocks[1]` | 0.622 | 0.605 | 0.605 |
| `up_blocks[2]` | 0.395 | 0.355 | 0.447 |
| `up_blocks[3]` | 0.618 | 0.449 | 0.548 |

* **HYPOTHESIS REFUTED.** Procrustes is FLAT in t (within +-0.03, often worse). The conjecture
  that the ~18% residual was an artifact of the bad t=10 operating point is wrong. Best anywhere
  = `down_blocks[3]` **0.857**, so **~14% of the rotation relation is unexplained by a global
  orthogonal Q** — the floor every downstream task inherits, compounding through the
  homomorphism check and the Casimir.
* **DISSOCIATION 1 — the best-Q tap is NOT the best-feature tap.** Q structure peaks in the
  encoder/bottleneck (`down_blocks[2..3]`, `mid_block`, 0.83-0.86); downstream utility peaks at
  `up_blocks[1]` (O1: beats raw voxels) whose Procrustes is only **0.622**. So any irrep
  structure lives where the features are LESS task-useful. This specifically weakens Task 5
  (local frames from ell=1/ell=2 blocks would come from a weaker tap).
* **DISSOCIATION 2 — relational content improves with t, orthogonal explicability does not.**
  CKA rises markedly at several taps (`down_blocks[1]` 0.649->0.831, `down_blocks[2]`
  0.709->0.790, `up_blocks[2]` 0.521->0.675) while Procrustes is flat. Higher t makes features
  more relationally CONSISTENT under rotation without making them more explicable by one global
  orthogonal map. Only the latter is what the irrep program needs.
* Consistent with Task 1's smoke (no (2l+1) degeneracy, pretrained ~= random): if `Q_R` explains
  only ~86%, `C`'s commutation is approximate and degeneracies are smeared.

**Next cheap test, anticipated by the brief itself:** 0.857 is a ceiling UNDER THE ORTHOGONALITY
CONSTRAINT. Fit a general linear `A_R`, and/or whiten by `C^{-1/2}` first — "the rep need not be
carried in an orthonormal basis." If an unconstrained/whitened fit is materially higher, the
program is alive and the orthogonal parameterisation was the limitation; if not, ~14% is
irreducible non-equivariant junk and Tasks 2-6 are built on sand.

### O4 TASK 1 (2026-08-28) — the degeneracy test is NOT MEASURABLE; drop it

`probes/o4_irrep_spectrum.py`, lab-frame voxels, t=261 coupled, pretrained vs random-weight
null. Statistic: sort eigenvalues of `C = E[ff^T]`, group consecutive ones with relative gap
below a tolerance, count runs of length >= 3 (singletons are trivially odd and dominate, so an
odd-fraction over all runs is meaningless — corrected mid-run).

`mid_block` (59 maps, 59k voxels, 512 ch, ~115/dim), runs>=3 pretrained/random:
| tol | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| pre/rand | 18/17 | **64/5** | 16/3 | 4/3 |

`up_blocks[1]` (60 maps, 240k voxels, 256 ch, ~937/dim), runs>=3 pretrained/random:
| tol | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| pre/rand | **0/25** | 4/17 | **33/3** | 8/1 |

**THE STATISTIC REVERSES DIRECTION WITH TOLERANCE.** At `up_blocks[1]` tol 0.005 random has 25
multi-runs and pretrained has ZERO; at tol 0.02 pretrained has 33 and random 3. Which arm
"shows (2l+1) degeneracy" is decided by the threshold, so the metric is measuring EIGENVALUE
SPECTRUM SHAPE, not group structure: whichever arm has flatter plateaus at a given scale wins at
that tolerance.

**Retraction.** `mid_block`'s 64-vs-5 spike at tol 0.01 was reported here as "the first positive
hint" for the irrep hypothesis, and the mid-vs-`up_blocks[1]` contrast as "two independent
methods agreeing on depth". Neither survives: it is one tolerance of four in a metric that
flips, and the same metric at another tolerance says the opposite.

**Task 1 is mathematically sound but not measurable at finite N.** Invariance does force
multiplicities of exactly (2l+1), but "near-equal eigenvalues" needs a tolerance and estimation
noise makes the answer tolerance-determined. A usable version would need bootstrap CIs on
eigenvalues, or model selection between an odd-multiplicity staircase and a smooth null — more
work than doing the direct fit properly. **DROP Task 1; rest on Task 2** (pooled `Q_R`), which
has no free threshold: the map either composes or it does not.

### ★ O4 TASK 2 (2026-08-28) — THE CORE CLAIM IS FALSE: Q_R is STRUCTURE-DEPENDENT

`probes/o4_pooled_qr.py`. 80 structures (48 fit / 32 HELD OUT), 60 residues, t=261 coupled,
shared rotations {R1, R2, R1R2, R3}, independent re-simulation per rotation. **All taps pass the
rank guard** (5.6-45 train samples per channel), so this is not the smoke's rank artifact.

| tap | ch | per_struct | pooled_ho | drop | homo |
|---|---|---|---|---|---|
| `conv_in` | 64 | 0.762 | **0.868** | −0.105 | 0.238 |
| `down_blocks[0]` | 64 | 0.316 | 0.483 | −0.167 | 0.122 |
| `down_blocks[1]` | 128 | 0.652 | 0.666 | −0.014 | 0.296 |
| `down_blocks[2]` | 256 | 0.845 | 0.526 | **+0.319** | 0.206 |
| `down_blocks[3]` | 512 | 0.879 | 0.507 | **+0.372** | 0.235 |
| `mid_block` | 512 | 0.875 | 0.452 | **+0.423** | 0.184 |
| `up_blocks[0]` | 512 | 0.817 | 0.473 | **+0.344** | 0.100 |
| `up_blocks[1]` | 256 | 0.624 | 0.567 | +0.057 | 0.298 |
| `up_blocks[2]` | 128 | 0.455 | **0.673** | −0.218 | 0.123 |
| `up_blocks[3]` | 64 | 0.595 | **0.719** | −0.124 | **0.530** |

`per_struct` = fresh Q per structure (what `layer_relational_invariance.py` measures).
`pooled_ho` = ONE Q_R shared across structures, scored on structures never used to fit it.
`homo` = median cosine of `X Q_R1 Q_R2` against the actual R1R2 features.

**A CLEAN DICHOTOMY, and it kills the hypothesis:**
* **Endpoint taps** (`conv_in`, `up_blocks[2]`, `up_blocks[3]`): pooling HELPS (negative drop), so
  Q genuinely is structure-independent — **but only because Q ~ identity**. These are the taps
  pinned to the TRIVIAL rep by `out_channels: 1`. A shared Q that is the identity carries no
  information.
* **Deep taps** (`down_blocks[2..3]`, `mid_block`, `up_blocks[0]`): pooling COLLAPSES the fit by
  **+0.32 to +0.42**. The 0.82-0.88 per-structure Procrustes was absorbing STRUCTURE-SPECIFIC
  variation — a 512x512 orthogonal matrix fitted on one structure can align that structure's two
  orientations, but the required matrix DIFFERS PER STRUCTURE.
* **=> There is no tap with both a NONTRIVIAL and a STRUCTURE-INDEPENDENT Q.** Where Q is shared
  it is trivial; where features are rich, Q depends on the input. That is a per-structure
  alignment, not a group action on channel space.
* **The homomorphism fails everywhere**: 0.10-0.30 at the deep taps. The best (0.530,
  `up_blocks[3]`) is the most trivial-rep tap, where composition is easy because each Q is
  near-identity.

**Retroactively explains Task 1's null**: with no shared `Q_R`, `C` need not commute with
anything, so no (2l+1) degeneracy should be expected. One fact, two consistent consequences.

**Consequences for the brief:** Tasks 3, 4, 5 have NO FOUNDATION — no irrep blocks to probe
against tensor fields, no pseudoscalar/chirality channel, no ell=1/ell=2 local frames (so O4's
route to fixing the atomic-model circularity is closed). Two survivors:
1. **General-linear / whitened fit** (the brief's own confound #3): everything here assumes
   ORTHOGONAL Q. If the rep sits in a non-orthonormal basis, orthogonal Procrustes understates
   it. Cheap; the last escape.
2. **Task 6 is NOT gated on any of this.** A rotation-consistency term with a chosen multiplicity
   budget does not DISCOVER emergent equivariance, it CREATES it. Given this negative, training
   for steerability is now the ONLY route to steerable CryoFM features.

### ★ O1 FINAL (2026-08-28) — CryoFM2 BEATS raw density on coarse structure, at 2-4x fewer dims

Composition test: best taps x best timesteps, `--per-chain 40` (so raw/prior differ slightly from
the tap sweep; compare within this block).

| tap | dim | setting | SS | AA |
|---|---|---|---|---|
| `up_blocks[1]` | 256 | t10d | 0.7128 | 0.1687 |
| `up_blocks[1]` | 256 | t261d | 0.7740 | 0.1903 |
| **`up_blocks[1]`** | 256 | **t261c** | **0.7839** | 0.1882 |
| `up_blocks[1]` | 256 | t500d | 0.7823 | 0.1967 |
| `up_blocks[2]` | 128 | t10d | 0.6885 | 0.2078 |
| `up_blocks[2]` | 128 | t261d | 0.7626 | 0.2379 |
| `up_blocks[2]` | 128 | t261c | 0.7708 | 0.2393 |
| **`up_blocks[2]`** | 128 | **t500d** | **0.7736** | **0.2475** |
| `raw` voxels | 512 | — | 0.7376-0.7379 | 0.2619-0.2709 |
| prior | | | 0.434 | 0.089-0.090 |

**THE KILL CRITERION IS MET ON SECONDARY STRUCTURE.** `up_blocks[1]`/t261c beats raw voxels by
**+4.6 points at half the dimensions**; `up_blocks[2]`/t500d beats it by **+3.6 at a QUARTER of
the dimensions**. On amino-acid identity CryoFM closes to **94.5% of raw** (0.2475 vs 0.2619) at
4x fewer dims but does not cross.

**THE ARC — my verdict moved three times, and the first version was badly wrong:**
| stage | measured | claim |
|---|---|---|
| `up_blocks[0]`, t=10 (the project's configuration) | 0.5882 / 0.1125 vs raw 0.7506 / 0.2780 | "loses by 16-17 points, teacher is out" |
| + tap sweep | best 0.7183 / 0.2193 | tap-specific, deficit 3.2 / 5.9 |
| + t sweep | `up_blocks[0]` t500 -> 0.7133 | t matters as much as tap; axes compose |
| + composition | **0.7839 / 0.2475** | **beats raw on SS; ~matches on AA, at 2-4x fewer dims** |

**Root cause of the original error: BOTH free axes were set near-worst by the alignment project.**
`up_blocks[0]` was chosen for maximal pose stability (0.826), which is bought by spatial
smoothing (~9-10 A half-decay); t=10 was never chosen at all, it is the SDK-style default at the
near-clean end of a 1000-step trajectory. Neither was ever validated against a downstream metric,
because usefulness was never measured. Measuring one point of a 2-D configuration space and
generalising was the methodological failure — the same shape as the user's original criticism of
the project's scoping.

**Standing implication:** CryoFM2 IS a useful density encoder (better than raw voxels on coarse
structure, 2-4x more compact). That does NOT revive O0: on simulated density the target remains a
deterministic function of the atomic model, so sequence->feature stays dominated by
fold-then-featurise. The value is as an ENCODER, not as an alignment target.

### ★ O1 HIGH-t SWEEP (2026-08-30) — the coupled arm TURNS OVER at t~500; decoupled SATURATES
The earlier t sweep stopped at 500, so the top half of the 0-1000 schedule was never tested. Run at
`up_blocks[1]`, per-residue linear probe, REAL experimental maps, 1,055 chains
(31,626 train / 6,485 test residues), anchored at t=261 coupled to tie it to the previous sweep.

| t | SS coupled | SS decoupled | AA coupled | AA decoupled |
|---|---|---|---|---|
| 261 | 0.7713 | - | 0.1821 | - |
| 500 | **0.7736** | - | 0.1767 | - |
| 750 | 0.7476 | **0.7747** | 0.1601 | **0.1949** |
| 900 | 0.6396 | 0.7730 | 0.1226 | 0.1928 |
| raw voxels (512-d) | 0.7241 | | 0.2476 | |
| random weights | 0.5739 | | 0.1098 | |
| prior | 0.4353 | | 0.0904 | |

- **COUPLED peaks at t~500 and then collapses** (SS 0.774 -> 0.640, AA 0.182 -> 0.123 over
  t=500->900). Expected and now measured: at t=900 the input is ~90% noise, so features drift toward
  the unconditional prior. **AA's coupled optimum is at or below 261** (monotone decline from there).
- **DECOUPLED saturates and stays high across the whole top of the range** (SS 0.7747/0.7730,
  AA 0.1949/0.1928 at t=750/900). Input is always clean, so t is purely a conditioning scalar and
  past ~500 it stops mattering. **Practical consequence: the decoupled operating point needs no
  tuning** - anything in t~500-900 is equivalent.
- **Fixes the operating point** previously logged as "t ~ 261-500 on the basis of a sweep that never
  tested the top half": coupled t~500, decoupled t~500-900 (flat). No reason to go past 750.
- **Absolutes are NOT comparable to the tap sweep** (31.6k vs 44.6k train residues); the same tap
  read 0.7839 SS at t261c there. Within-run ordering is the valid signal.
- **AA still favours raw voxels at this tap** (0.2476 vs 0.1949): a 512-d raw central cube keeps the
  fine detail a 256-d `up_blocks[1]` summary discards, and side-chain identity IS a fine-detail
  question. `up_blocks[2]` is the tap that closes it (0.2475 vs raw 0.2619 = **94.5% of raw at half
  the dims**). NOTE: that 94.5% is RELATIVE TO RAW VOXELS, not an absolute AA accuracy - an earlier
  note in this file could be misread as the latter.

Artifacts: `results/o1_thigh.json`.

#### Consequence for CleanDIFT: the "distillation headroom" is READOUT-DEPENDENT and reverses here
The t-sweep logged in CLAUDE.md measured coupled-minus-decoupled on per-residue **retrieval top1**
(simulated density, `up_blocks[0]`) and found coupled ahead by +0.045/+0.058/+0.071 at t=250/420/750,
reading that gap as the headroom a CleanDIFT student could recover.

**On linear CLASSIFICATION on real maps at `up_blocks[1]`, the sign reverses at high t**: decoupled
beats coupled by **+2.7 SS points** (0.7747 vs 0.7476) and **+3.5 AA points** (0.1949 vs 0.1601) at
t=750. Not a contradiction - different readout, tap, and density source - but it means the headroom
is not a property of the features alone:
- **Retrieval/vector-identity readout** -> coupled ahead -> CleanDIFT has something to recover.
- **Linear-probe classification readout** -> decoupled already >= coupled at high t -> **a clean-input
  student has NO deficit to make up, which removes the motivation for CleanDIFT** for this objective.
Since the downstream design (dense per-residue prediction / contrastive alignment with a linear-ish
head) is much closer to the classification readout than to retrieval, **CleanDIFT should be treated
as unmotivated until measured on the actual objective**, not adopted on the retrieval gap. Cheap
resolution: run the coupled/decoupled comparison once with the real head.

### ⚠ O4 FRAME ARMS (2026-08-30) — the axis CONVENTION is irrelevant; the frame-free question was NOT tested
`probes/o4_frame_arms.py`, 395 chains, `up_blocks[1]`, t=261 coupled, 5,900 train / 1,313 test
residues, per-residue logistic probe. Binomial SE ~0.012 (SS) / ~0.010 (AA).

| arm | SS | d vs aligned | AA | d vs aligned |
|---|---|---|---|---|
| aligned (backbone frame) | 0.7685 | - | 0.1455 | - |
| rand1 | 0.7837 | +0.015 | 0.1592 | +0.014 |
| rand2 | 0.7677 | -0.001 | 0.1523 | +0.007 |
| rand3 | 0.7654 | -0.003 | 0.1561 | +0.011 |
| randavg4 | 0.7776 | +0.009 | 0.1645 | +0.019 |
| randavg8 | 0.7746 | +0.006 | 0.1691 | +0.024 |
| raw voxels (512-d) | 0.6969 | | 0.1805 | |
| prior | 0.4410 | | 0.0967 | |

**⚠ SCOPE ERROR IN THE EXPERIMENT AS BUILT — do not cite this as the model-free result.** The arms
apply `frk = R @ fr[idx]`, i.e. R rotates the CONVENTION of an already residue-anchored backbone
frame. `R @ F_i` is still tied to residue i's N-CA-C geometry, so **every arm here is
pose-invariant by construction and every arm still requires an atomic model.** What it measures is
"does the axis convention of the backbone frame matter" -> **no** (all SS deltas within ~1.3 SE).
It does NOT measure "can the frame be dropped".
- **The same `einsum` is at `o4_frameavg_benchmark.py:183`**, so the earlier logged
  "generic SO(3) costs nothing (SS -0.005, AA +0.002)" carries the identical limitation:
  **neither SO(3) run ever tested the model-free condition.** The octahedral frame-averaging
  result is likewise a statement about averaging over conventions of a residue-anchored frame.
- What legitimately stands: convention-invariance (free), a marginal AA gain from averaging
  (+0.024, ~2.4 SE, consistent with ensembling away trilinear-interpolation noise rather than
  anything about orientation), and CryoFM again beating raw voxels on SS (+7 pts) while losing on
  AA (fine detail favours the 512-d raw cube).
- **Underpowered:** n_test=1,313 -> SE ~1.0-1.2 points, so this rules out large effects only.

**Corrected probe: `probes/o4_lab_arms.py`** — `frk` does not depend on the residue at all
(`lab` = identity/deposited frame, `lab2/lab3` = fixed global orientations, `labavg4/8` = deployable
frame averaging). NOTE it is **frame-free, not fully model-free**: the box CENTRE is still the CA
coordinate. That is the substantive half, since the circularity logged in CLAUDE.md is specifically
about the N-CA-C *frame*, and localisation has a measured model-free substitute (blurred-density
local maxima, 0.63-0.77 repeatability) whereas frame orientation never did.

Artifacts: `results/o4_frame_arms.json`.

### ★★ O4 FRAME-FREE ARMS (2026-08-30) — the backbone frame IS load-bearing; averaging buys back ~57%
`probes/o4_lab_arms.py`. The corrected version of the arm above: `frk` does NOT depend on the
residue, so the frame is genuinely removed. Same 395 chains, `up_blocks[1]`, t=261 coupled,
5,900 train / 1,313 test residues, per-residue logistic probe, binomial SE reported.

| arm | SS | d vs aligned | AA | d vs aligned |
|---|---|---|---|---|
| **aligned** (backbone frame) | 0.7799 | - | 0.1584 | - |
| lab (deposited frame) | 0.7532 | -0.027 (2.2 SE) | 0.1348 | -0.024 (2.5 SE) |
| lab2 | 0.7510 | -0.029 (2.4 SE) | 0.1120 | -0.047 (5.3 SE) |
| lab3 | 0.7403 | -0.040 (3.3 SE) | 0.1196 | -0.039 (4.3 SE) |
| *mean of 3 single orientations* | *0.7482* | *-0.032* | *0.1221* | *-0.036* |
| **labavg4** | 0.7662 | **-0.014 (1.2 SE)** | 0.1424 | **-0.016 (1.7 SE)** |
| labavg8 | 0.7669 | -0.013 (1.1 SE) | 0.1386 | -0.020 (2.1 SE) |
| raw voxels (512-d) | 0.7167 | | 0.1828 | |
| prior | 0.4539 | | 0.0876 | |

- **The frame is worth ~3.2 SS / ~3.6 AA points.** Three independent orientations cluster tightly
  (0.740-0.753 SS), so this is a systematic cost of removing the frame, NOT luck in which
  orientation was drawn. **REVERSES** the reading the convention-only run was heading toward.
- **Frame averaging recovers 57% (SS) / 56% (AA) at K=4**, residual -0.014 SS / -0.016 AA. Each
  residual is only 1.1-2.1 SE alone, but all four (2 tasks x 2 K) are negative -> sign is reliable,
  magnitude is not precise.
- **K=8 adds nothing over K=4** (SS 0.7669 vs 0.7662; AA marginally worse) - independently
  reproduces the octahedral "K=4 buys as much as K=24".
- **Deployable answer: dropping the atomic-model frame costs ~1.3 SS points at 4x inference.**
  CryoFM frame-free (0.766) still beats raw voxels (0.717) on SS by 5 pts. AA is raw voxels'
  throughout (fine detail favours the 512-d cube).
- **Mechanism is LINEAR-READOUT-specific:** with a residue-anchored frame a motif always presents
  in the same relative orientation (consistent template); in a lab frame the same motif appears at
  arbitrary orientations and a *linear* boundary must cover all of them. So a nonlinear /
  learned-invariant head could plausibly close more of the 1.3-pt residual than averaging does.
  **This run sizes that prize; it does not test it.**
- **Run-to-run calibration (useful):** `aligned` reads 0.7799 here vs 0.7685 in the convention run
  on the SAME 395 chains - the residue subsample is drawn from a fresh rng per process, so the
  held-out residues differ (SS prior 0.4539 vs 0.4410 confirms it). **~1.1 points ~= 1 SE of
  run-to-run wobble on a nominally identical measurement -> only WITHIN-run comparisons are valid.**

Artifacts: `results/o4_lab_arms.json`.

### ⚠ BUG (found 2026-08-31) — `centre2` reuses ONE noise field per batch slot; coupled arms affected
`probes/o4_frameavg_benchmark.py:57` `centre2` constructs the RNG **inside** the batch loop with a
hardcoded seed:
```python
for s in range(0, len(boxes), batch):
    if noise_level:
        g = torch.Generator(device="cpu").manual_seed(0)   # <-- inside the loop, fixed seed
        eps = torch.randn(x.shape, generator=g)
```
`probes/local_frame_stability.py:141` `centre_features` hoists it correctly and takes a `noise_seed`.
So the bug is specific to `centre2`, and `centre2` has no `noise_seed` argument at all.

**Consequence:** every batch draws the **same** `eps`, so with `--per-chain 20` (one batch/chain)
residue slot *j* gets an identical noise field in **every chain**. Only ~20 distinct fields exist
across the whole run instead of one per residue. Worse, the field for slot *j* is shared between
train and test residues, which is a leakage channel a linear probe can partially exploit.

**Blast radius — checked, and it spares the headline result:**
- **CLEAN: `o1_cryofm_benchmark.py` uses `centre_features`** (correct hoist). So the **high-t
  timestep sweep (`results/o1_thigh.json`) and the whole O1 tap/t sweep are UNAFFECTED** — coupled
  peaks at t~500 and collapses by t=900, decoupled saturates. That conclusion stands.
- **AFFECTED: `o4_lab_arms.py` and `o4_frameavg_benchmark.py` use `centre2`** with
  `nl = timestep/1000`, i.e. every COUPLED arm in `results/o4_lab_arms.json`,
  `results/o4_frame_arms.json` and the octahedral frame-averaging table.
- **The within-run CONTRASTS are probably robust** (all arms share the identical `eps` tensor per
  slot, so it is a systematic input perturbation common to every arm; and the frame-free deficit was
  2.2-5.3 SE with three independent orientations clustering tightly). **The ABSOLUTE accuracies sit
  on a non-i.i.d. input distribution and should be re-measured after the fix.** Treat "the backbone
  frame is worth ~3.2 SS points" as direction-reliable, magnitude-provisional.
- Decoupled arms are untouched (`noise_level=None` -> no `eps` drawn).

**Fix (not yet applied):** hoist the generator, add `noise_seed: int = 0`, add `--legacy-noise` to
reproduce the old numbers byte-for-byte. This must land **before** any K-draw experiment, because
with the current code K independent draws would remove the *same* field for every residue and leave
a shared residual the probe can absorb — making noise-averaging look artificially good.

---

## 2026-08-31 — CleanDIFT implemented; the `centre2` fix landed; **step 0b (power) FAILED**

### `centre2` noise bug: FIXED
The fix described above is now applied (`probes/o4_frameavg_benchmark.py:57`). The generator is
hoisted out of the batch loop, `noise_seed: int = 0` added, and `--legacy-noise` reproduces the old
per-batch re-seeding byte-for-byte so prior numbers can be regenerated deliberately rather than by
accident. Both `centre2` call sites thread the new arguments. Verified structurally: fixed and legacy
agree on batch 0 (both start from a fresh generator) and diverge on every later batch, which is
exactly the bug's signature.

### ★ STEP 0b — a 1.0-point endpoint is NOT resolvable on this split
New: `probes/o5_stats.py` (paired cluster bootstrap, McNemar, variance split) and
`probes/o5_power_check.py`. Run on the 395 cached chains in `data/o4_lab_parts`
(`results/o5_power_check.json`), CPU, minutes. **1,313 test residues but only 47 test clusters.**

| pair | diff | 95% CI | SE_cluster | design effect | MDE |
|---|---|---|---|---|---|
| `lab − aligned` | −0.0267 | [−0.053, +0.003] | 0.0145 | 1.31 | 0.0285 |
| `lab3 − lab2` (**null**) | **−0.0107** | [−0.032, +0.017] | 0.0127 | 1.15 | 0.0249 |
| `labavg8 − labavg4` | +0.0023 | [−0.011, +0.018] | 0.0074 | 0.94 | 0.0144 |
| `aligned − raw` | **+0.0617** | [+0.033, +0.092] | 0.0150 | 1.06 | 0.0293 |

- **Median MDE ≈ 0.027**, so the pre-registered +0.010 bar was ~2.7× below the noise floor. The bar
  in `PLAN_CLEANDIFT.md` has been raised to **+0.014** on this basis.
- **The null pair fails its own condition**: `lab3 − lab2` should be ~0 (two arbitrary fixed global
  orientations) and reads **−1.07 points**. Gate condition 2 would not have passed at this scale.
- **The design effect is ≈1.10, NOT the ~3× asserted in the plan.** Within-chain correlation inflates
  the SE by ~10%; the problem is the cluster COUNT, not correlation. Corrected in the plan.
- **`--per-chain` 20 → 60 IS worth it**, reversing an earlier claim: the measured SE ratio is **0.82
  per doubling** of residues per chain, so within-cluster sampling noise has not saturated at 20.
- **Projection to the full 1,500-chain split** (100 test clusters) at `--per-chain 60`:
  **MDE ≈ 0.0133**. So effects ≳1.4 points are measurable there; resolving 1.0 point needs
  **≈178 test clusters**, and resolving the 0.17-point decoupled `t`-spread that motivates the whole
  line is off the table by ~2 orders of magnitude.
- **Two calibration literals do not survive at cluster level:** "the backbone frame is worth 3.2 pt"
  reads −0.027 with a CI spanning 0, and the only solidly significant comparison in the table is
  `aligned − raw` = +0.062. Direction-reliable, magnitude-provisional — as the entry above warned.

**Reading.** This is the cheap screen doing its job: it cost minutes and would otherwise have
invalidated a ~20 GPU-hour run after the fact. It does not kill the line, but it does mean the
experiment can only ever detect an effect **much larger than its own stated rationale predicts**.
Either accept a ~1.4-point bar, or rebuild the eval split at ~3× the cluster count
(`build_alignment_set.py`; Cryo2StructData has 7,361 usable entries against the 1,500 chains used).

### Implementation verified on GPU (job 3489011) — every load-bearing identity holds
`probes/o5_verify.py`, H100, 68 s. `probes/o5_cpu_tests.py` is the fast pre-GPU screen.

| check | result |
|---|---|
| **D3 identity** (student-at-init == teacher decoupled at `t_init`) | **2.5e-06** (TF32 off) |
| Head identity-at-init | **exactly 0** |
| Gradient plumbing | `time_embedding.p` 100.6, `mid_block.conv1` 89.6 |
| **D14 truncation equality** (full vs `stop_after`) | **exactly 0**, feature scale 617 |
| `nearest_timestep` at init | **t=750**, residual 0.0, cos 1.0000 |
| `centre2` fix active | batch 0 identical, later batches differ by 57.0 |

**TF32 nearly cost a false negative.** The D3 check first failed at rel_err 5.6e-04 against a 1e-5
tolerance. Cause: cuDNN's TF32 conv path carries ~1e-3 relative error AND its algorithm selection
depends on the autograd context, so the grad-building student forward and the `no_grad` teacher
forward pick different kernels. With TF32 explicitly disabled and both arms under `no_grad`, the same
comparison gives **2.5e-06**. The identity was never broken. **Any exact-equality assertion on this
hardware must disable TF32 first**, and the check now logs both numbers so the diagnosis is in the
log rather than in someone's head.

**Correction to D5's stated diagnostic.** The plan says to log raw cosine beside the centred one and
expect "~0.99, thereby demonstrating why centring was needed". Measured on real boxes, teacher-vs-
student uncentred cosine is **0.16-0.26**, not 0.99. "Raw cosine ~0.99 for any pair" holds for
features from the SAME extractor at the SAME operating point; teacher (noisy `x_t` at t) and student
(clean `x_0` at `t_init`) are different operating points, and D13 itself records them as
near-orthogonal (median centred cosine 0.003 to -0.44). Centring is still correct and still
necessary -- the synthetic test in `o5_cpu_tests` shows uncentred 0.997 vs centred -0.0000 on a
shared common component -- but **a health check expecting raw ~0.99 on the teacher/student pair will
fire falsely**, which is exactly what a first version of the verifier did. The check is now "centring
materially changes the cosine" (measured shift 0.199-0.500 across taps).

**Trainer smoke, 30 steps, lr 1e-4, batch 2 (not a real config):** loss +0.033 -> **-0.720**;
val `cos_head` **+0.717**, `cos_bypass` **+0.715**. Weight deltas mid_block 1.99e-02 / down_blocks
1.63e-02 / up_blocks 1.43e-02 / conv_in 6.42e-03, all far above the 1e-3 training-failure floor;
`conv_norm_out` and `conv_out` are **exactly 0**, which independently confirms `stop_after` truncates
before them. **`head_share` = 0.997 / 1.000 / 1.000** -- the learning is in the SHIPPED TRUNK, not
the discarded heads, which is the D15 condition that matters most.

**Unexpected, and it bears on D13:** 30 steps reached val cosine 0.72 from a start of -0.30 to +0.18.
D13 budgeted 20k steps on the grounds that the target is near-orthogonal. It IS near-orthogonal at
init (centre-token cosines -0.82 / -0.31 / -0.48, and step-0 loss is *positive*), but most of the gap
appears to close almost immediately at lr 1e-4. The lr probe (step 6) should therefore also sweep
STEP COUNT, since 20k may be an order of magnitude more than needed -- and an over-long run on a
frozen teacher risks overfitting the student to the training maps, which `student_ctrl` would then
partly absorb.

### Two implementation bugs of mine, both fixed, both instructive
1. **`build_vol_cache` exited 0 having cached nothing.** `np.save(path, arr)` appends `.npy` to any
   path not already ending in it, so temp file `X.npy.tmp` was written as `X.npy.tmp.npy` and the
   rename failed on a path that never existed -- 1,147/1,147 maps raised `FileNotFoundError`, the
   `except` swallowed each one, and the job reported success after writing **69.8 GiB of orphans**.
   Fixed by writing through an open file handle. **The pattern, not the typo, was the fault:**
   "never raise out of the loop" converts a systematic failure into a successful-looking job.
   `build_vol_cache`, `o5_kdraw_gate` and `o5_cleandift_arms` now fail loudly on zero successes or
   >50% skips. Re-run: **1,147 maps, 0 skipped, 69.8 GiB, 35.4 min.**
2. **`FiLMHead` called `.view()` on a `chunk()` output** (non-contiguous) -> RuntimeError on GPU.
   Fixed with `.reshape`. Cost a cache run plus a GPU round trip because the CPU tests covered the
   loss and sampler but **not the head** -- the failure landed exactly in the coverage gap. Hence
   `probes/o5_cpu_tests.py`, which now covers stats, loss, sampler, head and split in ~2 s.

### ★ STEP 2 — the K-draw ceiling gate (job 3489041, 46 min, 298/300 chains, t=500)
`probes/o5_kdraw_gate.py`, `results/o5_kdraw_gate.json`. Cumulative draws, so K=1,2,4,8 come from
the same 8 forwards.

| | k1 | k2 | k4 | k8 | dec | K=inf (extrap) | gain vs dec |
|---|---|---|---|---|---|---|---|
| **SS** | 0.7774 | 0.7807 | 0.7802 | 0.7802 | 0.7711 | 0.7812 | **+0.0101** |
| **AA** | 0.1629 | 0.1720 | 0.1739 | 0.1677 | 0.1796 | 0.1731 | **−0.0066** |

K=8 vs dec, SS: **+0.0091, CI [−0.0044, +0.0209]**, MDE 0.0129, 40 test clusters.

**Nominally a PASS on SS by +0.0001 against a +0.0100 bar — i.e. a margin ~1% of the gate's own
noise floor. Treat it as UNRESOLVED, not as a pass.** Three things say so:
- The CI on the K=8-vs-dec difference **crosses zero**; MDE 0.0129 > the 0.010 bar, so this gate
  cannot resolve its own threshold at 40 clusters (the module says so in its own output).
- **On AA the gain is NEGATIVE** (−0.0066 extrapolated, −0.0119 at K=8). Noise-averaging the coupled
  teacher is *worse* than plain decoupled extraction for amino-acid identity.
- **Noise-averaging saturates at K=2** (0.7807 -> 0.7802 -> 0.7802). Whatever it buys is exhausted by
  the second draw, so the 1/K extrapolation is nearly flat and the linear-in-1/K model is misspecified
  — harmless here only because K=inf (0.7812) lands within 0.001 of K=8.

**The decision-relevant arithmetic, combining the two gates.** Step 0b says the primary endpoint is
only measurable for effects **≳ +0.014** on the full split. Step 2 puts the noise-marginalisation
ceiling at **~ +0.010 on SS and negative on AA**. So *the ceiling sits below the resolution of the
experiment designed to measure it*: a 20-GPU-hour training run would return an unresolvable result
**by construction**, whatever the student actually learns.

**One honest qualification against over-reading this.** The gate bounds NOISE MARGINALISATION at a
fixed t — it does not bound `t`-CONSOLIDATION, which is a separate channel and the one CleanDIFT's
own claim rests on ("a merge over the trajectory is not a point on it"). D8's stated logic ("the
target carries nothing the decoupled arm already has -> the run is guaranteed to fail") is therefore
stronger than what was measured. A marginal K-draw result does not by itself condemn the line; what
condemns the *current experiment* is the ceiling-below-resolution arithmetic above.

**Recommendation: do NOT launch the 20k-step runs as specified.** Ranked options:
1. **Stop and publish the negative** (~0 further cost). Two independent cheap gates say the
   measurable upside is thinner than the measurement. This is the honest, cheap outcome.
2. **Enlarge the eval split to ~180 test clusters** (`build_alignment_set.py`; Cryo2StructData has
   7,361 usable entries vs the 1,500 chains used) and re-run BOTH gates. ~1 day of mostly-CPU work,
   and it is the only route that makes a 1-point effect measurable at all.
3. **Re-run step 2 alone at `--limit 1000 --per-chain 60`** (~3 h) to settle whether the ceiling is
   really ~+0.010 or nearer zero. Cheapest way to firm up the number, but it does not fix the
   endpoint's resolution, so it informs the decision without enabling the experiment.

Do not spend the training budget until one of 2 or 3 has moved the numbers.

### CORRECTION (2026-08-31, same day) — the K-draw gate does NOT bound the distillation
The recommendation above ("do NOT launch the 20k-step runs") rested on ceiling-below-resolution
arithmetic, and **the ceiling was measured on the wrong channel**. K-draw averages the coupled teacher
over noise draws at a FIXED t, i.e. the *noise-marginalisation* channel. CleanDIFT's gain comes from
*`t`-consolidation*: one representation explaining the teacher across the whole schedule, which the
paper reports as **beating the best single timestep**. That is an extrapolation OUTSIDE the decoupled
family's observed range, so it is bounded neither by the family's spread (the "0.17 pt" framing, which
was only ever a bound on plan benefits 1-2) nor by this gate. The distinction was stated in the entry
above and then not applied — a bound on one channel was allowed to license a decision about another.

**What survives:** step 0b's ~1.4-point resolution floor, which is a property of the measuring
apparatus and holds regardless of mechanism. **The expected effect size is unknown**, not small.

**A prior I had weighted the wrong way.** The `dec_*` arms feed CLEAN input while declaring a high `t`
— off-manifold, an untrained hack — and are the best CryoFM arms measured on BOTH tasks (SS 0.7747 vs
best coupled 0.7736; AA 0.1949 vs best coupled 0.1821). CleanDIFT is the principled version of that
hack: rather than lying to the model about `t`, train it to be good on clean input. That argues for
student >= decoupled. The disanalogy with the paper cuts the other way and is worth stating: in SD,
clean-input features are poor, so CleanDIFT has a large gap to close; here the clean-input arm is
already on top, so the gap may be smaller. Both are priors, neither is a ceiling.

**Revised decision: PROCEED.** D13 lr probe launched (jobs 3489301/2/3, lr 1e-5 / 3e-5 / 1e-4,
500 steps each), primary runs to follow on its outcome. Interpretation set in advance: >~1.4 pt is
conclusive; a smaller positive is suggestive but unresolvable and points to enlarging the eval split
to ~180 test clusters.

### Reference architecture VERIFIED against the paper (2026-08-31) — two corrections
Fetched arXiv 2412.03439v2 directly rather than trusting this repo's inherited summary.

1. **Heads really are discarded at inference** — *"For feature extraction at inference time, we
   usually discard the projection heads and directly use the feature extraction model's internal
   representations."* So `cos_bypass` is measuring the right quantity.
2. **The "+0.24 PCK" figure was misattributed.** Appendix A had it as "discarded at inference (+0.24
   PCK in ablation)", implying discarding buys 0.24. Table 3 actually compares **training with heads
   vs training without heads at all**, and *both* arms discard at inference. The number argues for
   HAVING heads during training (+0.24 pp PCK_img, +0.06 pp PCK_bbox, cosine), not for keeping them.
3. **The head architecture is much larger than ours.** Paper: *"three stacked Feed Forward Networks
   (FFNs) that are zero-initialized such that initially they act as identity mappings due to their
   residual connections"*, *"a FiLM layer in each FFN block"*, *"the SwiGLU gating mechanism ... in
   each FFN block"*, **45M params for SD 2.1**. Ours (`FiLMHead`): one block, conv->SiLU->conv,
   bottlenecked to C/4 = **0.20M at C=512**, ~50x smaller.

**Our bottleneck rationale is inverted relative to the paper's design intent.** The paper gives the
head ample capacity *so that it absorbs the `t`-specific part*, leaving the trunk `t`-agnostic. We
narrowed it to stop the head absorbing the discrepancy and leaving the shipped trunk unmoved. Both
are coherent; they prescribe opposite widths. **Our own data favours the paper:** the lr probe put
`head_share` at 0.998-0.999, so absorption is not happening and the bottleneck guards a non-risk
while plausibly limiting the fit. `PaperHead` (3 x SwiGLU FFN, FiLM per block, 10.24M at C=512,
identity at init, unit-tested) implemented and launched as an arm.

### ★ Effect-size sanity check against the paper (2026-08-31) — the experiment CAN be conclusive
Fetched arXiv 2412.03439v2 for the headline magnitudes, which nothing in this project had checked.

- **Semantic correspondence: +1.79 PCK@alpha_img / +1.86 PCK@alpha_bbox** over DIFT (Table 1);
  **+2.81** against a baseline without noise-averaging. Depth (NYUv2) 0.469 -> **0.444** RMSE.
- Table 3 gives the absolute: cosine **68.32** PCK@alpha_img, so the DIFT baseline is ~66.53 and the
  headline gain is **+2.7% RELATIVE**.
- **Transferred by relative size onto our endpoint** (SS accuracy ~0.775): ~**+2.1 points**, against
  step 0b's ~**1.4-point** resolution floor. **So an effect of the paper's magnitude WOULD be
  detectable on the full split.** This is the number that should have been checked before the K-draw
  gate was ever framed as a stop condition.
- **Training range CONFIRMED as the full schedule:** `t_i ~ U(i/I*T, (i+1)/I*T)`, `I=3`, over
  **[1, 999]**. Our `t_max=1000` default is right; the 600 cap was the deviation. (Both are running.)

**Remaining inherited Appendix A claims — ALL VERIFIED, no further corrections:**
- **Loss form: cosine > L1 > L2.** PCK@alpha_img **68.32 / 66.91 / 66.23**; "cosine similarity
  consistently performs the best across the alignment objectives by a significant margin." D5's
  choice is correct, and note the loss form alone is worth ~2 pp — comparable to the headline gain.
- **FiLM > AdaRMS by only 0.1 pp**; removing the gating mechanism costs **0.12 pp**. So our
  SiLU-instead-of-SwiGLU deviation is worth ~0.1 pp — negligible. It is the head's CAPACITY, not its
  activation, that is the real open question (see the PaperHead arm).
- **Full fine-tune for SD 1.5 / 2.1 / DiT; LoRA rank 64 only for SDXL (Turbo) and Flux** "due to
  their large model size". CryoFM2 is 168M, so full fine-tune is the right analogue.
- **t=261** is named as a typical DIFT timestep, which the method beats without tuning.

**RISK now visible in the runs in flight.** Paper: **lr 2e-6, 400 steps, ~3k images, 30 min on one
A100.** Ours: **20,000 steps at 3e-5** — ~50x the steps at ~15x the lr. The lr probe selected 3e-5
empirically on `cos_bypass`, but it never probed below 1e-5 (i.e. never near the paper's value) and
ran for only 500 steps, so its choice may not extrapolate to 20k. Mitigation is genuine — best
checkpoint is selected on `cos_bypass` against a MAP-DISJOINT val set of 138 chains, so it is real
early stopping — but the `cos_bypass` curve must be checked for an early peak and decline rather than
assumed monotone.

### Student checkpoint round-trip VERIFIED (job 3489339) — a seam that had never been executed
`load_student` was written but never run, because no student checkpoint existed while the pipeline was
being built. A failure there would have surfaced only AFTER all six training runs completed. Tested
against a live step-3000 checkpoint: the D2 swap is applied before `load_state_dict` (keys are
`time_embedding.p`, and `load_cryofm2` raises on any mismatch), the loaded conditioning vector matches
the checkpoint bit-for-bit, two independent loads are deterministic, and student-vs-teacher
max|d| = **127.3** confirms the weights genuinely loaded. `--student-ckpt` is now part of `o5_verify`.

### Arms-evaluation SMOKE (job 3489397, 149 chains x 20 res, 15 arms, 18 min) — module works; GATE CONDITION 2 IS MIS-SPECIFIED
Everything plumbed: 7 student checkpoints loaded, val-selection ran, `dec_ens` 768 -> PCA-256 built,
cluster bootstrap and all four gate conditions evaluated. **20 test clusters only, so no result below
is trustworthy** (MDEs 0.03-0.06) — but two structural problems surfaced.

**1. Gate condition 2 tests the wrong thing — my specification error.** It requires
`cou_best` vs `cou_seed1` ("identical but for the noise seed") to have a CI containing 0 and
`|diff| < 0.005`, as a measurement-resolution check. Measured: **−0.0289, CI [−0.0577, −0.0061],
CI EXCLUDES ZERO.** But those two arms are *not* identical — different noise seeds give genuinely
different feature sets, so the true difference is a random draw with expectation 0, not 0 itself. With
enough clusters the CI will correctly exclude zero whenever that draw is nonzero, so as written the
condition **demands the measurement be too imprecise to detect a real difference** — backwards.
- Substantively it is still informative: **the coupled extractor has ~3 points of noise-draw slop**,
  which is a real (and previously unquantified) nuisance term — and an argument FOR a deterministic
  student, which has none.
- **It does not bound the primary endpoint.** `student` and `dec_best` are both DETERMINISTIC given
  the boxes, and the comparison is paired on identical residues, so there is no extractor-level noise
  in that difference; the cluster bootstrap is the correct error bar for it.
- **Replacement:** require the claimed `student − dec_best` gain to exceed the measured NUISANCE
  variation — (a) coupled noise-seed spread (`cou_best` vs `cou_seed1`) and (b) student training-seed
  spread (`student` vs `student_seed1`) — rather than requiring a null pair to read zero.

**2. Val-selection needs a real val split.** Val was only 180 residues here, and it picked
`dec_t900` as `dec_best` for SS although `dec_t261` scored higher on test (0.7125 vs 0.6998). At
`--limit 1500 --per-chain 60` val is ~8,280 residues, which is the configuration step 0b's floor was
projected against.

Directional hints only, all with CIs crossing zero at 20 clusters: `student_t700 − dec_best` +0.043,
`student_paperhead − dec_best` +0.040, `student − dec_best` +0.015. Two comparisons did clear their
CI: `student_t700 − student_ctrl` +0.052 [+0.007,+0.092] and `student_paperhead − student_ctrl`
+0.049 [+0.002,+0.082]. Every student beat `raw` significantly. `student_tonly − student_ctrl` = +0.002
(nil), consistent with the trunk-frozen arm being a no-op. **Do not quote any of these.**

## ★★★★★ 2026-08-31 — THE CLEANDIFT RESULT (job 3489410, 1,484 chains, 7h47m)
`results/o5_cleandift_arms.json`. **13,547 test residues / 100 test clusters / 8,160 val residues**,
all 16 arms in ONE process on IDENTICAL residues, paired cluster bootstrap over the 100 test clusters,
`dec_best` and `cou_best` selected on VAL, test read once.

### Secondary structure (the pre-registered endpoint, tap `up_blocks[1]`)

| arm | test acc | vs `dec_best` (t750 = 0.7760) | 95% CI | MDE |
|---|---|---|---|---|
| **`student_paperhead`** | **0.7908** | **+0.0148** | [+0.0087, +0.0226] | 0.0068 |
| `student` | 0.7862 | **+0.0101** | [+0.0027, +0.0200] | 0.0087 |
| `student_seed1` | 0.7850 | +0.0089 | [+0.0022, +0.0180] | 0.0078 |
| `student_t700` | 0.7849 | +0.0089 | [+0.0017, +0.0189] | 0.0088 |
| `cou_best` | 0.7811 | — | — | — |
| `dec_ens_pca` | 0.7781 | +0.0021 | [−0.0039, +0.0069] | ns |
| `student_ctrl` | 0.7772 | +0.0012 | [+0.0000, +0.0027] | ns |
| `student_ctrlsampled` | 0.7769 | +0.0009 | [−0.0047, +0.0080] | ns |
| `raw` (8^3 voxels) | 0.7475 | −0.0285 | — | — |
| `rand_student` | 0.5705 | — | — | (prior 0.4389) |

**THE CAUSAL ATTRIBUTION IS CLEAN, and this is the load-bearing part.** The two arms that SHOULD be
null are exactly the two that ARE:
- **`student_ctrl` +0.0012** — same corpus, same steps, same loss, trivial target -> **corpus
  adaptation contributes ~nothing.**
- **`student_ctrlsampled` +0.0009** — clean teacher at the SAMPLED t, i.e. full `t`-variation but
  **zero noise information** -> **`t`-consolidation of clean features contributes ~nothing.**
- => **the gain REQUIRES the noisy teacher.** It is genuine distillation of noised-input features, not
  corpus drift and not a better conditioning vector (`tonly` was +0.002 in the smoke, and its
  `cos_bypass` was 0.24 vs distill's 0.70 — two independent measures agreeing it is a no-op).
- **`dec_ens_pca` +0.0021, CI containing zero** -> the zero-training `t`-ensemble does **not**
  reproduce the effect. The free rival is ruled out.
- **Null pair `cou_best − cou_seed1` = +0.0005, CI [−0.0044,+0.0041]** -> gate condition 2 **PASSES**
  at 100 clusters. (The specification objection logged above stands in principle but does not bite
  here: with the full split the two seeds happen to agree to 0.0005.)
- **Seed spread** `student` vs `student_seed1` = 0.0012, so the effect is ~8x the nuisance term.

### The gate, read honestly
Reported `PASS: False`, on condition 1 alone: `student − dec_best` = **+0.0101 against the +0.014
bar**. Every other condition passed (null pair valid, beats ctrl by +0.0089 >= +0.007, beats raw by
+0.0386). Three things must be said together:
- **The +0.014 bar came from step 0b's PROJECTED MDE of 0.0133. The REALISED MDE for this comparison
  is 0.0087** — the paired difference is tighter than the projection assumed. So the effect is
  genuinely resolvable (CI excludes zero) even though it sits under a bar set by a conservative
  forecast. Both facts belong in any quotation of this result.
- **`student_paperhead` = +0.0148 DOES clear the bar** with CI [+0.0087,+0.0226]. But it was an
  ablation, not the pre-registered primary, so calling it the headline is post-hoc selection among 6
  student arms.
- **What defeats the multiplicity worry is the PATTERN, not the maximum:** 4 of 6 student arms are
  significantly positive in the same direction, and the 2 nulls are precisely the 2 CONTROLS that
  mechanism predicts should be null. That is not a fishing expedition.

### Amino acid — no effect, and raw voxels still win
`student − dec_best` = +0.0033 (ns); only `student_t700` reaches significance (+0.0070). **Every arm
loses to raw 8^3 voxels by ~5 points** (student 0.2204 vs raw 0.2735). Consistent with every earlier
AA finding in this project. So the effect is SS-specific.

### ★ FEATURE-MATCHING COSINE DOES NOT PREDICT DOWNSTREAM PERFORMANCE — twice, in opposite directions
- `paperhead` had **lower** `cos_bypass` (0.6999 vs 0.7058) but **higher** SS (0.7908 vs 0.7862).
- `t700` had **much higher** `cos_bypass` (0.798 vs 0.706) but **equal** SS (0.7849 vs 0.7862).
So the `t`-range cap is downstream-irrelevant while head capacity matters — the **opposite** of what
the training curves suggested. This retroactively vindicates the plan's insistence that the gate be a
downstream task and never teacher-feature agreement (Appendix B, P8): had we selected on cosine we
would have shipped `t700` and skipped `paperhead`.

### Magnitude vs the paper
Ours **+1.0 to +1.5 SS points** (1.3-1.9% relative); paper **+1.79 PCK@alpha_img** (2.7% relative).
Same order, roughly half the relative gain — reasonable for a different modality, task and backbone.

### One more spec bug of mine, found and fixed
`kill_matched_by_t_ensemble` reported **True** (i.e. "an ensemble does it for free"). The test was
`|student − ens| < bar`, which fires whenever the two are merely CLOSE — including when the ensemble
gains nothing, which is exactly the case where it does NOT match. Corrected to require the ensemble to
be a real gain of comparable size; re-evaluated on the saved JSON it is **False**. The ensemble does
not explain the result.

### ★ CleanDIFT vs "best t" — three baselines, both tasks (job 3492055, cached parts, CPU 7m48s)
`results/o5_cleandift_arms_baselines.json`. Same 100 test clusters, paired cluster bootstrap.
"Best t" is NOT one thing; the answer depends on which baseline, so all three are reported.

| baseline | what it is | SS acc | AA acc |
|---|---|---|---|
| `dec_best` | best DECOUPLED (clean input), chosen on **val** — deployable, and what the gate used | 0.7760 (t750) | 0.2171 (t500) |
| `dec_oracle` | best DECOUPLED by **test** acc — an ORACLE, unobtainable without peeking | 0.7798 (t900) | 0.2171 (t500) |
| `cou_best` | best COUPLED (noised input) — **the analogue of the paper's DIFT baseline** | 0.7811 | 0.2062 |

**SS** (`*` = 95% CI excludes 0):

| arm | vs `dec_best` | vs `dec_oracle` | vs `cou_best` |
|---|---|---|---|
| **`student_paperhead`** | **+0.0148*** | **+0.0110*** | **+0.0097*** |
| `student` | +0.0101* | +0.0063 | +0.0050 |
| `student_seed1` | +0.0089* | +0.0052 | +0.0038 |
| `student_t700` | +0.0089* | +0.0051 | +0.0038 |

**AA:**

| arm | vs `dec_best` = `dec_oracle` | vs `cou_best` |
|---|---|---|
| `student_t700` | +0.0070* | **+0.0179*** |
| `student` | +0.0033 | **+0.0142*** |
| `student_seed1` | +0.0027 | **+0.0135*** |
| `student_paperhead` | +0.0010 | **+0.0119*** |

**Readings:**
- **Against the paper's own baseline type (noised best-t): CleanDIFT wins on BOTH tasks.** AA is
  significant for every arm (+0.012 to +0.018); SS is significant only for `paperhead` (+0.0097),
  with the plain student at +0.0050 (ns).
- **Against the clean/decoupled best-t (the harder, deployable baseline): SS only.** Every distill arm
  is significantly positive on SS (+0.009 to +0.015); on AA nothing clears zero except `t700`
  marginally (+0.0070).
- **Against the ORACLE decoupled best-t: only `paperhead` survives** (+0.0110*). This is the paper's
  claim in its strongest form ("beats the best single timestep") and exactly one arm meets it.
- **`student_paperhead` is the ONLY arm significant against ALL THREE baselines on SS**, which
  strengthens the head-architecture finding and makes its missing seed replicate the highest-value
  remaining run.
- Raw 8^3 voxels still beat every arm on AA by ~5 points, unchanged from every prior AA result here.

**Most defensible one-liner:** CleanDIFT improves on noised best-`t` features on both tasks; the
improvement over the *clean* (decoupled) best-`t` is SS-only, and survives the oracle baseline only
with the paper-matched head.

### All-taps smoke + paperhead seed replicate (jobs 3492231 / 3492206)
**Seed replicate trains consistently:** `paperhead` best `cos_bypass` **0.7036** (seed 1) vs **0.6999**
(seed 0), spread 0.0037, wdelta 5.2e-02 both. Its DOWNSTREAM number needs the full arms run.

**★ CORRECTION TO MY OWN EXPECTATION: `up_blocks[1]` is far the best tap for these probes, and
`up_blocks[0]` is not close.** Smoke (150 chains x 20 res, so directional only), SS test accuracy for
`student`: `up_blocks[1]` **0.7143** vs `up_blocks[0]` **0.5787** vs `mid_block` **0.5913** — a ~13
point gap. Raw 8^3 voxels (0.6474) beat BOTH coarser taps and lose only to `up_blocks[1]`.
- I had expected `up_blocks[0]` to be competitive because `CLAUDE.md` calls it "the best per-residue
  tap". **That refers to POSE INVARIANCE (0.826 vs 0.706), which is a different quantity from
  downstream informativeness** — I conflated them. The receptive-field measurements in the same file
  actually predict the observed ordering: half-decay ~4-5 A (`up_blocks[1]`) vs ~9-10 A
  (`up_blocks[0]`) vs ~15 A (`mid_block`), and SS/AA are LOCAL per-residue labels, so the finest tap
  should win. It does.
- Consequence: the published headline was measured on the right tap by luck of the original choice,
  and the CleanDIFT gain looks `up_blocks[1]`-specific (at `mid_block` and `up_blocks[0]` the student
  did not beat `dec_best`, though nothing resolves at 20 clusters).

**`concat` is inconclusive from the smoke and needs full scale.** 1280 dims against only 2,240 train
residues is a bad ratio, so the smoke's concat numbers (student 0.7161, paperhead 0.6962) are
confounded by probe overfitting. At full scale there are ~66k train residues, which is a fair test.

## ★★★★★ 2026-09-01 — ALL-TAPS EVALUATION (job 3492533, 1,484 chains, 9h18m, 100 test clusters)
`results/o5_arms_multitap.json`. 7 students x 5 views x 2 tasks, one forward per arm capturing all
three taps. Diff = vs that view's own val-selected `dec_best`; `*` = 95% cluster-bootstrap CI excludes 0.

**SS** (`raw` = 0.7475 in every view):

| view (A/token) | `dec_best` | `student` | `paperhead` s0 | `paperhead` s1 |
|---|---|---|---|---|
| `mid_block` (12 A) | 0.6771 | 0.6686 −0.0084 | 0.6760 −0.0010 | 0.6703 −0.0068 |
| `up_blocks[0]` (6 A) | 0.7314 | 0.7083 **−0.0230\*** | 0.7152 **−0.0162\*** | 0.7143 **−0.0171\*** |
| **`up_blocks[1]` (3 A)** | 0.7760 | 0.7862 **+0.0101\*** | 0.7908 **+0.0148\*** | 0.7874 **+0.0114\*** |
| `concat` (1280-d) | 0.7788 | 0.7833 +0.0045 | 0.7874 +0.0086* | 0.7837 +0.0049 |
| `concat_pca` (256-d) | 0.7723 | 0.7766 +0.0042 | 0.7753 +0.0030 | 0.7749 +0.0025 |

**AA:** nothing significant anywhere except one NEGATIVE (`paperhead_s1` at `up_blocks[0]`, −0.0067*).
`raw` = 0.2735 beats every arm in every view by 5-15 points. The AA null is now definitive.

### Four conclusions
1. **The gain is real, REPLICATED, and specific to `up_blocks[1]`.** Four independent student runs are
   all significantly positive there (+0.0101, +0.0148, +0.0114, and +0.0089 for `seed1` from the
   earlier run). This is the finest tap (3 A/token) and SS/AA are local per-residue labels, matching
   the receptive-field prediction (half-decay ~4-5 A).
2. **★ CleanDIFT actively HURTS at `up_blocks[0]`** — −0.0162 to −0.0230, all three CIs excluding zero.
   The distillation TRADES coarse-tap quality for fine-tap quality. Plausible mechanism: D5 weights
   the three taps equally (1/3 each) and the student cannot satisfy all simultaneously, so it
   sacrifices the coarse ones; or matching high-`t` teacher features at coarse taps drags them toward
   noise statistics. **This is a concrete, well-motivated next experiment: re-weight the loss toward
   the judged tap, or distil `up_blocks[1]` alone, and see whether the fine-tap gain grows.**
   It also matters for the ESM-C alignment, whose target is `up_blocks[0]` — **the current student is
   the WRONG model for that use**, and would need a re-weighted or single-tap variant.
3. **Concatenating taps does NOT help.** `up_blocks[1]` alone (256-d, 0.7862) beats `concat` (1280-d,
   0.7833) and `concat_pca` (256-d, 0.7766), with 66k training residues so this is not a
   dimensionality artifact. There is no multi-layer gain to be had here.
4. **The `paperhead` advantage does NOT cleanly replicate.** Seeds give +0.0148 and +0.0114 (mean
   +0.0131) against film-head +0.0101 and +0.0089 (mean +0.0095). The ~+0.0036 head advantage is
   comparable to `paperhead`'s own seed spread (0.0034). **Critically, only ONE of the two
   `paperhead` seeds clears the pre-registered +0.014 bar.** Retract the earlier framing that the
   paper-matched head clears it — the honest statement is that all four students clear ZERO reliably,
   none clears +0.014 reliably, and the head effect is suggestive at best.

## ★★★★ 2026-09-01 — SPATIAL VARIABILITY OF THE CLEANDIFT / CRYOFM LATENTS
The sibling ESM project's three spatial analyses, ported and run on the o5 per-residue latents.
Artifacts: `probes/o7_recover_coords.py`, `probes/o7_spatial_variability.py`, `probes/o7_report.py`,
`slurm/o7_spatial.slurm`; `data/o7_coords.npz`; `results/o7_spatial_variability{,_ctrl}.json`,
`results/o7_spatial_correlogram{,_ctrl}.json`. Jobs 3502153 / 3502175 / 3502385 / 3502392, CPU only.
**1,484 chains, 88,251 residues, cluster split 1118/136/230 chains** (the same split the arms
evaluation uses, so no leakage across 30%-identity clusters).

**Why this is the right test.** The ESM half of this repo concluded that ESM-C per-residue
embeddings carry essentially no coarse within-protein spatial variance (50 A band 0.1%, 25 A 0.4%,
10 A 3%, local 86%), that residue->residue predictability collapses past ~10-15 A, and that the only
remaining source of genuine multi-scale spatial features would be a **structure-native encoder**.
CryoFM2 is exactly that, so running the identical measurements on its features is the direct test of
that prediction.

### Enabling step: the coordinates were recoverable, not lost
`o5_cleandift_arms.py` / `o5_arms_multitap.py` store `aa`/`ss`/`split` but **no residue index and no
Ca position**, so none of these analyses could run as stored, and re-extracting on GPU is hours.
The selection is instead a REPLAY: `chain_boxes` draws it from one `default_rng(0)` advanced over the
chains in `emd`-sorted order, and every failure path raises BEFORE `rng.choice`. `o7_recover_coords.py`
replays it and **verifies rather than assumes** -- the recovered index set must reproduce the stored
`aa` AND `ss` arrays exactly for every chain, or it aborts. **1,484/1,484 verified, 0 mismatches.**

### A — variance hierarchy (% of WITHIN-protein variance; `global` = between-protein, % of total)

| arm | dim | global% | 50 A | 25 A | 10 A | local |
|---|---|---|---|---|---|---|
| `esmc` (matched residues) | 1152 | 25.85 | 0.210 | 0.676 | 5.215 | 76.59 |
| `esmc_all` (all residues) | 1152 | 24.59 | 0.220 | 0.573 | 3.350 | 84.37 |
| `raw` (8^3 density voxels) | 512 | 25.93 | 0.143 | 0.470 | 4.116 | 80.89 |
| **rand** `mid_block` | 512 | 10.50 | 0.140 | 0.495 | 4.136 | 80.33 |
| **rand** `up_blocks[1]` | 256 | 14.43 | 0.154 | 0.562 | 4.656 | 78.93 |
| **rand** `up_blocks[0]` | 512 | 5.95 | 0.100 | 0.362 | 3.569 | 82.76 |
| `dec_t500@mid_block` | 512 | **63.69** | **0.777** | **2.179** | **11.008** | 54.02 |
| `dec_t500@up_blocks[1]` | 256 | 52.43 | 0.522 | 1.422 | 8.285 | 63.76 |
| `dec_t261@up_blocks[0]` | 512 | 40.66 | 0.360 | 1.069 | 6.781 | 70.37 |
| `dec_t500@up_blocks[0]` | 512 | 34.50 | 0.356 | 1.104 | 7.161 | 69.05 |
| `dec_t900@up_blocks[0]` | 512 | 30.84 | 0.391 | 1.260 | 8.269 | 65.16 |
| `cou_best@up_blocks[0]` | 512 | 35.05 | 0.352 | 1.105 | 7.341 | 68.78 |
| `student@up_blocks[0]` | 512 | 34.07 | 0.342 | 1.119 | 7.628 | 67.78 |
| `student_paperhead@up_blocks[0]` | 512 | 34.28 | 0.346 | 1.131 | 7.605 | 67.80 |

### B — ridge R2 predicting residue j's feature from residue i's, **through-space `|i-j|>8`**

| arm | 0-5 | 5-10 | 10-15 | 15-20 | 20-25 | 30-40 | 40-50 |
|---|---|---|---|---|---|---|---|
| `esmc` | 0.087 | 0.032 | 0.012 | 0.004 | 0.002 | 0.017 | 0.010 |
| `esmc_all` | 0.155 | 0.036 | 0.011 | 0.005 | 0.002 | 0.011 | 0.008 |
| `raw` | 0.070 | 0.009 | 0.002 | 0.000 | 0.001 | 0.001 | 0.001 |
| **rand** `mid_block` | 0.158 | 0.030 | 0.013 | 0.004 | 0.000 | 0.007 | 0.009 |
| **rand** `up_blocks[1]` | 0.130 | 0.043 | 0.017 | 0.005 | 0.001 | 0.010 | 0.011 |
| **rand** `up_blocks[0]` | 0.071 | 0.010 | 0.004 | 0.001 | 0.000 | 0.002 | 0.003 |
| `dec_t500@mid_block` | 0.279 | **0.176** | **0.089** | **0.031** | 0.007 | 0.034 | 0.042 |
| `dec_t500@up_blocks[1]` | 0.275 | 0.103 | 0.044 | 0.018 | 0.005 | 0.015 | 0.017 |
| `dec_t261@up_blocks[0]` | 0.149 | 0.056 | 0.027 | 0.009 | 0.003 | 0.013 | 0.016 |
| `dec_t500@up_blocks[0]` | 0.216 | 0.082 | 0.036 | 0.013 | 0.005 | 0.016 | 0.020 |
| `dec_t900@up_blocks[0]` | **0.333** | 0.133 | 0.057 | 0.022 | 0.011 | 0.027 | 0.029 |
| `cou_best@up_blocks[0]` | 0.176 | 0.076 | 0.034 | 0.011 | 0.005 | 0.014 | 0.018 |
| `student@up_blocks[0]` | 0.235 | 0.099 | 0.043 | 0.014 | 0.008 | 0.018 | 0.024 |
| `student_paperhead@up_blocks[0]` | 0.233 | 0.097 | 0.044 | 0.015 | 0.008 | 0.019 | 0.024 |

### C — ridge R2, neighbourhood(mean+std within r, `|i-j|>8`) -> residue

| arm | 10 A | 15 A | 20 A | 30 A | 50 A |
|---|---|---|---|---|---|
| `esmc` | 0.061 | 0.043 | 0.034 | 0.020 | 0.018 |
| `esmc_all` | 0.096 | 0.061 | 0.044 | 0.023 | 0.021 |
| **rand** `mid_block` | 0.056 | 0.043 | 0.036 | 0.025 | 0.038 |
| **rand** `up_blocks[1]` | 0.071 | 0.052 | 0.042 | 0.022 | 0.027 |
| **rand** `up_blocks[0]` | 0.024 | 0.017 | 0.016 | 0.014 | 0.028 |
| `dec_t500@mid_block` | **0.252** | **0.222** | **0.176** | **0.082** | 0.004 |
| `dec_t900@up_blocks[0]` | 0.206 | 0.148 | 0.109 | 0.058 | 0.052 |
| `dec_t500@up_blocks[1]` | 0.172 | 0.116 | 0.087 | 0.045 | 0.040 |
| `student@up_blocks[0]` | 0.150 | 0.119 | 0.089 | 0.050 | 0.039 |
| `dec_t500@up_blocks[0]` | 0.131 | 0.100 | 0.078 | 0.040 | 0.029 |
| `dec_t261@up_blocks[0]` | 0.093 | 0.080 | 0.065 | 0.033 | 0.020 |

### D — centred-cosine correlogram, half-decay L of the FEATURE
`esmc` **8.27** (cos@3A 0.171) · `esmc_all` 8.83 (0.223) · `raw` 8.19 (0.109) ·
`up_blocks[1]` 8.67 (0.365) · `mid_block` **11.80 (0.360)** · `up_blocks[0]` t261/t500/t900
11.95 / **12.00** / 11.55 · `cou_best` 12.03 · `student` 12.01 · `paperhead` 12.07.
**Random controls: `up_blocks[0]` L undefined (cos@3A −0.006), `mid_block` 8.70 (0.011),
`up_blocks[1]` 8.86 (0.054).** Read `cos@ref`, not L, for the controls: their L is the decay of
noise off a ~zero baseline, not a correlation length. The discriminating quantity is the
adjacent-residue correlation itself -- real 0.18-0.37 vs random 0.01-0.05, a 5-30x gap. **Note the
controls confirm the local-frame caveat below in the strongest form: a random-weight net sees two
adjacent residues' boxes as essentially UNCORRELATED (0.011) even though the boxes overlap almost
completely in content, because they are rotated into different N-Ca-C frames.**

### Conclusions
1. **★ THE RIGHT BASELINE IS THE TAP-MATCHED RANDOM CONTROL, NOT ESM-C — and this reverses a
   tempting reading.** A RANDOM-WEIGHT CryoFM at `mid_block` already scores through-space 0.158 at
   0-5 A (ESM-C 0.087) and MATCHES ESM-C on the neighbourhood task (0.056 vs 0.061 at 10 A). A
   box-based density feature is spatially smoothed BY CONSTRUCTION; a per-residue PLM embedding is
   not. So "density features carry more spatial information than ESM-C" is **partly architectural,
   not informational**, and any density-vs-ESM-C spatial comparison is unfair by default. This is
   the same coordinate-leakage class that inflated the pairwise contact probe to AUC 0.997.
   Running the control only at `up_blocks[0]` would have HIDDEN this: that tap's random arm is the
   weakest of the three (0.071), so leakage is strongly tap-dependent and must be matched per tap.
2. **★ Against the matched control the learned spatial content is real, large, and MID-RANGE.**
   `mid_block` real/random through-space: 0-5 A 0.2787/0.1577 (**1.77x**), 5-10 A 0.1758/0.0303
   (**5.80x**), 10-15 A 0.0888/0.0129 (**6.90x**), 15-20 A 0.0307/0.0042 (7.25x);
   neighbourhood@10 A 0.2523/0.0557 (4.53x). `up_blocks[0]` is the cleanest and its margin GROWS
   with distance: 5-10 A 0.0821/0.0104 (**7.87x**), 10-15 A 8.52x, 15-20 A **11.84x**.
   (Ratios are full-precision; computing them from the rounded table above gives 5.9/6.8/8.2,
   which is what an earlier draft of this section quoted -- use these.) The margin is SMALLEST where
   leakage is largest (0-5 A, where adjacent boxes overlap most) and largest at 5-20 A. **CryoFM
   genuinely encodes through-space structure at exactly the 5-20 A scale the sibling found ESM
   lacks** — the structure-native-encoder prediction is confirmed, on the honest baseline.
3. **But the shape of the curve is unchanged, and coarse is still nearly empty.** Every arm without
   exception preserves the sibling's ordering 50 A < 25 A < 10 A < local, and every arm's
   through-space R2 is ~0 by 20-25 A. The best 50 A band is 0.78% of within-protein variance
   (control 0.140%): 5.56x the control, still under 1%. A structure-native encoder shifts the curve
   **up and to the right; it does not change its shape.**
4. **★ NEW — the diffusion timestep is a SPATIAL-SCALE knob.** At `up_blocks[0]`, t=261 -> 500 -> 900
   is monotone in every spatial measure: through-space 0-5 A 0.149 -> 0.216 -> **0.333**, 5-10 A
   0.056 -> 0.082 -> **0.133**, neighbourhood@10 A 0.093 -> 0.131 -> **0.206**, 50 A band 0.360 ->
   0.356 -> 0.391. **Higher t = more spatial context.** The 2026-08-27 sweep tuned t for pose
   invariance and sequence-trackability and never measured spatial range; this is a separate,
   actionable axis for any downstream use wanting neighbourhood-scale features.
5. **The tap axis dominates and matches the MEASURED receptive field.** `mid_block` (perturbation
   half-decay ~15 A) > `up_blocks[0]` (~10 A) > `up_blocks[1]` (~4-5 A) on every through-space
   measure, and correlogram L 11.80 / 12.00 / 8.67 against ESM-C 8.27. Independent confirmation of
   the 2026-08-26 single-voxel-perturbation numbers from a completely different measurement.
6. **The CleanDIFT student carries MORE spatial context than its nominal teacher.**
   `student@up_blocks[0]` sits between `dec_t500` and `dec_t900` everywhere: through-space 5-10 A
   0.099 vs 0.082 / 0.133; neighbourhood@10 A 0.150 vs 0.131 / 0.206. Both students
   (`student`, `student_paperhead`) agree to ~0.001, so this is not seed noise. Consistent with
   distilling ACROSS the noise schedule — the student inherits context from the high-`t` teachers.
   Note the sign contrast with the all-taps result (CleanDIFT HURTS SS at `up_blocks[0]`,
   −0.016..−0.023): distillation appears to trade a local readout for spatial context at that tap.
   Stated as a contrast, not a controlled comparison — that `dec_best` was val-SELECTED over t.
7. **`cou_best` (coupled) is WORSE than decoupled at the same t** (0.176 vs 0.216 at 0-5 A, 0.076 vs
   0.082 at 5-10 A). Adding the sampled noise field costs spatial content. Consistent with the
   2026-08-27 warning that coupled and decoupled features at matched t are nearly unrelated.

### Caveats (all measured, not assumed)
- **The 60-residue subsample inflates the 10 A band and halves short-range predictability.**
  `esmc` (matched) vs `esmc_all` (full density): 10 A band 5.215 vs 3.350, local 76.59 vs 84.37,
  through-space 0-5 A 0.087 vs 0.155, neighbourhood@10 A 0.061 vs 0.096. Every density arm carries
  the SAME handicap, so density-vs-density comparisons are clean; density-vs-ESM-C must use the
  matched `esmc` row, which understates ESM-C relative to a full-density extraction.
- **Local backbone frames actively SUPPRESS measured spatial correlation.** `rand_student@up_blocks[0]`
  has centred cosine **−0.006 between adjacent residues** despite their 64^3 boxes overlapping almost
  completely: adjacent residues have very different N-Ca-C frames and CryoFM is not
  rotation-equivariant, so the boxes are near-identical in CONTENT but differently oriented. The
  density numbers here are therefore, if anything, understated relative to a lab-frame extraction.
- **The 30-50 A uptick is partly a large-protein/geometry artifact**, exactly as the sibling logged.
  It is present in the random control too (0.007/0.009) but 4-5x larger for real features
  (0.034/0.042), so it is not purely artifact either.
- **Analysis D's BAND columns are metric-limited and should not be read as representation
  properties.** Per-protein mean removal caps measurable L at ~diameter/4, so the 50 A and 25 A bands
  read 15.2 / 15.6 for EVERY arm including random weights and raw voxels; `esmc_all` reads 20.3 /
  20.4 purely because its chains are fully sampled. Only the `feature` column discriminates. This is
  an explicit, direct demonstration of the ceiling the sibling project flagged.
- Simulated-vs-experimental is untouched here: these are the o5 latents, extracted from raw
  Cryo2StructData maps, so the OOD caveat does not apply — but the 15.1% CryoFM2-pretrain overlap
  of that corpus does.

### Two methodology fixes worth carrying forward
- **A fixed ridge alpha grid is invalid across these arms.** Feature RMS spans **0.036 (`esmc`) to
  17.0 (`up_blocks[0]`), a 480x range**, so absolute penalties regularise the arms by wildly
  different amounts; the first pass produced R2 −0.06..−0.17 in every long-range density bin purely
  because the largest absolute alpha was still far too small. Now `alpha = a * trace(XtX)/D`.
  **Any future cross-representation probe in this project must use a scale-relative penalty.**
- **BLAS thread oversubscription dominated runtime.** The 60x60 smoothing kernels are tiny, so on a
  96-core box the analysis spent all its time in OpenMP barriers: analysis A took >60 s at the
  default thread count and **7.1 s at 8 threads**. The scripts now pin to 8 before importing numpy.

### E — ISOLATED NONLINEAR GAP (job 3502608, GPU, 6 arms) + FIGURES
`results/o7_nonlinear_gap.json`. Fit ridge FIRST, then a val-tuned MLP to the ridge
RESIDUAL; **gap = combined R2 − ridge R2 = the nonlinear signal with the linear part
removed.** The sibling's fairest test, and the check on whether the ridge-based tables
above understate the spatial content. MLP grid hidden ∈ {1024, 2048} × wd ∈ {1e-4, 1e-3},
selected on val, so it is regularisation-fair — the sibling found an untuned MLP scored
BELOW ridge on restricted bins and that dip was a fitting artifact, not masked structure.

**Through space (`|i-j|>8`), gap by 3D distance — nil for every arm:**

| arm | 0-5 | 5-10 | 10-15 | 15-20 | 30-40 | 40-50 |
|---|---|---|---|---|---|---|
| `esmc` | −0.221* | −0.024 | −0.009 | −0.009 | −0.008 | −0.019 |
| **rand** `mid_block` | +0.007 | −0.004 | −0.002 | −0.002 | −0.002 | −0.001 |
| `dec_t500@mid_block` | −0.000 | +0.001 | +0.002 | +0.003 | +0.004 | +0.002 |
| `dec_t900@up_blocks[0]` | +0.002 | +0.002 | +0.003 | +0.002 | +0.005 | +0.005 |
| `student@up_blocks[0]` | +0.000 | +0.001 | +0.001 | +0.001 | +0.003 | +0.007 |

\* 554 test pairs — an overfit artifact on a tiny bin, not a finding. Excluded from the plot's y-range.

**Neighbourhood → residue, ridge → combined (gap):**

| arm | 10 Å | 20 Å | 30 Å | 50 Å |
|---|---|---|---|---|
| `esmc` | 0.061→0.007 (−0.054) | 0.034→0.034 (−0.000) | 0.020→0.019 (−0.001) | 0.018→0.015 (−0.003) |
| **rand** `mid_block` | 0.056→0.050 (−0.006) | 0.036→0.042 (+0.006) | 0.025→0.039 (+0.014) | **0.038→0.072 (+0.034)** |
| `dec_t500@mid_block` | 0.252→0.253 (+0.001) | 0.176→0.183 (+0.007) | 0.082→0.097 (+0.015) | **0.004→0.082 (+0.078)** |
| `dec_t900@up_blocks[0]` | 0.206→0.208 (+0.002) | 0.109→0.129 (+0.020) | 0.058→0.086 (+0.028) | **0.052→0.113 (+0.061)** |
| `student@up_blocks[0]` | 0.150→0.150 (+0.001) | 0.089→0.103 (+0.014) | 0.050→0.073 (+0.023) | 0.039→0.097 (+0.058) |

**Sequence separation, ridge → combined (gap), |i−j| = 1-2:** `esmc` 0.318→0.281
(**−0.036**) · **rand** 0.238→0.236 (−0.002) · `dec_t500@mid_block` 0.347→0.350 (+0.003)
· `dec_t500@up_blocks[0]` 0.312→0.342 (**+0.030**) · `dec_t900@up_blocks[0]` 0.425→0.465
(**+0.040**) · `student@up_blocks[0]` 0.354→0.390 (**+0.036**).

### Conclusions from the gap
8. **★ The ridge-based tables above are NOT understating anything.** Through space the gap
   is +0.000 to +0.005 for every density arm and negative for ESM-C, at every distance.
   The residue↔residue relation is LINEAR — the sibling's conclusion, now confirmed on a
   structure-native encoder as well. Conclusions 1-7 stand as measured; a nonlinear probe
   would not change them.
9. **The one place a gap appears is WIDE-radius neighbourhood pooling, and the control
   takes most of it.** At r = 50 Å the linear probe collapses (`mid_block` ridge 0.004)
   while the MLP recovers 0.082 — but the random-weight control goes 0.038 → **0.072**, so
   the two end up nearly level and the *learned* margin is only ~0.010. `dec_t900` is the
   exception with a real margin (0.113 vs 0.072). Reading the gap alone (+0.078 vs +0.034)
   would have doubled the apparent effect; **the control has to be applied to the gap, not
   just to the ridge.**
10. **★ The sibling's "the only genuine nonlinearity is sequence-adjacent" REPRODUCES — but
   in the DENSITY features, not in ESM-C.** At |i−j| = 1-2 the density arms gain
   **+0.030 / +0.040 / +0.036** (t500 / t900 / student at `up_blocks[0]`) against a random
   control of −0.002, so it is learned. `mid_block` shows almost none (+0.003), consistent
   with its 15 Å field blurring past the covalent neighbourhood.
11. **ESM-C's sequence-adjacent gap is −0.036 here, where the sibling measured +0.082.**
   This does NOT reproduce, and the likely reason is the sampling, not the model: our
   60-residue-per-chain subsample makes |i−j| ≤ 2 pairs rare (5,570 test pairs drawn from
   230 chains), so the MLP overfits a thin bin. **Do not read this as refuting the sibling
   result** — it is a non-reproduction with a known confound, and testing it properly needs
   a contiguous-residue extraction.

### Figures
`scripts/make_o7_figures.py` → `scripts/figs/o7_spatial/` (7 PNGs + README.md).
Palette is the dataviz skill's reference instance, the same one `make_team_figures.py`
uses, re-verified with `scripts/validate_palette.py`. Two inherited deliberate deviations:
the random-weight control is chrome grey (fails the chroma floor by construction — it is
not a hue; the "highlight one, gray the rest" case), and aqua at 2.74:1 triggers the
RELIEF RULE, so every aqua series carries a visible direct label and the tables here are
the accessible twin. One rule violation of mine, caught on inspection and fixed: direct
labels initially wore the series colour; text now wears ink tokens with a coloured dot
carrying identity.

### ALL-LAYER CORRELOGRAM (2026-09-03) — the CleanDIFT student swept across every block
`probes/o7_alltap_extract.py` (job 3527496, GPU, 1h48m) re-forwards the SAME residues through the
student and captures every named block output -- `conv_in`, `down_blocks[0..3]`, `mid_block`,
`up_blocks[0..3]` -- plus the random-weight control at all ten. Residue selection is the same
verified rng replay (1,484/1,484 chains, 0 mismatches, 16 pre-rng skips). Correlogram by
`probes/o7_alltap_correlogram.py`, 1,482 chains. Figure `06b_correlogram_layers.png`.

**Adjacent-residue centred cosine (~3 Å bin), student / random control, and L (Å) where meaningful:**

| layer | Å/token | student | random | L |
|---|---|---|---|---|
| `conv_in` | 1.5 | 0.290 | 0.232 | 8 |
| `down_blocks[0]` | 3 | 0.283 | 0.210 | 9 |
| `down_blocks[1]` | 6 | 0.348 | 0.128 | 9 |
| `down_blocks[2]` | 12 | 0.316 | 0.069 | 12 |
| **`down_blocks[3]`** | 12 | **0.511** | 0.030 | 12 |
| `mid_block` | 12 | 0.416 | 0.024 | 12 |
| `up_blocks[0]` | 6 | 0.193 | **0.011** | 12 |
| `up_blocks[1]` | 3 | 0.347 | 0.052 | 9 |
| `up_blocks[2]` | 1.5 | 0.271 | 0.102 | 8 |
| `up_blocks[3]` | 1.5 | 0.362 | 0.249 | 9 |

1. **★ The ENCODER bottom, not the decoder, is where spatial correlation peaks.** `down_blocks[3]`
   (0.511) and `mid_block` (0.416) are the top two, and their controls are the smallest (0.030,
   0.024) — a 17x and 17x margin. The three taps stored in the o5 parts (`mid_block`,
   `up_blocks[0]`, `up_blocks[1]`) do NOT contain the maximum; `down_blocks[3]` was never extracted
   before and beats all of them.
2. **Half-decay L is coarse-grained but consistent:** 12 Å for every 12 Å/token tap plus
   `up_blocks[0]`, 8-9 Å for the 1.5-3 Å/token taps. It tracks Å/token, not depth.
3. **★ L IS AN INTERPRETIVE TRAP AT LOW CORRELATION and the first version of this figure fell into
   it.** L is a ratio to each arm's OWN short-range value, so an arm with no correlation still
   reports one — the random control's L (11.8 at `down_blocks[1]`, 13.5 at `down_blocks[2]`)
   EXCEEDS the student's there purely because its cos is 0.13 / 0.07, i.e. it is measuring the decay
   of noise. The figure now plots `cos@ref`, which has no such trap, with L annotated only where
   cos ≥ 0.10. **Never rank arms by L without showing the reference correlation.**
4. **The random control is U-shaped in depth** (0.23 → 0.011 at `up_blocks[0]` → 0.25 at
   `up_blocks[3]`): near the input and output an untrained net still passes through smooth density,
   but in the abstract middle it destroys all inter-residue correlation. So the leakage floor is
   strongly depth-dependent, which is exactly why the control has to be matched per tap.
5. `up_blocks[0]` — the tap the ESM-C alignment targets — has the LOWEST student correlation of the
   deep taps (0.193) but also the lowest control (0.011), so its 17x margin is the cleanest in the
   sweep. Consistent with it being the best per-residue tap under a local frame.

**Cross-pipeline check, and one thing NOT to compare.** The all-tap sweep and the original
three-tap correlogram overlap at four arms, computed by two independently written scripts. The
pretrained arm agrees to **4.5e-7** (`student@up_blocks[0]`: max|Δcurve| 4.5e-7, Δref 4.2e-8,
ΔL 9.2e-6) — a genuine cross-validation of both paths, now enforced by an assertion in
`scripts/make_o7_figures.py` where the two files are merged. The `rand_student` arms, by contrast,
disagree by ~1e-2 (mid_block ref_cos **0.011 vs 0.024**, up_blocks[0] 1.7e-2) because each
extraction instantiated **its own random weights** — expected, not a bug, and it changes no
conclusion since both draws are near zero against a student at 0.19-0.51. But **never quote a
random-control number across the two files**, and never difference them: use the control from the
same extraction as the arm it prices. Figure 06 was rebuilt on this basis: every density arm in it is now the SAME
student extraction (`down_blocks[3]`, `mid_block`, `up_blocks[0]` — the deep middle of the U)
with that extraction's own control, instead of mixing `dec_t500` teacher taps with a student
tap. The depth ordering it shows is therefore a property of depth, not of which arm was
plotted: 0.511 → 0.416 → 0.193 past the bottleneck, at a near-constant L ≈ 11.8–12.0 Å.

**Performance note worth keeping:** the generic `--analyses D` path took ~10 min PER ARM because it
recomputes `compute_pyramid_bands` (three Gaussian smoothings per chain) for 5 series. The band
columns are metric-limited and carry no layer information, so the dedicated script computes only
the `feature` series and finishes all 20 arms in **under a minute**. Measured, not guessed: I/O is
1,484 files in ~6 s, so reads were never the bottleneck.
