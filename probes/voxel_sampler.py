"""Voxel selection and soft residue targets for the DINO.txt voxel objective.

TWO POOLS, AND THE MODEL-FREE ONE DOES NOT YET WORK. Measured 2026-08-31.

The plan wanted voxel selection to be model-free (density percentile) so that
training and inference draw from the same pool and §8.6's "input: a map, no
atomic model" holds end to end. **Measured, that criterion fails on real maps**
and the number is not marginal. Fraction of selected voxels actually within 5 A
of a heavy atom, 8 maps:

    criterion          25414 0322 11263 31059 4873 27276 33528 10760
    raw p90             0.16 0.22  0.27  0.81 0.16   0.26  0.29   0.17
    blur(3) p99         0.78 0.05  0.20  0.99 1.00   0.51  0.86   0.31

Diagnosis, and it is NOT a coordinate bug -- atoms sit at the **98-99.9th
percentile** of density in 7 of 8 maps, so the transform is right:
  * **True protein occupancy is only 2.3-17%** (median ~4%), so a p90 threshold
    is ~95% solvent before anything else goes wrong.
  * Some maps contain artifacts BRIGHTER than protein. EMD-0322 has box
    p99 = 2.86 against a median atom density of 1.61, so the top 1% misses the
    protein entirely -- which is why blurring makes it *worse* (0.22 -> 0.05).
  * EMD-11263 is a separate case: its atoms sit at the 61st percentile, i.e. the
    model is barely in density at all. Worth flagging in any per-map QC.
This is the same failure this project already logged for canonicalisation
("`vol > 0` thresholding is indefensible on experimental maps"), now quantified
for sampling.

CONSEQUENCE, stated plainly rather than papered over:
  * `pool="model"` (DEFAULT, training) selects foreground by distance to a heavy
    atom. The atomic model is a LABELLING INSTRUMENT here, exactly as it is for
    the target itself -- so this costs nothing that the target did not already
    cost, and it unblocks the G0 gate, which asks whether the alignment exists at
    all and does not depend on how voxels are chosen at deployment.
  * `pool="density"` is kept, model-free, for the deployment experiment. Making
    it work is an OPEN PROBLEM and the honest blocker on "no atomic model at
    inference". Until it is solved, that claim is not established.
Percentile selection remains safe for ORIENTATION either way (a percentile of the
value distribution is an isometry-invariant scalar); the problem is what it
selects, not that it biases pose.

PER-CHAIN QUOTAS ARE CLASS BALANCING, NOT A DISTRIBUTION SHIFT. Maps here hold a
median of 9 chains and up to 56, so volume-proportional sampling would let a few
large complexes dominate the gradient. The quota reweights WHICH candidates from
the (model-free) pool we train on; it does not change the pool, the input, or the
features. That is the same status as class weighting anywhere else.

Two further constraints, both from measured facts:

  * **Edge margin.** `extract_local_boxes` pads with zeros and 0 in preprocessed
    units is raw density 0.04, which is ABOVE background (raw 0 maps to -0.44), so
    an out-of-bounds sample gets a slab of mean density rather than vacuum.
  * **Minimum separation.** Adjacent feature cells are roughly half-redundant
    (4-5 A half-decay at `up_blocks[1]` against 3 A per cell), so densely sampling
    one map inflates the nominal count far above the effective sample size.
    Thinning to a minimum separation buys independent samples per unit of cache.
"""

from __future__ import annotations

import numpy as np

# 1.5 A/voxel, matching MODEL_VOXEL_SIZE. Imported lazily where needed so this
# module stays importable without torch/CUDA for the self-test.
BG_LABEL = -1


_NEIGH27 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     for c in (-1, 0, 1)], dtype=np.int64)


def _thin(coords: np.ndarray, min_sep: float, rng,
          existing: np.ndarray | None = None) -> np.ndarray:
    """Greedy Poisson-disk thinning: no two kept points closer than `min_sep`.

    Grid-hashed to stay O(27n) rather than O(n^2). ONE-PER-CELL IS NOT ENOUGH and
    the self-test catches it: two candidates either side of a cell boundary sit in
    different cells and can be arbitrarily close, so a pure occupancy hash gives
    no separation guarantee at all (measured min sep 1.00 for a requested 3.0).
    Cells are `min_sep` wide, so any point within `min_sep` must lie in one of the
    27 neighbouring cells -- checking those gives a real guarantee.
    """
    if min_sep <= 0:
        return coords
    order = rng.permutation(len(coords))
    cells: dict = {}
    keep = []
    # Seed the grid with points that are already committed, so a later pool is
    # separated from an earlier one too. Thinning the foreground and background
    # pools INDEPENDENTLY left cross-class pairs at 2.24 for a requested 3.0 --
    # each pool was internally valid and the union was not.
    if existing is not None:
        for q in existing:
            cells.setdefault(tuple((q / min_sep).astype(np.int64)), []).append(q)
    for j in order:
        p = coords[j]
        base = (p / min_sep).astype(np.int64)
        clash = False
        for off in _NEIGH27:
            got = cells.get(tuple(base + off))
            if got is None:
                continue
            for q in got:
                if np.sum((q - p) ** 2) < min_sep * min_sep:
                    clash = True
                    break
            if clash:
                break
        if clash:
            continue
        cells.setdefault(tuple(base), []).append(p)
        keep.append(j)
    return coords[np.sort(np.asarray(keep, dtype=np.int64))]


def sample_voxels(
    vol: np.ndarray,
    n_fg: int,
    rng,
    *,
    pool: str = "model",
    atom_vox: np.ndarray | None = None,
    fg_radius: float = 4.0,
    bg_radius: float = 12.0,
    bg_frac: float = 0.15,
    fg_pct: float = 90.0,
    bg_pct: float = 40.0,
    margin: int = 32,
    min_sep: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(coords [N,3] in voxel units, is_background [N] bool).

    `pool="model"` (default): foreground = within `fg_radius` A of a heavy atom,
    background = beyond `bg_radius` A. Requires `atom_vox` ([M,3], VOXEL units).
    The gap between the two radii is a deliberate dead zone -- the ambiguous shell
    belongs to neither class, which is the fix this project already paid to learn
    for its spatial contrastive loss.

    `pool="density"`: model-free percentile thresholding. See the module docstring
    for why this currently does not work; kept for the deployment experiment.

    `min_sep` is in voxels (3.0 voxels = 4.5 A ~ one `up_blocks[1]` half-decay).
    """
    vol = np.asarray(vol)
    sl = tuple(slice(margin, s - margin) for s in vol.shape)
    inner = vol[sl]
    if min(inner.shape) <= 0:
        raise ValueError(f"map {vol.shape} is too small for margin {margin}")

    if pool == "model":
        if atom_vox is None:
            raise ValueError('pool="model" requires atom_vox')
        return _sample_model(vol.shape, n_fg, rng, atom_vox, fg_radius, bg_radius,
                             bg_frac, margin, min_sep)
    if pool != "density":
        raise ValueError(f"unknown pool {pool!r}")

    lo, hi = np.percentile(inner, [bg_pct, fg_pct])
    # Candidate pools. Subsample the index arrays before thinning: a 256^3 map
    # has ~1.7e6 candidates above p90 and thinning all of them is wasted work.
    def pool(mask, want, existing=None):
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        take = min(len(idx), max(want * 40, 20000))
        idx = idx[rng.choice(len(idx), take, replace=False)] if len(idx) > take else idx
        idx = idx.astype(np.float64) + np.array([s.start for s in sl], dtype=np.float64)
        idx = _thin(idx, min_sep, rng, existing=existing)
        if len(idx) > want:
            idx = idx[rng.choice(len(idx), want, replace=False)]
        return idx

    n_bg = int(round(n_fg * bg_frac / max(1.0 - bg_frac, 1e-6)))
    fg = pool(inner >= hi, n_fg)
    # Background is thinned AGAINST the accepted foreground, so the union honours
    # min_sep rather than only each class separately.
    bg = pool(inner <= lo, n_bg, existing=fg)
    coords = np.concatenate([fg, bg]) if len(bg) else fg
    is_bg = np.concatenate([np.zeros(len(fg), bool), np.ones(len(bg), bool)])
    return coords, is_bg


def _sample_model(shape, n_fg, rng, atom_vox, fg_radius, bg_radius, bg_frac,
                  margin, min_sep):
    """Model-defined pools: near-atom foreground, far-from-atom background."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(atom_vox, dtype=np.float64))
    vs = 1.5
    lo = np.array([margin] * 3, dtype=np.float64)
    hi = np.array(shape, dtype=np.float64) - margin

    # Foreground: jitter around atoms rather than scanning the grid. Atoms ARE
    # the foreground region, so this samples it directly and costs O(n) instead
    # of O(box^3). Snap to the voxel lattice so cached features are grid-exact.
    a = np.asarray(atom_vox, dtype=np.float64)
    a = a[np.all((a >= lo) & (a < hi), axis=1)]
    if len(a) == 0:
        raise ValueError("no heavy atom is inside the edge margin")
    pick = a[rng.choice(len(a), min(len(a), max(n_fg * 30, 20000)), replace=True)]
    jit = rng.normal(0.0, fg_radius / vs / 2.0, pick.shape)
    fg = np.round(pick + jit)
    keep = np.all((fg >= lo) & (fg < hi), axis=1)
    fg = fg[keep]
    d, _ = tree.query(fg, k=1)
    fg = fg[d * vs <= fg_radius]
    fg = _thin(fg, min_sep, rng)
    if len(fg) > n_fg:
        fg = fg[rng.choice(len(fg), n_fg, replace=False)]

    # Background: uniform in the box, kept only if far from every atom.
    n_bg = int(round(n_fg * bg_frac / max(1.0 - bg_frac, 1e-6)))
    cand = rng.uniform(lo, hi, size=(max(n_bg * 200, 20000), 3))
    cand = np.round(cand)
    cand = cand[np.all((cand >= lo) & (cand < hi), axis=1)]
    d, _ = tree.query(cand, k=1)
    bg = cand[d * vs >= bg_radius]
    bg = _thin(bg, min_sep, rng, existing=fg)
    if len(bg) > n_bg:
        bg = bg[rng.choice(len(bg), n_bg, replace=False)]

    coords = np.concatenate([fg, bg]) if len(bg) else fg
    is_bg = np.concatenate([np.zeros(len(fg), bool), np.ones(len(bg), bool)])
    return coords, is_bg


def soft_targets(
    coords_vox: np.ndarray,
    inv: dict,
    origin_zyx: np.ndarray,
    *,
    sigma: float = 4.0,
    n_max: int = 8,
    voxel_size: float = 1.5,
    bg_weight: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft distribution over residues for each voxel, plus a background weight.

    Returns (res_idx [N, n_max] int32, weight [N, n_max] float32, w_bg [N]).
    `res_idx` is -1 where unused. Rows sum to 1 including `w_bg`.

    Residues are indexed by the per-map (sequence, position) table that
    `build_map_chains.py` writes, NOT by chain instance -- which is what makes
    this direction single-valued under homo-oligomer symmetry: a voxel on copy 3
    of a C12 ring has one correct target, not twelve.

    `d(v, i)` is the distance to residue i's NEAREST HEAVY ATOM, not to its Ca. A
    residue is an extended object and its side chain carries the density a voxel
    actually sits in.

    `bg_weight` is background's UNNORMALISED weight, and it must be well below 1.
    Residue weights are `exp(-d^2/2 sigma^2) <= 1`, so setting it to 1.0 (the first
    version) made background tie exactly with a residue at zero distance -- a voxel
    sitting on an atom was 50% background. At the default it takes over once the
    summed residue weight drops below 0.1, i.e. beyond
    `sigma*sqrt(2 ln(1/bg_weight))` ~ 8.6 A for sigma=4 -- a sensible solvent
    boundary rather than an arbitrary threshold.
    """
    from scipy.spatial import cKDTree

    atom_vox = (inv["atom_xyz"].astype(np.float64) -
                np.asarray(origin_zyx)[None]) / voxel_size
    tree = cKDTree(atom_vox)
    cutoff_vox = 3.0 * sigma / voxel_size          # 3 sigma, in voxel units

    n = len(coords_vox)
    res_idx = np.full((n, n_max), -1, dtype=np.int32)
    weight = np.zeros((n, n_max), dtype=np.float32)
    w_bg = np.ones(n, dtype=np.float32)

    neigh = tree.query_ball_point(coords_vox, cutoff_vox)
    atom_res = inv["atom_res"]
    for j, aid in enumerate(neigh):
        if not len(aid):
            continue                                 # pure solvent: w_bg stays 1
        aid = np.asarray(aid)
        d_ang = np.linalg.norm(atom_vox[aid] - coords_vox[j][None], axis=1) * voxel_size
        r = atom_res[aid]
        # nearest heavy atom PER RESIDUE
        order = np.argsort(d_ang)
        r, d_ang = r[order], d_ang[order]
        _, first = np.unique(r, return_index=True)
        r, d_ang = r[first], d_ang[first]
        w = np.exp(-(d_ang ** 2) / (2.0 * sigma ** 2))
        if len(w) > n_max:                            # keep the closest n_max
            k = np.argsort(-w)[:n_max]
            r, w = r[k], w[k]
        tot = w.sum() + bg_weight
        res_idx[j, :len(r)] = r
        weight[j, :len(w)] = w / tot
        w_bg[j] = bg_weight / tot
    return res_idx, weight, w_bg


def chain_quota(coords_vox: np.ndarray, inv: dict, origin_zyx: np.ndarray,
                *, voxel_size: float = 1.5) -> np.ndarray:
    """Per-voxel sampling weight that equalises CHAINS rather than volume.

    Training-time only. Assigns each voxel to its nearest chain and returns
    1/count for that chain, so a 56-chain complex does not swamp a monomer.
    Voxels with no chain within reach (solvent) get weight 1.
    """
    from scipy.spatial import cKDTree

    atom_vox = (inv["atom_xyz"].astype(np.float64) -
                np.asarray(origin_zyx)[None]) / voxel_size
    d, i = cKDTree(atom_vox).query(coords_vox, k=1)
    ch = np.where(np.isfinite(d), inv["atom_chain"][i], -1)
    cnt = {c: int((ch == c).sum()) for c in np.unique(ch)}
    return np.array([1.0 / max(cnt[c], 1) for c in ch], dtype=np.float32)


def _self_test() -> None:
    """Synthetic checks that would each be silent if wrong."""
    rng = np.random.default_rng(0)

    # A 128^3 map with a dense blob; background elsewhere.
    vol = rng.normal(-0.4, 0.05, (128, 128, 128)).astype(np.float32)
    zz, yy, xx = np.mgrid[0:128, 0:128, 0:128]
    blob = ((zz - 64) ** 2 + (yy - 64) ** 2 + (xx - 64) ** 2) < 20 ** 2
    vol[blob] += 2.0

    from scipy.spatial import cKDTree

    # --- density pool (model-free). Kept working even though it is not the
    # default: on this SYNTHETIC map there is no artifact brighter than the blob,
    # so it succeeds here. That is exactly why the real-map measurement in the
    # module docstring matters and a synthetic self-test cannot replace it.
    c, bg = sample_voxels(vol, 400, rng, pool="density", margin=32, min_sep=3.0)
    assert len(c) > 0 and c.shape[1] == 3
    assert (c >= 32).all() and (c < 128 - 32).all(), "edge margin violated"
    inb = blob[tuple(c.astype(int).T)]
    assert inb[~bg].mean() > 0.9, f"foreground off-blob: {inb[~bg].mean():.2f}"
    assert inb[bg].mean() < 0.1, f"background on-blob: {inb[bg].mean():.2f}"
    d, _ = cKDTree(c).query(c, k=2)
    assert d[:, 1].min() >= 3.0 - 1e-9, f"min sep {d[:,1].min():.2f}"

    # --- model pool (the default). Atoms scattered through the blob.
    at = np.argwhere(blob).astype(np.float64)
    at = at[rng.choice(len(at), 500, replace=False)]
    c2, bg2 = sample_voxels(vol, 400, rng, pool="model", atom_vox=at, margin=32,
                            min_sep=3.0, fg_radius=4.0, bg_radius=12.0)
    assert (c2 >= 32).all() and (c2 < 128 - 32).all(), "edge margin violated"
    dm, _ = cKDTree(at).query(c2, k=1)
    dm *= 1.5
    assert dm[~bg2].max() <= 4.0 + 1e-6, f"fg beyond fg_radius: {dm[~bg2].max():.2f}"
    assert dm[bg2].min() >= 12.0 - 1e-6, f"bg inside bg_radius: {dm[bg2].min():.2f}"
    assert bg2.sum() > 0 and (~bg2).sum() > 0, "a class is empty"
    d2, _ = cKDTree(c2).query(c2, k=2)
    assert d2[:, 1].min() >= 3.0 - 1e-9, f"model-pool min sep {d2[:,1].min():.2f}"

    # Soft targets: two residues, one voxel sitting on the first.
    inv = {
        "atom_xyz": np.array([[0, 0, 0], [0, 0, 6.0]], dtype=np.float32),  # 6 A apart
        "atom_res": np.array([0, 1], dtype=np.int32),
        "atom_chain": np.array([0, 1], dtype=np.int32),
    }
    v = np.array([[0.0, 0.0, 0.0]])                       # on residue 0
    ri, w, wb = soft_targets(v, inv, np.zeros(3), sigma=4.0)
    tot = w[0].sum() + wb[0]
    assert abs(tot - 1.0) < 1e-5, f"rows must sum to 1, got {tot}"
    j0 = list(ri[0]).index(0)
    assert w[0, j0] > 5 * wb[0], (
        f"a voxel ON an atom must decisively beat BG: {w[0,j0]:.3f} vs {wb[0]:.3f}")
    j1 = list(ri[0]).index(1)
    assert w[0, j0] > w[0, j1], "nearer residue must get more mass"
    # exp(-36/32) = 0.3247 relative to exp(0) = 1
    assert abs(w[0, j1] / w[0, j0] - np.exp(-36 / 32)) < 1e-4, "kernel is not Gaussian"

    # A far-away voxel must be almost pure background.
    far = np.array([[0.0, 0.0, 100.0]])
    _, w2, wb2 = soft_targets(far, inv, np.zeros(3), sigma=4.0)
    assert wb2[0] > 0.999 and w2[0].sum() < 1e-3, "distant voxel is not background"

    # Chain quota equalises chains, not voxels.
    many = np.concatenate([np.zeros((10, 3)), np.full((2, 3), 6.0)])
    q = chain_quota(many, inv, np.zeros(3))
    assert abs(q[:10].sum() - q[10:].sum()) < 1e-5, "quota does not equalise chains"

    print("voxel_sampler self-test OK")


if __name__ == "__main__":
    _self_test()
