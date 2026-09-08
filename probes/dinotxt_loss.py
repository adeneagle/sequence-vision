"""Soft-target cross-entropy for voxel <-> residue alignment.

VOXEL-ANCHORED IS PRIMARY; the residue-anchored dual is a low-weight auxiliary.
Four reasons, settled before implementing:

1. **It matches the deliverable.** E1 asks "for each voxel, which of these ~9
   sequences owns it", which is exactly this direction's inference shape.
2. **The denominator is natural.** v2r normalises over RESIDUES, a set the data
   gives us. r2v normalises over SAMPLED VOXELS, a set we chose -- so its loss
   scale is a function of our own sampling density, a nuisance baked into the
   objective.
3. **v2r is single-valued under homo-oligomer symmetry.** With residues indexed
   by (sequence, position), a voxel on copy 3 of a C12 ring has ONE correct
   target. Residue -> voxel has twelve equally-correct answers, irreducibly.
   Symmetry has broken four separate things in this project already.
4. **Background only exists voxel-anchored.** A voxel can belong to no residue; a
   resolved residue always has density somewhere. The directions are not
   symmetric objects.

Part 6.2 reached for the dual to fix volume-weighting (large complexes dominating
because they have more voxels). That conflates the LOSS DIRECTION with SAMPLE
WEIGHTING: the second is the sampler's job, and `voxel_sampler.chain_quota`
does it directly. The dual is kept because it is nearly free (the same similarity
matrix, normalised along the other axis) and because a large v2r/r2v asymmetry is
a free diagnostic that the sampler, not the representation, is at fault.

SOFT TARGETS, NO MASKS. The distance kernel is used as a target distribution
rather than thresholded into positive/dead-zone/negative sets. Nothing is
thresholded, so the "0.99 sigma attracts, 1.01 sigma repels" discontinuity that
the sibling project's `spatial_contrastive_loss` was built on cannot arise, and
false negatives are down-weighted smoothly instead of being asserted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def v2r_loss(h_vox, g_res, bg, res_idx, weight, w_bg, scale, sample_w=None,
             vox_map=None, res_map=None, lambda_within: float = 0.0):
    """Voxel -> residue soft-target cross-entropy.

    h_vox   [V, d]  L2-normalised voxel embeddings
    g_res   [R, d]  L2-normalised residue embeddings for ALL maps in the batch
    bg      [d]     background embedding
    res_idx [V, K]  GLOBAL residue indices into g_res, -1 = padding
    weight  [V, K]  target mass on each, rows sum to 1 together with w_bg
    w_bg    [V]     target mass on background
    scale   scalar  1/tau
    sample_w [V]    optional per-voxel weight (chain quota); mean-normalised

    Residues of OTHER maps in the batch appear in the denominator with zero
    target mass -- that is where the contrastive pressure comes from.

    `lambda_within` ADDS a second cross-entropy whose denominator is restricted to
    the voxel's OWN map. The global denominator's own-map share is EXACTLY
    1/batch_maps -- an identity (sum_j R_j/R_tot = 1 over M maps), not an
    empirical fact -- so at the default batch_maps=8 it is **12.5% own / 87.5%
    cross** (measured over real cluster-disjoint batches on the 1147-map cache:
    2 maps 50.0%, 4 maps 25.0%, 8 maps 12.5%, 16 maps 6.3% own). An earlier
    version of this docstring quoted 5.6%/94.4% as an 8-map measurement; that is
    a 16-MAP figure and was mislabeled. So the
    global term's gradient is dominated by cross-structure negatives -- different
    fold, different composition, different instrument -- which are the EASY ones,
    while E1 scores exactly the within-map discrimination that makes up the
    remaining 12.5%. The within-map term is that discrimination as its own objective, at a
    weight we control instead of one that falls out of batch composition.
    Cross-structure negatives are kept because the retrieval endpoint (E2) needs
    them; this rebalances rather than replaces.
    """
    logits = scale * torch.cat([h_vox @ g_res.t(), (h_vox @ bg).unsqueeze(1)], dim=1)
    logp = F.log_softmax(logits, dim=1)                       # [V, R+1]

    valid = res_idx >= 0
    idx = res_idx.clamp(min=0)
    # gather the log-prob of each target residue; zero out padding
    lp = torch.gather(logp[:, :-1], 1, idx)
    ce = -(weight * valid * lp).sum(1) - w_bg * logp[:, -1]

    if lambda_within > 0.0:
        if vox_map is None or res_map is None:
            raise ValueError("lambda_within > 0 requires vox_map and res_map")
        # Restrict the denominator to own-map residues (+ background) by masking.
        same = vox_map.unsqueeze(1) == res_map.unsqueeze(0)          # [V, R]
        lg = logits.clone()
        lg[:, :-1] = lg[:, :-1].masked_fill(~same, float("-inf"))
        lpw = F.log_softmax(lg, dim=1)
        lpw_g = torch.gather(lpw[:, :-1], 1, idx)
        ce_w = -(weight * valid * lpw_g).sum(1) - w_bg * lpw[:, -1]
        ce = ce + lambda_within * ce_w

    if sample_w is not None:
        sample_w = sample_w / sample_w.mean().clamp(min=1e-12)
        ce = ce * sample_w
    return ce.mean()


def v2r_loss_blocked(h_vox, g_res, bg, res_idx, weight, w_bg, scale,
                     vox_map, res_map, n_cross: int, rng, sample_w=None):
    """v2r with the CROSS-STRUCTURE negatives SUBSAMPLED, per map.

    THIS IS THE SAMPLER-SIDE FIX, and it is preferable to the `lambda_within`
    loss term it replaces. That term put own-map residues in BOTH denominators,
    so they received gradient from two differently-normalised losses -- not a
    rebalance so much as own-map counted twice.

    WHY THE OBVIOUS VERSION DOES NOT WORK. "Cross-structure" is VOXEL-RELATIVE:
    map j's voxels need map j's residues present, so the union over all voxels of
    their own-map sets is the entire batch. Subsampling the SHARED residue block
    would delete some other map's positives. Measured own-map share is ~1/batch_maps
    (1 map 99.9%, 2 maps 50.0%, 8 maps 12.5%), so `batch_maps` is a knob -- but a
    coarse one that trades the ratio against gradient quality, and at batch_maps=1
    there are no cross-structure negatives left at all, which kills E2 calibration.

    So: one softmax PER MAP over [that map's own residues] + [n_cross residues
    sampled from the OTHER maps in the batch] + [background]. The ratio is then set
    by `n_cross` independently of `batch_maps`, there is one proper cross-entropy
    per voxel, and it is cheaper -- sum_j V_j*(R_j + n_cross) against V*R, ~4x less
    at batch_maps=8.

    Relies on residues being CONTIGUOUS per map in `g_res`, which `assemble`
    guarantees (it appends one map's block at a time); asserted below.
    """
    total = h_vox.new_zeros(())
    n_tot = 0
    vm = vox_map.detach().cpu().numpy()
    rm = res_map.detach().cpu().numpy()
    for j in np.unique(vm):
        vsel = np.nonzero(vm == j)[0]
        own = np.nonzero(rm == j)[0]
        if len(vsel) == 0 or len(own) == 0:
            continue
        assert own.max() - own.min() + 1 == len(own), (
            f"map {j}'s residues are not contiguous in g_res; the local index "
            "remapping below would be wrong")
        start = int(own.min())
        cross = np.nonzero(rm != j)[0]
        if n_cross > 0 and len(cross) > n_cross:
            cross = rng.choice(cross, n_cross, replace=False)
        cand = np.concatenate([own, cross]) if len(cross) else own
        ci = torch.from_numpy(cand).to(h_vox.device)
        vi = torch.from_numpy(vsel).to(h_vox.device)

        hj = h_vox[vi]
        logits = scale * torch.cat(
            [hj @ g_res[ci].t(), (hj @ bg).unsqueeze(1)], dim=1)
        logp = F.log_softmax(logits, dim=1)

        ridx, wj, wbj = res_idx[vi], weight[vi], w_bg[vi]
        valid = ridx >= 0
        # global -> position within `cand`: own occupies [0, len(own)) in order
        local = (ridx - start).clamp(min=0, max=len(own) - 1)
        assert bool(((ridx[valid] >= start) &
                     (ridx[valid] < start + len(own))).all()), (
            f"map {j}: a positive residue lies outside the map's own block")
        lp = torch.gather(logp[:, :-1], 1, local)
        ce = -(wj * valid * lp).sum(1) - wbj * logp[:, -1]
        if sample_w is not None:
            sw = sample_w[vi]
            ce = ce * (sw / sw.mean().clamp(min=1e-12))
        total = total + ce.sum()
        n_tot += len(vsel)
    return total / max(n_tot, 1)


def r2v_loss(h_vox, g_res, res_idx, weight, scale, min_mass: float = 1e-6):
    """Residue -> voxel dual. Normalises over the SAMPLED voxels, so it inherits
    a dependence on sampling density -- hence the low default weight.

    Residues with no positive voxel in the batch are DROPPED, not given a uniform
    target: their conditional is undefined, and filling it with uniform mass would
    train every unobserved residue toward the mean voxel.
    """
    V, R = h_vox.shape[0], g_res.shape[0]
    # scatter the sparse (voxel, residue) weights into a dense [R, V] target
    tgt = h_vox.new_zeros(R, V)
    valid = res_idx >= 0
    vi = torch.arange(V, device=h_vox.device).unsqueeze(1).expand_as(res_idx)
    tgt.index_put_((res_idx.clamp(min=0)[valid], vi[valid]), weight[valid],
                   accumulate=True)

    mass = tgt.sum(1)
    keep = mass > min_mass
    if not bool(keep.any()):
        return h_vox.new_zeros(())
    tgt = tgt[keep] / mass[keep].unsqueeze(1)
    logp = F.log_softmax(scale * (g_res[keep] @ h_vox.t()), dim=1)
    return -(tgt * logp).sum(1).mean()


def dinotxt_loss(model, h_vox, g_res, res_idx, weight, w_bg, sample_w=None,
                 lambda_dual: float = 0.3, vox_map=None, res_map=None,
                 lambda_within: float = 0.0, n_cross: int = 2000, rng=None):
    """`n_cross > 0` uses the blocked/subsampled path (preferred, see
    `v2r_loss_blocked`). `n_cross = 0` uses the full-batch denominator, and
    `lambda_within` is then the older two-term rebalance, kept for ablation."""
    scale = model.scale()
    bg = model.bg_embedding()
    if n_cross and vox_map is not None:
        l_v2r = v2r_loss_blocked(h_vox, g_res, bg, res_idx, weight, w_bg, scale,
                                 vox_map, res_map, n_cross,
                                 rng or np.random.default_rng(0), sample_w)
    else:
        l_v2r = v2r_loss(h_vox, g_res, bg, res_idx, weight, w_bg, scale, sample_w,
                         vox_map=vox_map, res_map=res_map,
                         lambda_within=lambda_within)
    l_r2v = (r2v_loss(h_vox, g_res, res_idx, weight, scale)
             if lambda_dual > 0 else h_vox.new_zeros(()))
    return l_v2r + lambda_dual * l_r2v, {"v2r": float(l_v2r), "r2v": float(l_r2v)}


def collapse_stats(h: torch.Tensor) -> dict:
    """RAW off-diagonal cosine and relative variation -- NOT effective rank.

    Effective rank is computed on CENTRED data and read 197.7 on a provably
    degenerate set in this project, so it does not detect collapse. Raw cosine
    near 1 with relative variation near 0 is what collapse actually looks like.
    """
    with torch.no_grad():
        n = min(len(h), 512)
        x = h[:n]
        c = (x @ x.t())
        off = c[~torch.eye(n, dtype=torch.bool, device=x.device)]
        mu = x.mean(0, keepdim=True)
        relvar = float((x - mu).norm(dim=1).mean() / mu.norm().clamp(min=1e-12))
        return {"offdiag_cos": float(off.mean()), "rel_variation": relvar}


def _self_test() -> None:
    torch.manual_seed(0)
    from probes.dinotxt_model import DinoTxt

    d, V, R, K = 16, 6, 5, 3
    m = DinoTxt(d_vox=8, d_seq=8, d=d, hidden=16, depth=1)
    scale = m.scale()
    bg = m.bg_embedding()

    # Build a target: voxel v prefers residue v % R, with a little mass elsewhere.
    res_idx = torch.full((V, K), -1, dtype=torch.long)
    weight = torch.zeros(V, K)
    for v in range(V):
        res_idx[v, 0] = v % R
        res_idx[v, 1] = (v + 1) % R
        weight[v, 0], weight[v, 1] = 0.7, 0.2
    w_bg = 1.0 - weight.sum(1)
    assert torch.allclose(weight.sum(1) + w_bg, torch.ones(V))

    g = F.normalize(torch.randn(R, d), dim=-1)

    # (a) An ORACLE voxel embedding (equal to its dominant residue) must beat a
    #     random one. If it does not, the loss is not measuring what it claims.
    h_rand = F.normalize(torch.randn(V, d), dim=-1)
    h_oracle = F.normalize(g[res_idx[:, 0]] + 1e-3 * torch.randn(V, d), dim=-1)
    l_rand = float(v2r_loss(h_rand, g, bg, res_idx, weight, w_bg, scale))
    l_orac = float(v2r_loss(h_oracle, g, bg, res_idx, weight, w_bg, scale))
    assert l_orac < l_rand, f"oracle {l_orac:.4f} not better than random {l_rand:.4f}"

    # (b) Padding must be ignored: extending K with -1 columns changes nothing.
    pad_i = torch.cat([res_idx, torch.full((V, 4), -1, dtype=torch.long)], 1)
    pad_w = torch.cat([weight, torch.zeros(V, 4)], 1)
    l_pad = float(v2r_loss(h_rand, g, bg, pad_i, pad_w, w_bg, scale))
    assert abs(l_pad - l_rand) < 1e-6, f"padding changed the loss: {l_pad} vs {l_rand}"

    # (c) A pure-background voxel must be driven to the background embedding.
    bi = torch.full((1, K), -1, dtype=torch.long)
    bw = torch.zeros(1, K)
    on_bg = F.normalize(bg.unsqueeze(0) + 1e-3 * torch.randn(1, d), dim=-1)
    off_bg = F.normalize(-bg.unsqueeze(0) + 1e-3 * torch.randn(1, d), dim=-1)
    l_on = float(v2r_loss(on_bg, g, bg, bi, bw, torch.ones(1), scale))
    l_off = float(v2r_loss(off_bg, g, bg, bi, bw, torch.ones(1), scale))
    assert l_on < l_off, f"background not learnable: on {l_on:.4f} off {l_off:.4f}"

    # (d) Other maps' residues must act as negatives: appending unrelated
    #     residues to g (with no target mass) must INCREASE the loss.
    g_big = torch.cat([g, F.normalize(torch.randn(7, d), dim=-1)])
    l_neg = float(v2r_loss(h_rand, g_big, bg, res_idx, weight, w_bg, scale))
    assert l_neg > l_rand, "extra in-batch residues did not act as negatives"

    # (e) sample_w is mean-normalised, so uniform weights are a no-op.
    l_sw = float(v2r_loss(h_rand, g, bg, res_idx, weight, w_bg, scale,
                          sample_w=torch.full((V,), 3.0)))
    assert abs(l_sw - l_rand) < 1e-5, f"uniform sample_w changed the loss ({l_sw})"

    # (e2) The within-map term must use ONLY own-map residues in its denominator.
    #      Build 2 maps; adding far-away OTHER-map residues must leave the
    #      within-map term untouched while changing the global term.
    vmap = torch.zeros(V, dtype=torch.long)
    rmap = torch.zeros(R, dtype=torch.long)
    g2 = torch.cat([g, F.normalize(torch.randn(9, d), dim=-1)])
    rmap2 = torch.cat([rmap, torch.ones(9, dtype=torch.long)])
    only_w = lambda gg, rm: float(v2r_loss(
        h_rand, gg, bg, res_idx, weight, w_bg, scale, vox_map=vmap, res_map=rm,
        lambda_within=1.0)) - float(v2r_loss(
            h_rand, gg, bg, res_idx, weight, w_bg, scale))
    a_, b_ = only_w(g, rmap), only_w(g2, rmap2)
    assert abs(a_ - b_) < 1e-4, (
        f"within-map term changed when OTHER-map residues were added: {a_} vs {b_}")
    l_glob_a = float(v2r_loss(h_rand, g, bg, res_idx, weight, w_bg, scale))
    l_glob_b = float(v2r_loss(h_rand, g2, bg, res_idx, weight, w_bg, scale))
    assert l_glob_b > l_glob_a, "global term ignored the extra negatives"

    # (f) The dual runs, is finite, and drops residues with no positive voxel.
    l_d = float(r2v_loss(h_rand, g_big, res_idx, weight, scale))
    assert l_d > 0 and torch.isfinite(torch.tensor(l_d))

    # (g) Gradients reach both towers and the background vector.
    hv = m.encode_voxels(torch.randn(V, 8))
    gr = m.encode_residues(torch.randn(R, 8))
    loss, parts = dinotxt_loss(m, hv, gr, res_idx, weight, w_bg,
                               vox_map=torch.zeros(V, dtype=torch.long),
                               res_map=torch.zeros(R, dtype=torch.long))
    loss.backward()
    for n_, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no grad: {n_}"
    assert m.bg.grad.abs().sum() > 0, "background embedding gets no gradient"
    assert parts["r2v"] > 0

    # (g2) lambda_within without map indices must RAISE, not silently skip: a
    #      caller that forgets them would train the global term only and the
    #      within-map rebalance would vanish without any signal.
    try:
        dinotxt_loss(m, hv, gr, res_idx, weight, w_bg, lambda_within=1.0,
                     n_cross=0)
        raise SystemExit("lambda_within silently ignored missing map indices")
    except ValueError:
        pass

    # (i) BLOCKED path == FULL path when n_cross covers every cross residue.
    #     This is the equivalence test: if subsampling is implemented correctly,
    #     then not actually subsampling must reproduce the original loss exactly.
    V2, R2 = 8, 10
    vmap2 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    rmap2 = torch.tensor([0] * 5 + [1] * 5)
    ridx2 = torch.full((V2, 2), -1, dtype=torch.long)
    wt2 = torch.zeros(V2, 2)
    for v in range(V2):
        base = 0 if v < 4 else 5           # own-map block start
        ridx2[v, 0] = base + (v % 4)
        wt2[v, 0] = 0.8
    wbg2 = 1.0 - wt2.sum(1)
    h2 = F.normalize(torch.randn(V2, d), dim=-1)
    g3 = F.normalize(torch.randn(R2, d), dim=-1)
    l_full = float(v2r_loss(h2, g3, bg, ridx2, wt2, wbg2, scale))
    l_blk = float(v2r_loss_blocked(h2, g3, bg, ridx2, wt2, wbg2, scale,
                                   vmap2, rmap2, 99, np.random.default_rng(0)))
    assert abs(l_full - l_blk) < 1e-5, \
        f"blocked != full with no subsampling: {l_full:.6f} vs {l_blk:.6f}"

    # (j) Subsampling must REDUCE the loss (fewer negatives = easier softmax)
    #     and must never drop a positive -- the assert inside would fire.
    l_sub = float(v2r_loss_blocked(h2, g3, bg, ridx2, wt2, wbg2, scale,
                                   vmap2, rmap2, 1, np.random.default_rng(0)))
    assert l_sub < l_blk, f"subsampling did not reduce the loss: {l_sub} vs {l_blk}"

    # (k) Non-contiguous residue blocks must RAISE, not silently mis-index.
    try:
        v2r_loss_blocked(h2, g3, bg, ridx2, wt2, wbg2, scale, vmap2,
                         torch.tensor([0, 1] * 5), 99, np.random.default_rng(0))
        raise SystemExit("non-contiguous residue block silently accepted")
    except AssertionError:
        pass

    # (h) collapse_stats must FLAG a collapsed set and pass a healthy one.
    col = collapse_stats(F.normalize(torch.ones(64, d) + 1e-4 * torch.randn(64, d), dim=-1))
    hea = collapse_stats(F.normalize(torch.randn(64, d), dim=-1))
    assert col["offdiag_cos"] > 0.99 and col["rel_variation"] < 0.05, col
    assert hea["offdiag_cos"] < 0.5, hea

    print("dinotxt_loss self-test OK")


if __name__ == "__main__":
    _self_test()
