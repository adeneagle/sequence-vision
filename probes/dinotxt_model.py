"""Two towers for the DINO.txt-style density <-> sequence alignment.

LiT ON BOTH SIDES. dino.txt freezes DINOv2 and trains a text encoder from
scratch; here BOTH unimodal models are strong and frozen (the CleanDIFT student,
ESM-C 600m), so what trains is a light head on each side. There is no reason to
train a protein language model when ESM-C exists, and the sample size (466
clusters after the map-level split) forbids it anyway.

V1 IS POINTWISE, DELIBERATELY. dino.txt adds two learnable TRANSFORMER blocks on
the frozen backbone; those mix across tokens, which would defeat caching
individually-sampled voxels and put a 3D UNet back in the training loop. So v1
uses per-voxel heads over cached features -- minutes per run, every control arm
affordable -- and convolutional blocks are v2, gated on v1 clearing G0. This
project's own CleanDIFT result says head capacity is worth revisiting
(`paperhead` beat `student` on SS, 0.7908 vs 0.7862, purely on head architecture)
so v2 is a real expectation, not a formality.

INIT. Every residual branch is zero-init, so at step 0 each tower is exactly its
linear projection: well-conditioned, and the nonlinear capacity has to earn its
way in. A 16-block residual decoder in the sibling project once diverged to
R2 -371 without this, and it is logged as a standing rule.

THE LAYERNORM ON THE SEQUENCE SIDE IS LOAD-BEARING, not decoration. Per-layer
ESM-C RMS spans 1.37..108 across depth, so a consumer that skips normalisation is
decided by scale. (v1 uses layer 32 alone -- the best single layer, R2 0.184 --
and the ESMFold2-style mix over all 36 is an ablation worth only +0.017.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """LayerNorm -> linear projection -> zero-init residual MLP -> L2 normalise."""

    def __init__(self, d_in: int, d_out: int = 256, hidden: int = 512,
                 depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            blk = nn.Sequential(
                nn.LayerNorm(d_out), nn.Linear(d_out, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, d_out),
            )
            nn.init.zeros_(blk[-1].weight)      # identity at init
            nn.init.zeros_(blk[-1].bias)
            self.blocks.append(blk)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-normalisation output, so a pair channel can be added before L2."""
        h = self.proj(self.norm(x))
        for blk in self.blocks:
            h = h + blk(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embed(x), dim=-1)


RELPOS_CLIP = 32
_LOG_MAX_SCALE = math.log(100.0)      # CLIP's temperature clamp


class PairSeqHead(nn.Module):
    """Sequence tower with an OPTIONAL ESMFold2-style pair channel.

    WHY THIS EXISTS, and a correction. The pair route was previously written off
    on Stage 0/1 evidence, but that evidence does not transfer: it was a
    FEATURE-RECONSTRUCTION R2 -- exactly the class of proxy metric this project's
    own P1 says never to gate on, having twice measured that feature similarity
    mispredicts downstream performance -- and it was measured against
    `up_blocks[0]` targets at **t=10**, both of which are now known to be poor
    settings (the timestep sweep moved that same target 0.145 -> 0.230 by changing
    t alone; the tap sweep puts `up_blocks[1]` 7.6 SS points above `up_blocks[0]`).
    So the question is reopened and must be settled on E1, the downstream task.

    The pair channel is also the one thing the per-residue path structurally
    cannot supply: which residues are SPATIALLY co-located. That is implicit
    structure, and it is what a voxel needs to be matched against.

    CHEAP BY CONSTRUCTION, which is the point given 505 independent clusters.
    `d_pair=32` keeps it ~100 K parameters -- an order of magnitude below a
    vision-side transformer block, so it is compatible with the sample-size
    constraint that ruled those out.

    ZERO-INIT EMIT: training starts EXACTLY at the per-residue baseline, so the
    pair channel can only earn its way in. `RowAttnPool.out_proj` and
    `PairUpdateBlock.w3` are already zero-init upstream; `emit` closes the path.

    LENGTH CAP: TriMul is O(L^3). Median chain here is 145 residues and p99 is
    1,116, so `max_len=1000` covers 98.5% of chains; longer ones fall back to the
    per-residue path rather than being dropped.

    MEMORY: GRADIENT CHECKPOINTING PER CHAIN IS NOT OPTIONAL. The pair rep is
    [n, n, d_pair] per chain and the loop below holds one per chain until
    backward, so activation memory is O(sum_c n_c^2), NOT O(n_res). Measured: all
    three pair arms died with **78 GB allocated on an 80 GB H100** at
    batch_maps=8 (~72 chains/batch, up to 1000 residues each -> ~128 MB per pair
    tensor before the several intermediates a TriMul block needs). Checkpointing
    each chain drops peak activation memory to a single chain's worth at ~1.3x
    compute. `checkpoint=False` reproduces the OOM and exists only for a
    memory/speed A-B.
    """

    def __init__(self, d_in: int = 1152, d_out: int = 256, hidden: int = 512,
                 depth: int = 1, dropout: float = 0.0, pair: str = "none",
                 n_tri: int = 2, d_s: int = 128, d0: int = 32, d_pair: int = 32,
                 max_len: int = 1000, checkpoint: bool = True):
        super().__init__()
        self.checkpoint = checkpoint
        from probes.stage1_pair_head import PairUpdateBlock, RowAttnPool, SingleToPair

        self.base = Head(d_in, d_out, hidden, depth, dropout)
        self.pair, self.max_len = pair, max_len
        self.use_esm = pair in ("esm", "esm+relpos")
        self.use_relpos = pair in ("relpos", "esm+relpos")
        if pair == "none":
            return
        if self.use_esm:
            self.single_proj = nn.Sequential(nn.Linear(d_in, d_s), nn.GELU())
            self.to_pair = SingleToPair(d_s, d0, d_pair)
        if self.use_relpos:
            self.relpos = nn.Linear(2 * RELPOS_CLIP + 1, d_pair, bias=False)
        self.blocks = nn.ModuleList([PairUpdateBlock(d_pair) for _ in range(n_tri)])
        self.pool = RowAttnPool(d_pair)
        self.emit = nn.Linear(d_pair, d_out)
        nn.init.zeros_(self.emit.weight)
        nn.init.zeros_(self.emit.bias)

    def _relpos_oh(self, L: int, device):
        i = torch.arange(L, device=device)
        b = (i[:, None] - i[None, :]).clamp(-RELPOS_CLIP, RELPOS_CLIP) + RELPOS_CLIP
        return F.one_hot(b, 2 * RELPOS_CLIP + 1).float()[None]

    def _chain(self, xi: torch.Tensor, n: int) -> torch.Tensor:
        """One chain's pair-channel contribution, [n, d_out]. Split out so it can
        be wrapped in a checkpoint (see the memory note in the class docstring)."""
        z = 0.0
        if self.use_esm:
            z = z + self.to_pair(self.single_proj(xi)[None])
        if self.use_relpos:
            z = z + self.relpos(self._relpos_oh(int(n), xi.device))
        for blk in self.blocks:
            z = blk(z)
        return self.emit(self.pool(z))[0]

    def forward(self, x: torch.Tensor, chain_sizes=None) -> torch.Tensor:
        """x [R, d_in] for a batch of concatenated CHAINS.

        `chain_sizes` gives the residue count of each chain, in order; the pair
        channel runs per chain because a pair rep across unrelated chains would
        assert relationships that do not exist (and is O(L^2) in the wrong L).
        """
        h = self.base.embed(x)
        if self.pair == "none" or chain_sizes is None:
            return F.normalize(h, dim=-1)
        out, off = [], 0
        for n in chain_sizes:
            n = int(n)
            xi = x[off:off + n]
            if n > self.max_len or n < 2:
                out.append(torch.zeros(n, h.shape[1], device=x.device, dtype=h.dtype))
                off += n
                continue
            if self.checkpoint and self.training and xi.requires_grad is not None:
                # use_reentrant=False so the no-grad-input case (frozen features)
                # still recomputes correctly; reentrant checkpointing silently
                # returns detached output when no input requires grad.
                oi = torch.utils.checkpoint.checkpoint(
                    self._chain, xi, n, use_reentrant=False)
            else:
                oi = self._chain(xi, n)
            out.append(oi)
            off += n
        assert off == len(x), f"chain_sizes sum to {off}, expected {len(x)}"
        return F.normalize(h + torch.cat(out), dim=-1)


class VoxMix(nn.Module):
    """Distance-biased self-attention over the voxels of ONE map (§8.18.3, T0-B).

    WHY. Our tap (`up_blocks[1]`) has a ~4-5 A half-decay, but the neighbourhood
    that defines a residue's structural environment is the ~10 A contact scale --
    which is why `up_blocks[0]` (~10 A) won the per-residue REGRESSION while
    `up_blocks[1]` won SS. Mixing lets a fine tap rebuild neighbourhood context
    instead of paying ~15 h to re-cache a coarser one.

    DISTANCE-BIASED, NOT POSITION-ENCODED. The attention logit gets a learned
    per-head function of the pair distance (an RBF expansion), and absolute
    coordinates never enter. So the module is exactly translation- and
    rotation-equivariant, which matters because the cache's rotations are an
    augmentation: an absolute-position encoding would make the four rotations of
    a map disagree and would not survive deployment on an arbitrarily-framed map.

    ZERO-INIT, like every other residual branch here: `out_proj` and the FFN's
    last layer start at zero, so at step 0 the tower is EXACTLY the pointwise
    baseline and the mixing has to earn its way in. Without this the comparison
    against v1 is confounded by a different starting point.

    ONE MAP AT A TIME. Voxels of different maps must not attend to each other --
    that would leak the batch composition into the prediction. The forward takes
    `vox_map` and loops; per-map n is ~384 (train) or ~800 (eval), so the [n, n]
    attention is trivial next to the rest of the step.

    THE CONTROL THIS ARM REQUIRES. Neighbouring voxels usually share a chain, so
    averaging features over a neighbourhood raises E1 by making predictions
    locally consistent, WITHOUT improving the underlying alignment. The matched
    control is post-hoc smoothing of the POINTWISE model's per-voxel scores over
    the same neighbourhoods (`--smooth-sigma` in `dinotxt_eval`): if mixing's
    gain is no larger, the attention learned nothing beyond smoothing. Cf. the
    coordinate-leakage artifact that put the sibling project's contact probe at
    AUC 0.997.
    """

    def __init__(self, d: int, heads: int = 4, n_rbf: int = 16,
                 r_max: float = 30.0, depth: int = 1, hidden: int = 512):
        super().__init__()
        self.h, self.depth = heads, depth
        assert d % heads == 0, f"d={d} not divisible by heads={heads}"
        self.register_buffer("centers", torch.linspace(0.0, r_max, n_rbf))
        self.gamma = float(n_rbf - 1) ** 2 / max(r_max, 1e-6) ** 2
        self.bias = nn.Linear(n_rbf, heads, bias=False)
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(depth)])
        self.out = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
            for _ in range(depth)])
        for i in range(depth):
            nn.init.zeros_(self.out[i].weight); nn.init.zeros_(self.out[i].bias)
            nn.init.zeros_(self.ffn[i][-1].weight); nn.init.zeros_(self.ffn[i][-1].bias)

    def _bias(self, xyz: torch.Tensor) -> torch.Tensor:
        """[heads, n, n] learned function of pair distance."""
        dm = torch.cdist(xyz[None], xyz[None])[0]
        r = torch.exp(-self.gamma * (dm[..., None] - self.centers) ** 2)
        return self.bias(r).permute(2, 0, 1)

    def _one(self, h: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        n, d = h.shape
        b = self._bias(xyz)[None]                        # [1, H, n, n]
        for i in range(self.depth):
            q, k, v = self.qkv[i](self.n1[i](h)).chunk(3, -1)
            q, k, v = (t.view(1, n, self.h, d // self.h).transpose(1, 2)
                       for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=b)
            h = h + self.out[i](a.transpose(1, 2).reshape(n, d))
            h = h + self.ffn[i](self.n2[i](h))
        return h

    def forward(self, h: torch.Tensor, xyz: torch.Tensor,
                vox_map: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(h)
        for m in torch.unique(vox_map):
            s = vox_map == m
            out[s] = self._one(h[s], xyz[s])
        return out


class DinoTxt(nn.Module):
    """Density voxel tower + sequence residue tower + a background embedding.

    The background embedding is a real learned vector, not an implicit "none of
    the above": a voxel in solvent has a positive target on it, and the loss can
    only express that if it has somewhere to point. Folding background into the
    negatives instead would train the model to push solvent away from every
    sequence without ever giving it a correct answer.
    """

    def __init__(self, d_vox: int = 256, d_seq: int = 1152, d: int = 256,
                 hidden: int = 512, depth: int = 1, dropout: float = 0.0,
                 tau_init: float = 0.07, pair: str = "none", n_tri: int = 2,
                 d_pair: int = 32, pair_max_len: int = 1000,
                 mix_depth: int = 0, mix_heads: int = 4, mix_rmax: float = 30.0,
                 blur_sigma: float = 0.0):
        super().__init__()
        # FEATURE BLUR: the content-INDEPENDENT special case of VoxMix, and the
        # control that decides whether learned attention is needed at all. A
        # fixed Gaussian distance kernel over the map's voxels, applied to the
        # RAW cached features before the head -- so the head still trains on top
        # of it, which makes this strictly stronger than post-hoc smoothing of
        # the scores. If it matches VoxMix, the gain is "a coarser receptive
        # field" and the right move would have been to tap up_blocks[0] (~10 A
        # half-decay) rather than build an attention module.
        self.blur_sigma = float(blur_sigma)
        self.vox = Head(d_vox, d, hidden, depth, dropout)
        self.mix = (VoxMix(d, heads=mix_heads, depth=mix_depth, hidden=hidden,
                           r_max=mix_rmax) if mix_depth > 0 else None)
        self.seq = PairSeqHead(d_seq, d, hidden, depth, dropout, pair=pair,
                               n_tri=n_tri, d_pair=d_pair, max_len=pair_max_len)
        self.bg = nn.Parameter(torch.randn(d) * 0.02)
        # Parameterised as log(1/tau) like CLIP, so the optimiser sees a scale
        # that is linear in the logit multiplier rather than 1/x.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / tau_init)))

    def scale(self) -> torch.Tensor:
        # Clamped to CLIP's range: an unclamped temperature runs away and the
        # softmax saturates, at which point the gradient dies silently.
        #
        # `max` is a PYTHON FLOAT, not a tensor. `torch.log(torch.tensor(100.))`
        # builds a CPU tensor, which raises on a CUDA parameter -- and a CPU-only
        # self-test cannot catch it, so this failed first on the cluster.
        return self.logit_scale.clamp(max=_LOG_MAX_SCALE).exp()

    def encode_voxels(self, x: torch.Tensor, xyz=None,
                      vox_map=None) -> torch.Tensor:
        if self.blur_sigma > 0:
            if xyz is None:
                raise ValueError("blur_sigma>0 needs coordinates")
            if vox_map is None:
                vox_map = torch.zeros(len(x), dtype=torch.long, device=x.device)
            xb = torch.empty_like(x)
            for m in torch.unique(vox_map):
                sl = vox_map == m
                p = xyz[sl]
                w = torch.exp(-0.5 * (torch.cdist(p[None], p[None])[0]
                                      / self.blur_sigma) ** 2)
                # Row-normalised, so the result is a weighted MEAN and does not
                # depend on how densely the map happened to be sampled.
                xb[sl] = (w @ x[sl]) / w.sum(1, keepdim=True)
            x = xb
        h = self.vox.embed(x)
        if self.mix is not None:
            if xyz is None:
                raise ValueError(
                    "mix_depth>0 but no coordinates were passed -- the mixing "
                    "arm silently degrading to pointwise would look like a null "
                    "result. Pass xyz (probes/recover_voxel_coords.py).")
            if vox_map is None:
                vox_map = torch.zeros(len(x), dtype=torch.long, device=x.device)
            h = self.mix(h, xyz, vox_map)
        return F.normalize(h, dim=-1)

    def encode_residues(self, x: torch.Tensor, chain_sizes=None) -> torch.Tensor:
        return self.seq(x, chain_sizes)

    def bg_embedding(self) -> torch.Tensor:
        return F.normalize(self.bg, dim=-1)


def _self_test() -> None:
    torch.manual_seed(0)
    m = DinoTxt(d_vox=256, d_seq=1152, d=64, hidden=128, depth=2)

    # zero-init residual branches => at init the head IS its linear projection
    x = torch.randn(7, 256)
    h = m.vox(x)
    lin = F.normalize(m.vox.proj(m.vox.norm(x)), dim=-1)
    assert torch.allclose(h, lin, atol=1e-6), "residual branches are not zero-init"

    # outputs are unit norm
    for t, dim in ((m.encode_voxels(torch.randn(5, 256)), 64),
                   (m.encode_residues(torch.randn(9, 1152)), 64)):
        assert t.shape[-1] == dim
        assert torch.allclose(t.norm(dim=-1), torch.ones(len(t)), atol=1e-5), \
            "embeddings are not L2-normalised"
    assert torch.allclose(m.bg_embedding().norm(), torch.tensor(1.0), atol=1e-5)

    # temperature clamp actually binds
    with torch.no_grad():
        m.logit_scale.fill_(50.0)
    assert float(m.scale()) <= 100.0 + 1e-3, "logit scale is not clamped"

    # gradients reach every trainable piece, including the background vector
    with torch.no_grad():
        m.logit_scale.fill_(2.6)
    loss = (m.encode_voxels(torch.randn(4, 256)) @ m.bg_embedding()).sum() \
        + m.encode_residues(torch.randn(4, 1152)).sum() * m.scale()
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no grad: {n}"
    assert m.bg.grad.abs().sum() > 0, "background embedding gets no gradient"

    # ---- pair channel -------------------------------------------------
    R, sizes = 40, [12, 18, 10]
    x = torch.randn(R, 1152)
    for mode in ("esm", "relpos", "esm+relpos"):
        mp = DinoTxt(d_vox=256, d_seq=1152, d=64, hidden=128, depth=1,
                     pair=mode, n_tri=2, d_pair=16)
        # ZERO-INIT: at step 0 the pair channel contributes exactly nothing, so
        # the tower is identical to the per-residue baseline. If this fails the
        # pair arm does not start at the baseline and no comparison is clean.
        got = mp.encode_residues(x, chain_sizes=sizes)
        want = F.normalize(mp.seq.base.embed(x), dim=-1)
        assert torch.allclose(got, want, atol=1e-6), \
            f"{mode}: pair channel is not zero at init"
        # ...but it must be ABLE to move once emit is non-zero.
        with torch.no_grad():
            mp.seq.emit.weight.normal_(0, 0.1)
            mp.seq.pool.out_proj.weight.normal_(0, 0.1)
        moved = mp.encode_residues(x, chain_sizes=sizes)
        assert not torch.allclose(moved, want, atol=1e-4), \
            f"{mode}: pair channel cannot influence the output"

        # gradients reach the pair modules
        moved.sum().backward()
        pair_par = [n for n, q in mp.named_parameters()
                    if n.startswith("seq.") and "base" not in n]
        assert pair_par, f"{mode}: no pair parameters"
        for n_ in pair_par:
            g = dict(mp.named_parameters())[n_].grad
            assert g is not None and torch.isfinite(g).all(), f"{mode}: no grad {n_}"
        # chain_sizes must be validated, not trusted
        try:
            mp.encode_residues(x, chain_sizes=[5, 5])
            raise SystemExit(f"{mode}: bad chain_sizes silently accepted")
        except AssertionError:
            pass
        # length cap falls back to the per-residue path rather than exploding
        mp2 = DinoTxt(d_seq=1152, d=64, hidden=128, depth=1, pair=mode,
                      n_tri=1, d_pair=16, pair_max_len=8)
        capped = mp2.encode_residues(x, chain_sizes=sizes)
        assert torch.allclose(capped, F.normalize(mp2.seq.base.embed(x), dim=-1),
                              atol=1e-6), f"{mode}: length cap did not fall back"

    # ---- device parity: the clamp bug above was CPU-invisible ----------
    if torch.cuda.is_available():
        mc = DinoTxt(d_seq=1152, d=64, hidden=128, depth=1, pair="esm",
                     n_tri=1, d_pair=16).cuda()
        _ = mc.scale()
        _ = mc.encode_residues(torch.randn(20, 1152).cuda(), chain_sizes=[20])
        _ = mc.encode_voxels(torch.randn(5, 256).cuda())
        print("  CUDA parity OK")
    else:
        print("  (CUDA unavailable: device parity NOT exercised)")

    # ---- voxel token mixing -------------------------------------------
    mm = DinoTxt(d_vox=256, d_seq=1152, d=64, hidden=128, depth=1, mix_depth=2)
    xv = torch.randn(30, 256)
    xyz = torch.randn(30, 3) * 12.0
    vm = torch.tensor([0] * 17 + [1] * 13)
    got = mm.encode_voxels(xv, xyz, vm)
    want = F.normalize(mm.vox.embed(xv), dim=-1)
    assert torch.allclose(got, want, atol=1e-6), "mixing is not zero at init"
    with torch.no_grad():
        for i in range(mm.mix.depth):
            mm.mix.out[i].weight.normal_(0, 0.1)
    got2 = mm.encode_voxels(xv, xyz, vm)
    assert not torch.allclose(got2, want, atol=1e-4), "mixing cannot influence output"

    # MAPS MUST NOT SEE EACH OTHER: perturbing map 1's voxels must leave map 0's
    # outputs bit-identical. A batch-composition leak here would inflate E1.
    xv2 = xv.clone(); xv2[17:] = torch.randn(13, 256)
    got3 = mm.encode_voxels(xv2, xyz, vm)
    assert torch.allclose(got2[:17], got3[:17], atol=1e-6), \
        "voxels attend across maps -- batch composition leaks into predictions"

    # DISTANCE BIAS MUST BE TRANSLATION+ROTATION EQUIVARIANT (invariant output).
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    got4 = mm.encode_voxels(xv, xyz @ R.t() + 5.0, vm)
    assert torch.allclose(got2, got4, atol=1e-4), \
        "mixing is not invariant to a rigid motion of the coordinates"

    print("dinotxt_model self-test OK (incl. pair channel + voxel mixing)")


if __name__ == "__main__":
    _self_test()
