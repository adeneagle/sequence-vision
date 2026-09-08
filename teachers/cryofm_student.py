"""CleanDIFT student pieces for CryoFM2: learned conditioning + projection heads.

CleanDIFT (Stracke et al., CVPR 2025) distils a diffusion model that needs NOISY
input into a student that reads CLEAN input, by training a copy of the backbone to
match the frozen teacher's activations across the whole noise schedule. The student
still has to pass SOME timestep, because the UNet signature demands one; `t` enters
the loss only through FiLM-conditioned projection heads, which are DISCARDED at
inference. So downstream there is one feature set and no `t` to choose.

Two deliberate departures from the reference implementation, both explained where
they are implemented: the conditioning is a free 256-d vector rather than a learned
scalar (`LearnedTimeEmb`), and the heads are identity-at-init with a bottleneck
(`FiLMHead`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LearnedTimeEmb(nn.Module):
    """A constant, learnable replacement for `UNet3DModel.time_embedding`.

    WHY A FREE VECTOR, NOT A LEARNED SCALAR `t`. `get_timestep_embedding` IS
    differentiable w.r.t. `t`, but with `max_period=10000` over 32 half-channels
    the highest-frequency channel completes a cycle every ~6.3 timesteps, so
    d(emb)/dt oscillates violently and the scalar is badly conditioned as a
    trainable parameter. A free 256-d vector is better conditioned and strictly
    more expressive -- at the cost that the learned conditioning may leave the
    1-D manifold of valid timestep embeddings entirely, which is why
    `nearest_timestep` exists to report how far off it lands.

    The input is ignored, so the student's `timestep` argument becomes a no-op and
    none of the `dtype=torch.long` call sites need changing.
    """

    def __init__(self, init_vec: torch.Tensor):
        super().__init__()
        if init_vec.ndim != 1:
            raise ValueError(f"expected a 1-D init vector, got {tuple(init_vec.shape)}")
        self.p = nn.Parameter(init_vec.detach().clone().float())

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        # .repeat, not .expand: expand returns ONE shared buffer for every batch
        # element. The current resnets never write to `temb` in place so expand
        # would be safe today, but 256 floats is not worth the footgun.
        return self.p.unsqueeze(0).repeat(t_emb.shape[0], 1).to(t_emb.dtype)


@torch.no_grad()
def embedding_for_t(model, t: int, device=None) -> torch.Tensor:
    """The teacher's own conditioning vector for timestep `t`, as [256].

    This is `time_embedding(time_proj(t))` -- exactly what `UNet3DModel.forward`
    computes at `unet.py:311` -- so initialising `LearnedTimeEmb` with it makes
    the student bit-identical to the naive decoupled teacher at `t`.
    """
    device = device or next(model.parameters()).device
    tt = torch.full((1,), int(t), device=device, dtype=torch.long)
    t_emb = model.time_proj(tt).to(dtype=model.dtype)
    return model.time_embedding(t_emb)[0].float()


def attach_const_time_embedding(model, init_vec: torch.Tensor) -> LearnedTimeEmb:
    """Swap the student's `time_embedding` for a learned constant.

    A one-line attribute swap is enough because `emb` has exactly one writer in
    `UNet3DModel.forward` (`emb = self.time_embedding(t_emb)`, `unet.py:311`) and
    is then threaded unchanged into every FiLM site as `temb`; the only other
    writer is the `class_embedding` branch, which is dead for this config
    (`class_embedding is None`).
    """
    mod = LearnedTimeEmb(init_vec).to(next(model.parameters()).device)
    model.time_embedding = mod
    return mod


@torch.no_grad()
def nearest_timestep(teacher, p: torch.Tensor, t_max: int = 1000) -> dict:
    """Closest real timestep to a learned conditioning vector, and how far off.

    Leaving the 1-D manifold of valid timestep embeddings is the honest cost of
    using a free vector: once it happens, "the student learned t=683" stops being
    a sentence one can say. A large residual is a FINDING to report, not a
    failure -- but it must be reported, not assumed away.
    """
    p = p.detach().float()
    embs = torch.stack([embedding_for_t(teacher, t, p.device) for t in range(t_max + 1)])
    d = torch.linalg.norm(embs - p[None], dim=1)
    i = int(torch.argmin(d))
    cos = torch.nn.functional.cosine_similarity(embs[i][None], p[None]).item()
    return {"nearest_t": i,
            "rel_residual": float(d[i] / (p.norm() + 1e-12)),
            "cos": float(cos)}


class SwiGLUFFN(nn.Module):
    """One zero-initialised, FiLM-conditioned, SwiGLU FFN block. Identity at init.

    Matches the reference architecture, verified against the paper (arXiv
    2412.03439v2): the projection head is "three stacked Feed Forward Networks
    (FFNs) that are zero-initialized such that initially they act as identity
    mappings due to their residual connections", with "a FiLM layer in each FFN
    block to adaptively scale activations depending on the timestep t" and "the
    SwiGLU gating mechanism as an activation function in each FFN block" -- 45M
    additional trainable parameters for SD 2.1.
    """

    def __init__(self, channels: int, t_dim: int = 256, expand: int = 4):
        super().__init__()
        hid = channels * expand
        self.norm = nn.GroupNorm(1, channels)
        self.film = nn.Linear(t_dim, 2 * channels)
        self.gate = nn.Conv3d(channels, 2 * hid, 1)   # SwiGLU: value and gate
        self.out = nn.Conv3d(hid, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.hidden = hid

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.film(t_emb).chunk(2, dim=1)
        v = (x.shape[0], -1) + (1,) * (x.ndim - 2)
        h = h * (1.0 + scale.reshape(*v)) + shift.reshape(*v)
        a, b = self.gate(h).chunk(2, dim=1)
        return x + self.out(a * torch.nn.functional.silu(b))


class PaperHead(nn.Module):
    """`n_blocks` stacked SwiGLU FFN blocks -- the reference head shape.

    WHY THIS EXISTS ALONGSIDE `FiLMHead`. `FiLMHead` is one block bottlenecked to
    C/4 (~0.2M params), chosen to stop the head absorbing the teacher/student
    discrepancy and leaving the SHIPPED trunk unmoved. The paper does the opposite
    on purpose: it gives the head *ample* capacity precisely so it can absorb the
    t-specific part, leaving the trunk free to be t-agnostic. Both arguments are
    coherent and they prescribe opposite widths, so it is an empirical question --
    and the measured `head_share` of 0.998-0.999 in the lr probe says absorption is
    NOT a problem here, i.e. the bottleneck guards a risk that has not
    materialised while possibly limiting the fit. Hence this arm.
    """

    def __init__(self, channels: int, t_dim: int = 256, n_blocks: int = 3,
                 expand: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SwiGLUFFN(channels, t_dim, expand) for _ in range(n_blocks)])
        self.hidden = self.blocks[0].hidden

    def forward(self, f: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            f = b(f, t_emb)
        return f


class FiLMHead(nn.Module):
    """`proj(f; t) = f + conv2(silu(FiLM(conv1(f); t)))`, identity at init.

    THREE THINGS THIS GETS RIGHT, each of which quietly breaks the run otherwise:

    * **Identity at init, not zero at init.** A purely zero-initialised head
      outputs the zero vector, and a *centred cosine* against zero is undefined
      (NaN, or a dead gradient). Zero-initialising only `conv2` makes the head the
      identity map at step 0 while still starting the residual branch from zero.
    * **Conditioned on the TEACHER's sampled `t`**, never on the student's learned
      vector. That asymmetry is the entire mechanism: one clean student
      representation has to explain the teacher's features at *every* `t`, and the
      head absorbs the `t`-specific part.
    * **Bottlenecked** (`hidden = C // ratio`). At full width the head is a
      per-token, full-rank, `t`-conditioned MLP, and any invertible linear channel
      remap it applies is INVISIBLE to the linear probe downstream. Since only the
      trunk ships, a full-width head can absorb the whole teacher/student
      discrepancy and leave the shipped student exactly at its initialisation
      while the training curve looks healthy. Narrowing it makes that outcome
      architecturally harder rather than merely detectable.
    """

    def __init__(self, channels: int, t_dim: int = 256, ratio: int = 4):
        super().__init__()
        hid = max(8, channels // ratio)
        self.conv1 = nn.Conv3d(channels, hid, 1)
        self.conv2 = nn.Conv3d(hid, channels, 1)
        self.film = nn.Linear(t_dim, 2 * hid)
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        # FiLM starts as the identity modulation (scale 0 -> *(1+0), shift 0).
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.hidden = hid

    def forward(self, f: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(f)
        scale, shift = self.film(t_emb).chunk(2, dim=1)
        # .reshape, not .view: `chunk` returns non-contiguous slices, and .view
        # raises on them ("at least one dimension spans across two contiguous
        # subspaces"). Broadcast over however many spatial dims f has.
        v = (f.shape[0], -1) + (1,) * (f.ndim - 2)
        h = h * (1.0 + scale.reshape(*v)) + shift.reshape(*v)
        return f + self.conv2(self.act(h))


def freeze_trunk(model) -> None:
    """`student_t_only` (D12): only the conditioning vector and the heads train.

    If that arm recovers most of the gain, all the distillation bought was a
    better conditioning vector -- which a plain `t` sweep gets for free.
    """
    for n, prm in model.named_parameters():
        prm.requires_grad_(n.startswith("time_embedding."))


# ---------------------------------------------------------------------------
# Self-test: the D3 identity and the gradient plumbing.
# ---------------------------------------------------------------------------

def _self_test(ckpt: str, t_init: int = 750, device: str = "cuda") -> None:
    from teachers.cryofm_tap import DEFAULT_TAPS, CryoFM2Tap

    taps = DEFAULT_TAPS
    teacher = CryoFM2Tap(ckpt, taps=taps, device=device, batch_size=2)
    student = CryoFM2Tap(ckpt, taps=taps, device=device, batch_size=2,
                         detach=False, trainable=True)
    vec = embedding_for_t(teacher.model, t_init, device)
    attach_const_time_embedding(student.model, vec)

    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 1, 64, 64, 64, generator=g)
    x = torch.cat([x, torch.zeros_like(x)], 1).to(device)
    tt = torch.full((2,), t_init, device=device, dtype=torch.long)
    t0 = torch.zeros(2, device=device, dtype=torch.long)

    # D3: student at init == teacher decoupled at t_init.
    #
    # TF32 MUST BE OFF FOR THIS CHECK. cuDNN's TF32 conv path carries ~1e-3
    # relative error and its algorithm selection depends on the autograd context,
    # so the grad-building student forward and the no_grad teacher forward pick
    # different kernels and disagree at ~5e-4 -- which is what a first run of this
    # test reported, and it is measurement noise, not a broken identity. Both
    # arms are also run under no_grad here so the contexts match; the gradient
    # plumbing is checked separately below on its own forward.
    def _identity_err(strict: bool) -> tuple[float, dict]:
        prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        if strict:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.no_grad():
                a = {k: v.float().clone() for k, v in teacher.forward_taps(x, tt).items()}
                b = {k: v.float().clone() for k, v in student.forward_taps(x, t0).items()}
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev
        errs = {k: ((b[k] - a[k]).norm() / a[k].norm()).item() for k in taps}
        return max(errs.values()), errs

    loose, _ = _identity_err(strict=False)
    worst, errs = _identity_err(strict=True)
    print(f"D3 identity at t_init={t_init}:")
    for k in taps:
        print(f"  {k:15s} rel_err {errs[k]:.3e}")
    print(f"  worst {worst:.3e} with TF32 OFF   ({loose:.3e} with TF32 as configured "
          f"-- the gap is TF32/kernel-selection noise, not the identity)")
    assert worst < 1e-5, f"D3 identity FAILED: rel_err {worst:.3e} with TF32 off"

    # a grad-carrying forward for the plumbing checks below
    with torch.no_grad():
        tf = {k: v.float().clone() for k, v in teacher.forward_taps(x, tt).items()}
    sf = student.forward_taps(x, t0)

    # heads must be the identity at init
    head = FiLMHead(sf[taps[-1]].shape[1]).to(device)
    h = head(sf[taps[-1]].float(), vec[None].repeat(2, 1))
    hid_err = ((h - sf[taps[-1]].float()).abs().max()).item()
    print(f"head identity-at-init: max|out-in| {hid_err:.3e} (hidden={head.hidden})")
    assert hid_err == 0.0, "FiLMHead is not the identity at init"

    # grad plumbing: a loss on the taps must reach the trunk AND the D2 parameter
    assert sf[taps[-1]].requires_grad, "student taps do not require grad"
    loss = sum(v.float().pow(2).mean() for v in sf.values())
    loss.backward()
    pg = student.model.time_embedding.p.grad
    mid = student.model.mid_block.resnets[0].conv1.weight.grad
    print(f"grad: time_embedding.p {None if pg is None else float(pg.norm()):.4g}  "
          f"mid_block conv1 {None if mid is None else float(mid.norm()):.4g}")
    assert pg is not None and pg.norm() > 0, "no grad on the learned conditioning"
    assert mid is not None and mid.norm() > 0, "no grad in the trunk"

    # stop_after must not change the taps it does capture
    trunc = CryoFM2Tap(ckpt, taps=taps, device=device, batch_size=2,
                       stop_after="up_blocks[1]")
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            ref = {k: v.float().clone() for k, v in teacher.forward_taps(x, tt).items()}
            cut = {k: v.float().clone() for k, v in trunc.forward_taps(x, tt).items()}
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev
    err = max(((cut[k] - ref[k]).abs().max()).item() for k in taps)
    scale = max(float(ref[k].abs().max()) for k in taps)
    print(f"D14 truncation equality: max|full - truncated| {err:.3e} "
          f"(feature scale {scale:.3g})")
    # Truncating cannot change what already ran, so this should be exact; the
    # tolerance only covers cuDNN picking a different algorithm for an identical
    # op, which is benign. Anything larger means stop_after changed the compute.
    assert err <= 1e-6 * max(scale, 1.0), "stop_after changed the captured features"

    print(f"nearest_timestep(init) -> {nearest_timestep(teacher.model, vec)}")
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--t-init", type=int, default=750)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    _self_test(a.ckpt, a.t_init, a.device)
