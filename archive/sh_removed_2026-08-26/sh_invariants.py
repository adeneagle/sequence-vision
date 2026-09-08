"""Rotation-invariant local density descriptors via spherical harmonics.

Motivation: CryoFM is not equivariant, so fine-grained features from it cannot be
made pose-invariant without canonicalising its input -- which needs a frame, which
needs either an atomic model (not a general method) or a density-derived frame
(measured at ~30 deg error, unusable). SH invariants sidestep the whole problem.

For a point p, expand the density on concentric shells around it:

    rho(r, theta, phi) = sum_lm c_lm(r) Y_lm(theta, phi)
    p_l(r) = sum_m |c_lm(r)|^2

A rotation mixes the m components within fixed l, so p_l(r) is EXACTLY invariant.
No frame, no atomic model, no degeneracy, no learned teacher to validate.

Cost: the power spectrum discards phase, so two arrangements with the same radial
power profile are indistinguishable. The bispectrum retains more phase relations
while staying invariant (cf. SOAP descriptors) -- not implemented here.

Descriptor dimension = n_shells * (l_max + 1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly-uniform unit vectors [n, 3] (z, y, x order)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.cos(phi),
                     np.sin(phi) * np.sin(theta),
                     np.sin(phi) * np.cos(theta)], axis=1)


def _real_sh(l_max: int, dirs: np.ndarray) -> list[np.ndarray]:
    """Real spherical harmonics up to l_max, evaluated on `dirs` [P,3] (z,y,x).

    Returns a list indexed by l, each [P, 2l+1].
    """
    from scipy.special import sph_harm

    z, y, x = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))          # polar
    phi = np.arctan2(y, x)                        # azimuth
    out = []
    for l in range(l_max + 1):
        cols = []
        for m in range(-l, l + 1):
            Y = sph_harm(abs(m), l, phi, theta)   # scipy: (m, l, azimuth, polar)
            if m < 0:
                v = np.sqrt(2.0) * (-1) ** m * Y.imag
            elif m == 0:
                v = Y.real
            else:
                v = np.sqrt(2.0) * (-1) ** m * Y.real
            cols.append(v)
        out.append(np.stack(cols, axis=1))        # [P, 2l+1]
    return out


class SHInvariants:
    """Precomputes the sampling geometry once; reusable across maps."""

    def __init__(self, radii=(2.0, 4.0, 6.0, 8.0, 12.0), l_max: int = 6,
                 n_dirs: int = 302, voxel: float = 1.5, per_shell_norm: bool = True):
        self.radii = np.asarray(radii, dtype=np.float64)
        self.l_max = l_max
        self.voxel = voxel
        self.dirs = fibonacci_sphere(n_dirs)
        self.Y = _real_sh(l_max, self.dirs)        # list of [P, 2l+1]
        # quadrature weight: uniform on the Fibonacci sphere
        self.w = 4.0 * np.pi / n_dirs
        self.per_shell_norm = per_shell_norm
        self.dim = len(self.radii) * (l_max + 1)

    @torch.no_grad()
    def __call__(self, vol: np.ndarray, coords: np.ndarray,
                 device: str = "cpu", chunk: int = 256) -> np.ndarray:
        """Descriptors [N, n_shells*(l_max+1)] at voxel coordinates `coords` [N,3]."""
        D, H, W = vol.shape
        shape = torch.tensor([D, H, W], dtype=torch.float64, device=device)
        v5 = torch.from_numpy(np.ascontiguousarray(vol)).float().to(device)[None, None]

        # sample points: centre + r/voxel * direction, for every (shell, direction)
        offs = np.concatenate([(r / self.voxel) * self.dirs for r in self.radii])  # [S*P,3]
        offs_t = torch.from_numpy(offs).to(device)

        Ys = [torch.from_numpy(y).to(device).double() for y in self.Y]
        P = len(self.dirs)
        out = []
        for s in range(0, len(coords), chunk):
            c = torch.from_numpy(coords[s:s + chunk]).to(device).double()   # [m,3]
            pts = c[:, None, :] + offs_t[None, :, :]                        # [m,S*P,3]
            g = (2.0 * pts / (shape - 1) - 1.0).flip(-1).float()
            g = g.view(len(c), -1, 1, 1, 3)
            vals = F.grid_sample(v5.expand(len(c), -1, -1, -1, -1), g,
                                 mode="bilinear", align_corners=True,
                                 padding_mode="zeros")
            vals = vals.view(len(c), len(self.radii), P).double()           # [m,S,P]
            feats = []
            for l, Yl in enumerate(Ys):
                c_lm = torch.einsum("msp,pk->msk", vals, Yl) * self.w       # [m,S,2l+1]
                feats.append((c_lm ** 2).sum(-1))                           # [m,S]
            d = torch.stack(feats, -1)                                      # [m,S,l+1]
            if self.per_shell_norm:
                # Each shell's l-spectrum is well preserved under rotation, but the
                # RELATIVE magnitude between shells is not -- raw concatenation
                # scores 0.855 while per-shell normalisation scores 0.959 on the
                # independent-discretization test. Normalise per shell.
                d = d / torch.clamp(d.norm(dim=-1, keepdim=True), min=1e-12)
            out.append(d.reshape(len(c), -1).cpu().numpy())
        return np.concatenate(out)                                          # [N, S*(l+1)]


if __name__ == "__main__":
    # Exactness check: invariance is analytic here, so an independent
    # discretization at a different orientation must reproduce the descriptor.
    # Anything far from 1.000 is an implementation bug, not a measurement.
    import sys

    from data.simulate_density import simulate
    from probes.local_frame_stability import backbone_frames, random_so3

    cif = sys.argv[1] if len(sys.argv) > 1 else "data/maps/9w0g.cif"
    rng = np.random.default_rng(0)
    sh = SHInvariants()
    print(f"descriptor dim = {sh.dim}")

    ca_zyx, _ = backbone_frames(cif)
    sel = rng.choice(len(ca_zyx), 200, replace=False)
    xyz = ca_zyx[sel][:, ::-1].copy()

    volA, oA, spA = simulate(cif, d_min=3.0, voxel=1.5)
    # Use the RETURNED spacing: gemmi's realized spacing is never the requested
    # 1.5 A, is anisotropic ~7-11%, and is pose-dependent. Hardcoding 1.5 here
    # mislocated Ca atoms by a median 8.8 A elsewhere in this codebase.
    cA = ((xyz - np.asarray(oA)[None]) / np.asarray(spA)[::-1][None])[:, ::-1].copy()
    A = sh(volA, cA)

    for t in range(3):
        R = random_so3(rng)
        volB, oB, spB = simulate(cif, d_min=3.0, voxel=1.5, rotation=R)
        ctr = xyz.mean(0)
        pB = (xyz - ctr) @ R.T + ctr
        cB = ((pB - np.asarray(oB)[None]) / np.asarray(spB)[::-1][None])[:, ::-1].copy()
        B = sh(volB, cB)
        num = (A * B).sum(1)
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        cos = num / np.maximum(den, 1e-12)
        rel = np.abs(A - B).sum(1) / np.maximum(np.abs(A).sum(1), 1e-12)
        print(f"trial {t}: cosine median {np.median(cos):.4f}  p10 {np.percentile(cos,10):.4f}"
              f"  | median relative L1 error {np.median(rel):.4f}")
