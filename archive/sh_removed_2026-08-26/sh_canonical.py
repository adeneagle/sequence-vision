"""Canonical local orientation from spherical harmonics -- model-free.

This is SH used for ALIGNMENT, not as an invariant descriptor. The power spectrum
(teachers/sh_invariants.py) is rotation-invariant but discards phase, and cannot be
combined with a non-equivariant feature extractor. Here we instead recover a
canonical rotation from the density, put the local box in standard position, and
then run CryoFM on it -- preserving phase and keeping the foundation model in play.
Classical precedent: fast rotational matching in cryo-EM does the same thing.

Why this should beat the structure tensor (measured at ~30 deg axis error): that
is essentially a second-order (l=2) object, so it inherits the eigenvector
degeneracy that also sinks inertial frames. Here:

  axis 1  <- the l=1 DIPOLE of the local density. A vector, not a tensor: no sign
             ambiguity, no eigenvalue ordering, degenerate only if it vanishes.
  axis 2  <- the residual rotation about axis 1, fixed by maximising a real l=2 or
             l=3 coefficient. A 1-D search, so no tie-breaking pathology.

A frame is therefore ill-defined only under genuine local spherical symmetry.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from teachers.sh_invariants import fibonacci_sphere


class SHCanonicalizer:
    """Deterministic local frames from density, via dipole + azimuthal search."""

    def __init__(self, radii=(3.0, 5.0, 8.0), n_dirs: int = 302, voxel: float = 1.5,
                 n_azimuth: int = 180, degree: int = 3):
        self.radii = np.asarray(radii, dtype=np.float64)
        self.voxel = voxel
        self.dirs = fibonacci_sphere(n_dirs)          # [P,3] (z,y,x)
        self.n_azimuth = n_azimuth
        self.degree = degree

    @torch.no_grad()
    def _shell_samples(self, vol, coords, device):
        """Density on each shell: [N, S, P]."""
        D, H, W = vol.shape
        shape = torch.tensor([D, H, W], dtype=torch.float64, device=device)
        v5 = torch.from_numpy(np.ascontiguousarray(vol)).float().to(device)[None, None]
        offs = np.concatenate([(r / self.voxel) * self.dirs for r in self.radii])
        offs_t = torch.from_numpy(offs).to(device)
        c = torch.from_numpy(coords).to(device).double()
        out = []
        for s in range(0, len(c), 128):
            cc = c[s:s + 128]
            pts = cc[:, None, :] + offs_t[None, :, :]
            g = (2.0 * pts / (shape - 1) - 1.0).flip(-1).float().view(len(cc), -1, 1, 1, 3)
            v = F.grid_sample(v5.expand(len(cc), -1, -1, -1, -1), g, mode="bilinear",
                              align_corners=True, padding_mode="zeros")
            out.append(v.view(len(cc), len(self.radii), len(self.dirs)).double().cpu())
        return torch.cat(out).numpy()

    def __call__(self, vol: np.ndarray, coords: np.ndarray,
                 device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """Frames [N,3,3] (rows = axes) and a per-residue confidence [N].

        Confidence is the normalised dipole magnitude; near zero means a locally
        isotropic neighbourhood where axis 1 is ill-defined.
        """
        vals = self._shell_samples(vol, coords, device)      # [N,S,P]
        d = self.dirs                                        # [P,3]

        # ---- axis 1: dipole (l=1), summed over shells with equal weight
        dip = np.einsum("nsp,pk->nk", vals, d)               # [N,3]
        mag = np.linalg.norm(dip, axis=1)
        scale = np.abs(vals).sum(axis=(1, 2)) + 1e-12
        conf = mag / scale
        e1 = dip / np.maximum(mag, 1e-12)[:, None]

        # ---- axis 2: rotate about e1 to maximise a real degree-`degree` coefficient.
        # Build an orthonormal pair (u, w) perpendicular to e1, then search phi.
        tmp = np.tile(np.array([1.0, 0.0, 0.0]), (len(e1), 1))
        alt = np.tile(np.array([0.0, 1.0, 0.0]), (len(e1), 1))
        use_alt = np.abs(np.einsum("nk,nk->n", e1, tmp)) > 0.9
        tmp[use_alt] = alt[use_alt]
        u = tmp - e1 * np.einsum("nk,nk->n", tmp, e1)[:, None]
        u /= np.maximum(np.linalg.norm(u, axis=1), 1e-12)[:, None]
        w = np.cross(e1, u)

        # direction components in the (e1, u, w) basis
        c_par = d @ e1.T                                     # [P,N]
        c_u = d @ u.T
        c_w = d @ w.T
        azim = np.arctan2(c_w, c_u)                          # [P,N]
        # weight by a band around the equator so the azimuth is well determined
        band = np.clip(1.0 - c_par**2, 0.0, 1.0)
        m = self.degree
        wsum = vals.sum(axis=1).T                            # [P,N]
        cos_t = np.einsum("pn,pn,pn->n", wsum, band, np.cos(m * azim))
        sin_t = np.einsum("pn,pn,pn->n", wsum, band, np.sin(m * azim))
        phi = np.arctan2(sin_t, cos_t) / m                   # canonical azimuth

        e2 = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * w
        e2 -= e1 * np.einsum("nk,nk->n", e2, e1)[:, None]
        e2 /= np.maximum(np.linalg.norm(e2, axis=1), 1e-12)[:, None]
        e3 = np.cross(e1, e2)
        return np.stack([e1, e2, e3], axis=1), conf


if __name__ == "__main__":
    # Equivariance test on INDEPENDENT discretizations: frame_B should equal R.frame_A.
    # Structure-tensor frames scored ~30 deg here; a working construction gives ~0.
    import sys

    from data.simulate_density import simulate
    from probes.local_frame_stability import backbone_frames, random_so3
    from probes.pose_invariance_clean import _R_zyx

    cif = sys.argv[1] if len(sys.argv) > 1 else "data/maps/9w0g.cif"
    rng = np.random.default_rng(0)
    ca, _ = backbone_frames(cif)
    sel = rng.choice(len(ca), 200, replace=False)
    xyz = ca[sel][:, ::-1].copy()

    can = SHCanonicalizer()
    volA, oA, spA = simulate(cif, d_min=3.0, voxel=1.5, margin=20.0)
    # Realized spacing, not the requested 1.5: see data/simulate_density.py.
    cA = ((xyz - np.asarray(oA)[None]) / np.asarray(spA)[::-1][None])[:, ::-1].copy()
    FA, confA = can(volA, cA)
    print(f"dipole confidence: median {np.median(confA):.4f}  "
          f"frac<0.01 {np.mean(confA < 0.01):.0%}")

    for t in range(3):
        R = random_so3(rng)
        volB, oB, spB = simulate(cif, d_min=3.0, voxel=1.5, rotation=R, margin=20.0)
        ctr = xyz.mean(0)
        pB = (xyz - ctr) @ R.T + ctr
        cB = ((pB - np.asarray(oB)[None]) / np.asarray(spB)[::-1][None])[:, ::-1].copy()
        FB, _ = can(volB, cB)
        Rz = _R_zyx(R)
        pred = np.einsum("ij,mkj->mki", Rz, FA)
        # axis 1 is a true vector -> signed comparison; axes 2/3 too, since the
        # azimuthal search fixes them (no sign ambiguity by construction).
        ang = np.degrees(np.arccos(np.clip(np.sum(pred * FB, axis=2), -1, 1)))
        good = confA > np.median(confA)
        print(f"trial {t}: axis1 median {np.median(ang[:,0]):5.1f} deg | "
              f"all-axis median {np.median(ang):5.1f} deg | "
              f"high-confidence half {np.median(ang[good]):5.1f} deg")
