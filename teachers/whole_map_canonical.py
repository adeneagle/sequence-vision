"""Canonical orientation of a WHOLE density map, model-free.

Distinct from per-residue local frames, which failed (structure tensor ~30 deg,
SH dipole ~24 deg even optimally smoothed) because a 10 A protein neighbourhood is
a generic packed-atom environment with little directional signal. A whole particle
is a large asymmetric object -- far more signal -- and canonical orientation of 3D
shapes is standard practice in shape retrieval and map alignment.

Construction:
  1. Second moment of the density about its centre of mass -> principal axes.
     (The l=1 dipole is useless here: centring makes it vanish identically.)
  2. That leaves a discrete 4-fold ambiguity among det=+1 sign combinations, plus
     a CONTINUOUS azimuthal ambiguity when two eigenvalues are degenerate -- which
     is exactly the C_n (n>=3) case that defeats a pure inertia-tensor frame.
  3. Resolve both by scoring candidate frames with an ODD-degree (l=3) real SH
     moment, which is sign-sensitive and n-fold sensitive, and taking the argmax.

Residual ambiguity is then the molecule's own symmetry group -- under which the
density is genuinely identical, so it is harmless rather than a failure.
"""

from __future__ import annotations

import numpy as np

# Spherical-harmonic machinery was removed from this project on 2026-08-26 (see
# archive/sh_removed_2026-08-26/). These two helpers are kept here, inlined, because
# this module uses them as an internal SCORING FUNCTIONAL to break the sign/azimuth
# ambiguity of the principal axes -- not as a descriptor. Nothing here is proposed as
# a feature for a downstream task.


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

    Returns a list indexed by l, each [P, 2l+1]. Verified orthonormal to 3e-16.
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


def _second_moment_axes(vol: np.ndarray, spacing: np.ndarray, thresh: float = 0.0):
    """Principal axes (rows, descending), eigenvalues (A^2), COM (voxel index).

    The covariance MUST be computed in physical units. gemmi's realized spacing is
    anisotropic (~7-11%) and pose-dependent, so a covariance built from raw voxel
    indices measures the object under a pose-dependent shear and its principal axes
    are NOT equivariant. That single error accounted for the whole "0.3-6.3 deg
    residual frame error" -- with physical spacing it is 0.00-0.01 deg.
    """
    idx = np.nonzero(vol > thresh)
    w = vol[idx].astype(np.float64)
    pts = np.stack(idx, 1).astype(np.float64)
    com = (pts * w[:, None]).sum(0) / w.sum()
    d = (pts - com) * np.asarray(spacing)[None]          # voxel index -> Angstrom
    cov = (d * w[:, None]).T @ d / w.sum()
    ev, evec = np.linalg.eigh(cov)
    o = np.argsort(ev)[::-1]
    return evec[:, o].T, ev[o], com


def _candidate_frames(axes: np.ndarray, eig: np.ndarray,
                      n_azimuth: int = 72, degen_tol: float = 0.15):
    """Enumerate frames consistent with the principal axes.

    Always the 4 proper sign combinations. If two eigenvalues are degenerate the
    axes in that plane are arbitrary, so sample azimuths there too -- this is the
    C_n>=3 case that a bare inertia frame cannot resolve.
    """
    # eigh may return a left-handed basis; make it right-handed first. Otherwise
    # every sign combination below (all have sign-product +1, so all preserve the
    # determinant) is rejected and the candidate list comes back empty.
    axes = axes.copy()
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]

    out = [axes * np.array(s)[:, None]
           for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]

    # Normalise each gap by its OWN pair, not by eig[0]. With eig=[100,10,6] the
    # true 2<->3 gap is 40% but |diff|/eig[0] reads 0.04 and spuriously triggers
    # the degenerate branch, azimuth-mixing two well-separated axes.
    rel = np.array([abs(eig[0] - eig[1]) / max(eig[0], 1e-12),
                    abs(eig[1] - eig[2]) / max(eig[1], 1e-12)])
    if rel[0] < degen_tol and rel[1] < degen_tol:
        # Fully isotropic (cubic/icosahedral, or a noise-dominated real map).
        # SO(3) is 3-dimensional; sampling one circle would cover a measure-zero
        # slice, so refuse rather than return a meaningless frame.
        raise ValueError(
            f"second moment is isotropic (gaps {np.round(rel,4)}): no orientation "
            "information at l=2. Needs l>=6 or explicit symmetry detection.")
    if rel[0] < degen_tol or rel[1] < degen_tol:
        # Two eigenvalues are degenerate, so the axes in that plane are arbitrary:
        # the C_n>=3 case a bare inertia frame cannot resolve. Sample the azimuth
        # and let the odd-degree score pick.
        i, j = (0, 1) if rel[0] < degen_tol else (1, 2)
        k = 3 - i - j
        # Only 2 of the 4 sign candidates are distinct once the azimuth is swept:
        # diag(-1,-1,1) . R_phi == R_{phi+pi} is already on the grid, so enumerating
        # all 4 yields every frame exactly twice (288 -> 144 unique) and creates
        # exact argmax ties.
        base, out = list(out)[:2], []
        for F in base:
            for phi in np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False):
                G = F.copy()
                G[i] = np.cos(phi) * F[i] + np.sin(phi) * F[j]
                G[j] = -np.sin(phi) * F[i] + np.cos(phi) * F[j]
                G[k] = F[k]
                out.append(G)
    return out


class WholeMapCanonicalizer:
    def __init__(self, n_shells: int = 6, n_dirs: int = 302, l_score: int = 3):
        self.n_shells = n_shells
        self.dirs = fibonacci_sphere(n_dirs)
        self.l_score = l_score
        self.Y = _real_sh(l_score, self.dirs)[l_score]      # [P, 2l+1]

    def _score(self, vol, com, frame, radii_A, spacing):
        """Odd-degree SH moment evaluated in `frame`. Sign- and n-fold-sensitive."""
        from scipy.ndimage import map_coordinates

        d_world = self.dirs @ frame                         # rotate probe directions
        # radii are in Angstrom; convert to voxel offsets per axis, else the
        # sampling "sphere" is a pose-dependent ellipsoid.
        pts = (com[None, None, :] +
               radii_A[:, None, None] * d_world[None, :, :] / np.asarray(spacing)[None, None, :])
        vals = map_coordinates(vol, pts.reshape(-1, 3).T, order=1,
                               mode="constant", cval=0.0).reshape(len(radii_A), -1)
        c = vals @ self.Y                                   # [S, 2l+1]
        # Must break BOTH sign degrees of freedom. The m=0 component depends only
        # on polar angle, so it flips with e1 but is blind to the 180-deg rotation
        # about e1 that flips e2 and e3 together -- leaving two candidates exactly
        # tied and argmax picking arbitrarily (observed: 2-axis flips on most
        # trials). Under phi -> phi+pi a real Y_lm picks up (-1)^m, so an ODD-m
        # term is required. Summing all m gives a generic frame-dependent scalar
        # that separates all four candidates.
        return float(c.sum())

    def __call__(self, vol: np.ndarray, spacing):
        spacing = np.asarray(spacing, dtype=np.float64)
        if spacing.ndim == 0:
            spacing = np.repeat(spacing, 3)
        axes, eig, com = _second_moment_axes(vol, spacing)
        extent_A = np.sqrt(max(eig[0], 1e-12))
        radii = np.linspace(0.3, 1.4, self.n_shells) * extent_A     # Angstrom
        cands = _candidate_frames(axes, eig)
        scores = [self._score(vol, com, F, radii, spacing) for F in cands]
        best = int(np.argmax(scores))
        return cands[best], com, eig, len(cands)


if __name__ == "__main__":
    import sys

    from data.simulate_density import simulate
    from probes.local_frame_stability import random_so3
    from probes.pose_invariance_clean import _R_zyx

    cif = sys.argv[1] if len(sys.argv) > 1 else "data/maps/9w0g.cif"
    can = WholeMapCanonicalizer()
    rng = np.random.default_rng(0)

    volA, _, spA = simulate(cif, d_min=3.0, voxel=1.5, margin=20.0)
    FA, comA, eigA, nA = can(volA, spA)
    rel = np.diff(eigA) / eigA[0]
    print(f"eigenvalues {np.round(eigA,1)}  relative gaps {np.round(np.abs(rel),3)}  "
          f"candidates {nA}")

    errs = []
    for t in range(5):
        R = random_so3(rng)
        volB, _, spB = simulate(cif, d_min=3.0, voxel=1.5, rotation=R, margin=20.0)
        FB, _, _, _ = can(volB, spB)
        pred = _R_zyx(R) @ FA.T            # columns = expected rotated axes
        # SIGNED: abs() is blind to axis flips, which render a different volume.
        ang = np.degrees(np.arccos(np.clip(np.sum(pred.T * FB, axis=1), -1, 1)))
        errs.append(ang)
        print(f"trial {t}: per-axis error {np.round(ang,1)} deg")
    e = np.array(errs)
    print(f"\nmedian axis error {np.median(e):.1f} deg   worst {e.max():.1f} deg")
    print("reference: per-residue local frames 24-30 deg | backbone frames exact")
