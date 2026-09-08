"""Simulate cryo-EM density from an atomic model (gemmi, electron scattering).

Serves two jobs:

  1. **Clean pose-invariance measurement.** Local-frame features are pose-invariant
     by construction, so what limits them is DISCRETIZATION, not pose. Measuring
     that honestly needs two *independent discretizations* of the same continuous
     object at different orientations -- which you cannot get from a single
     experimental map. Rotating a sampled volume resamples it (double
     interpolation) and overstates the error. Simulating the same model at two
     orientations gives two genuinely independent grids.

  2. **Phase 0d**, the simulated<->experimental gap, which the utility of the whole
     project reduces to.

Uses DensityCalculatorE: electron scattering factors with the Mott-Bethe
correction, i.e. the electrostatic potential that cryo-EM actually images -- not
the X-ray electron density. Note this is the Independent Atom Model, so it
ignores bonding and charge redistribution (charged carboxylates are known to
deviate); it is deterministic and closed-form, NOT exact.
"""

from __future__ import annotations

import gemmi
import numpy as np


def _prepared_atoms(cif_path: str, chain: str | None = None):
    """The exact atom set `simulate` samples, and its centroid.

    Exists so callers can reproduce the rotation centre EXACTLY. `simulate`
    rotates about the ALL-ATOM centroid of the (chain-filtered) model; a caller
    that rotates its Ca coordinates about the *Ca* centroid instead introduces a
    silent offset of `(I - R)(c_CA - c_allatom)`, which is a few Angstrom and
    therefore invisible in code review but not in the features. Use this rather
    than recomputing a centroid by eye.
    """
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if chain is not None:
        for model in st:
            for ch in [c.name for c in model if c.name != chain]:
                model.remove_chain(ch)
    pos = np.array([[a.pos.x, a.pos.y, a.pos.z]
                    for m in st for c in m for r in c for a in r], dtype=np.float64)
    if len(pos) == 0:
        raise ValueError(f"{cif_path}: no atoms"
                         + (f" for chain {chain!r}" if chain else ""))
    return st, pos, pos.mean(0)


def model_centre(cif_path: str, chain: str | None = None) -> np.ndarray:
    """Centroid `simulate` rotates about, for the same (path, chain)."""
    return _prepared_atoms(cif_path, chain)[2]


def _good_fft_size(n: int) -> int:
    """Smallest even 5-smooth integer >= n (radix-2/3/5 FFT-friendly)."""
    n = max(2, int(n))
    while True:
        m, k = n, {}
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
                k[f] = k.get(f, 0) + 1
        if m == 1 and n % 2 == 0:
            return n
        n += 1


def simulate(cif_path: str, d_min: float = 3.0, voxel: float = 1.5,
             blur: float = 0.0, rotation: np.ndarray | None = None,
             margin: float = 12.0, chain: str | None = None):
    """Simulate density on an EXACTLY ISOTROPIC grid of the requested spacing.

    Returns ``(vol[z,y,x], origin_xyz, spacing_zyx)``, where ``spacing`` is
    ``[voxel, voxel, voxel]`` to floating-point exactness. It is still returned
    (rather than left implicit) so callers cannot silently reintroduce the bug
    below by hardcoding a constant that a future change invalidates.

    **Why this is not the default gemmi behaviour, and why it matters.** Left to
    itself, gemmi picks each grid dimension independently as the next FFT-friendly
    size and keeps the cell fixed, so the realized spacing is ``span_i / n_i``:
    never the request, ANISOTROPIC by ~7-11%, and POSE-DEPENDENT, since rotating
    the model changes the bounding box and hence the rounding. Measured on 9w0g at
    a requested 1.5 A:

        identity  grid (90,96,80)  spacing [1.347 1.442 1.409]   aniso 1.071
        rotated   grid (90,90,80)  spacing [1.478 1.337 1.406]   aniso 1.105

    A pose-dependent anisotropic scaling is a shear, so anything built from voxel
    indices is not equivariant *by construction*. This one defect accounted for:
    the entire "0.3-6.3 deg residual frame error" in whole-map canonicalisation
    (true value 0.00-0.01 deg); canonicalised volume correlation 0.85-0.90 vs a
    true 0.99+; the false conclusion that C_n symmetric complexes break the
    canonicaliser (the shear masked the degeneracy test); and a median 8.8 A Ca
    mislocation in probes/pose_invariance_clean.py, against a box only ~12 A wide.

    The fix is to expand the cell to ``n_i * voxel`` for an FFT-friendly ``n_i``
    and set the grid size explicitly, rather than letting gemmi round the size
    against a fixed cell. Cost is a slightly larger margin on the +side of each
    axis; the origin is unchanged.

    `rotation` is a 3x3 applied to the ATOMIC COORDINATES before sampling, which
    is what makes the resulting grid an independent discretization rather than a
    resampling of another grid.
    """
    st, pos, centre = _prepared_atoms(cif_path, chain)
    if rotation is not None:
        pos = (pos - centre) @ np.asarray(rotation, dtype=np.float64).T + centre

    lo = pos.min(0) - margin
    hi = pos.max(0) + margin
    # Round the CELL up to a whole number of voxels on an FFT-friendly grid, so
    # that spacing is exactly `voxel` on every axis instead of span/round(span).
    nvox = np.array([_good_fft_size(int(np.ceil(s / voxel))) for s in (hi - lo)])
    span = nvox * voxel

    # Re-place atoms into a P1 box whose corner is the grid origin.
    i = 0
    for m in st:
        for c in m:
            for r in c:
                for a in r:
                    a.pos = gemmi.Position(*(pos[i] - lo))
                    i += 1
    st.cell = gemmi.UnitCell(*span, 90, 90, 90)
    st.spacegroup_hm = "P 1"

    rate = d_min / (2.0 * voxel)
    if rate < 1.0:
        raise ValueError(
            f"voxel={voxel} is coarser than d_min/2={d_min/2}; gemmi clamps the rate "
            "to 1.0 and silently ignores the request. Lower `voxel` or raise `d_min`.")

    dc = gemmi.DensityCalculatorE()
    dc.d_min = d_min
    dc.blur = blur
    dc.rate = rate
    dc.set_grid_cell_and_spacegroup(st)
    # Set the size ourselves instead of `put_model_density_on_grid`, whose
    # `initialize_grid()` would re-derive it from d_min/rate against the fixed
    # cell and reintroduce the anisotropy this function exists to avoid.
    dc.grid.set_size(int(nvox[0]), int(nvox[1]), int(nvox[2]))
    dc.add_model_density_to_grid(st[0])
    dc.grid.symmetrize_sum()                               # no-op in P1

    g = dc.grid
    arr = np.array(g, copy=True, dtype=np.float32)         # [x, y, z]
    spacing_xyz = span / np.array([g.nu, g.nv, g.nw], dtype=np.float64)
    assert np.allclose(spacing_xyz, voxel), (
        f"grid spacing {spacing_xyz} != requested {voxel}; gemmi resized the grid")
    return (np.ascontiguousarray(arr.transpose(2, 1, 0)),  # [z, y, x]
            lo,                                            # origin, xyz
            spacing_xyz[::-1].copy())                      # spacing, zyx


def simulate_pair(cif_path: str, rotation: np.ndarray, **kw):
    """Two independent discretizations of the same model: identity and `rotation`.

    Each element is ``(vol, origin_xyz, spacing_zyx)``. The two grids generally
    have DIFFERENT spacing -- see `simulate` -- so callers must use each grid's own
    spacing rather than a shared constant.
    """
    return simulate(cif_path, rotation=None, **kw), simulate(cif_path, rotation=rotation, **kw)


if __name__ == "__main__":
    import sys

    cif = sys.argv[1] if len(sys.argv) > 1 else "data/maps/9w0g.cif"
    vol, origin, sp = simulate(cif, d_min=3.0, voxel=1.5)
    print(f"{cif}\n  grid {vol.shape}  origin {np.round(origin,1)}  spacing {np.round(sp,3)}")
    print(f"  min {vol.min():.4f}  max {vol.max():.4f}  mean {vol.mean():.4f}  "
          f"nonzero {np.mean(vol > 0.01):.1%}")

    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)   # 90 deg about z
    (a, oa, sa), (b, ob, sb) = simulate_pair(cif, R, d_min=3.0, voxel=1.5)
    print(f"  identity {a.shape} spacing {np.round(sa,3)}")
    print(f"  rotated  {b.shape} spacing {np.round(sb,3)}   <- spacing must MATCH")
    assert np.allclose(sa, sb) and np.allclose(sa, 1.5), "spacing is pose-dependent"
    # Box shapes still differ (the bounding box rotates); only the spacing must not.
    # Total mass must be conserved: a rotation cannot create or destroy density.
    print(f"  sum(a)={a.sum():.1f}  sum(b)={b.sum():.1f}  "
          f"ratio={b.sum()/max(a.sum(),1e-9):.4f}  (want ~1.000)")
