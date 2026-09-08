"""Homolog diagnostic: does a density descriptor vary SMOOTHLY with sequence?

This is the cheap gate that should have run before any of the pose work. A
sequence model can only exist if the density feature it predicts is (a) not
swamped by orientation noise and (b) *closer for homologs than for unrelated
proteins*. (b) is the part nothing so far has tested: a feature can be perfectly
pose-invariant and perfectly discriminative and still be unpredictable from
sequence, because it varies chaotically with it. Then no model can interpolate.

Three comparison levels, all on the same descriptor:

    pose       same chain, two independent SO(3) orientations   -> noise floor
    homolog    30-60% sequence identity, different entries      -> the signal
    unrelated  no detectable alignment, LENGTH-MATCHED          -> full spread

Written as a VARIANCE DECOMPOSITION, not a threshold test. Two invented
thresholds have already failed in this project; a quantity that bounds what any
downstream model can achieve needs no pass mark. With
`d = mu + f(structure) + eps_pose` and squared distances,

    D2_pose  = 2 s2_pose
    D2_hom   = 2 s2_pose + 2 s2_within
    D2_unrel = 2 s2_pose + 2 s2_between

so the headline is

    homolog_signal = (D2_unrel - D2_hom) / (D2_unrel - D2_pose) = 1 - s2_within/s2_between

1.0 = homologs map to identical features; 0.0 = homologs are no closer than
unrelated proteins, i.e. the target is unpredictable from sequence and the
project stops. Also reported: pose_fraction = D2_pose / D2_unrel, the share of
the total spread that is pure orientation nuisance.

**Features come from SIMULATED density**, both members of a pair at identical
d_min and on an exactly isotropic grid. Comparing two EXPERIMENTAL maps of
homologs would confound structure with resolution, sharpening, solvent, box and
reconstruction -- it would measure the instrument. Simulation is the clean case,
so a failure here is decisive while a pass still needs the experimental check.

**A descriptor LADDER, not one number**, so that a negative is interpretable:

    cryofm          the actual target
    cryofm_random   same architecture, no checkpoint -- separates learned
                    content from "any 3D conv over density"
    trivial         [n_res, Rg, 20-d AA composition]. Mandatory: two scalars
                    explained 84.5% of a naive pair-distance descriptor.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.simulate_density import model_centre, simulate
from probes.local_frame_stability import random_so3
from teachers.cryofm_tap import (MODEL_VOXEL_SIZE, CryoFM2Tap, preprocess,
                                 cube_rotations, rotate_coords, rotate_volume,
                                 sample_at)

AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def chain_ca(pdb_path: str, chain_id: str):
    """(Ca xyz [N,3], residue-name list) for one chain of model 0."""
    import gemmi
    st = gemmi.read_structure(pdb_path)
    st.setup_entities()
    st.remove_alternative_conformations()
    xyz, names = [], []
    for model in st:
        for ch in model:
            if ch.name != chain_id:
                continue
            for res in ch:
                a = res.find_atom("CA", "*")
                if a is not None:
                    xyz.append([a.pos.x, a.pos.y, a.pos.z])
                    names.append(res.name)
        break
    return np.array(xyz, dtype=np.float64), names


def pool(x: np.ndarray) -> np.ndarray:
    """mean+std over residues -- same construction on both sides of the study."""
    return np.concatenate([x.mean(0), x.std(0)])


def _experimental(pdb, chain, R, args):
    """Volume + Ca voxel coords from the EXPERIMENTAL map, not a simulation.

    The OOD control. CryoFM2 was trained on experimental maps, so clean
    simulated density is out of distribution for it -- which is an alternative
    explanation for a poor CryoFM result that the simulated arm cannot rule out.
    Note the headline `r2_ceiling = (D2unrel - D2hom)/D2unrel` does not involve
    the pose term at all, so this arm needs no independent re-discretisation;
    `R` here is a lossless CUBE rotation of the sampled volume, reported only for
    context and optimistic by construction (it is CryoFM2's augmentation group).

    Caveat that does not apply to the simulated arm: the raw map contains EVERY
    chain, so features near the pooled chain include its neighbours, and the two
    members of a pair differ in resolution, sharpening and reconstruction as well
    as in structure. Both effects add nuisance variance, so this arm is a LOWER
    bound on the ceiling.
    """
    from probes.stability import load_map
    d = Path(pdb).parent
    mp = d / f"{d.name}_raw_emd.map"
    if not mp.exists():
        raise FileNotFoundError(str(mp))
    vol, vs, origin_A = load_map(str(mp))
    norm = preprocess(vol, vs)
    xyz, names = chain_ca(pdb, chain)
    coords = ((xyz[:, ::-1] - origin_A[None]) / MODEL_VOXEL_SIZE)
    pre_shape = norm.shape
    if R is not None:
        norm = rotate_volume(norm, R)
        coords = rotate_coords(coords, R, pre_shape)
    return norm, coords, xyz, names


def descriptors(pdb, chain, R, taps, tap_real, tap_rand, args):
    """All descriptors for one chain at one orientation."""
    if args.experimental:
        norm, coords, xyz, names = _experimental(pdb, chain, R, args)
        vol = norm
        shape = np.array(norm.shape)
        keep = np.all((coords >= 1) & (coords < shape[None] - 2), axis=1)
        if keep.sum() < 20:
            raise ValueError(f"only {int(keep.sum())} Ca inside the map")
        coords = coords[keep]
        idx = np.rint(coords).astype(int)
        contrast = float(norm[idx[:, 0], idx[:, 1], idx[:, 2]].mean() - norm.mean())
        if contrast < 0.5:
            raise ValueError(f"Ca/bulk density contrast only {contrast:+.2f}")
        return _finish(vol, norm, coords, xyz, names, taps, tap_real, tap_rand, args)

    vol, origin, spacing = simulate(
        pdb, d_min=args.d_min, voxel=MODEL_VOXEL_SIZE, rotation=R,
        chain=chain, margin=args.margin)
    xyz, names = chain_ca(pdb, chain)
    if R is not None:
        # MUST be simulate()'s own rotation centre (all-atom centroid of the
        # chain), not the Ca centroid -- otherwise the coordinates drift off the
        # density by (I-R)(c_CA - c_allatom) and the "pose noise" measured here
        # is partly a coordinate bug.
        c = model_centre(pdb, chain)
        xyz = (xyz - c) @ np.asarray(R).T + c
    vox_xyz = (xyz - np.asarray(origin)[None]) / np.asarray(spacing)[::-1][None]
    coords = vox_xyz[:, ::-1].copy()                       # -> [z,y,x]

    shape = np.array(vol.shape)
    keep = np.all((coords >= 1) & (coords < shape[None] - 2), axis=1)
    if keep.sum() < 20:
        raise ValueError(f"only {int(keep.sum())} Ca inside the grid")
    coords = coords[keep]
    return _finish(vol, preprocess(vol, MODEL_VOXEL_SIZE), coords, xyz, names,
                   taps, tap_real, tap_rand, args)


def _finish(vol, norm, coords, xyz, names, taps, tap_real, tap_rand, args):
    """Descriptor ladder, shared by the simulated and experimental arms."""
    out = {}
    for label, tp in (("cryofm", tap_real), ("cryofm_random", tap_rand)):
        if tp is None:
            continue
        fvs = tp.feature_volumes(norm, timestep=args.timestep)
        for t in taps:
            out[f"{label}:{t}"] = pool(sample_at(fvs[t], coords))
    comp = np.zeros(20)
    for n in names:
        if n in AA3:
            comp[AA3.index(n)] += 1
    comp = comp / max(comp.sum(), 1)
    rg = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
    out["trivial"] = np.concatenate([[len(names), rg], comp])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/homolog_pairs.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--limit", type=int, default=80, help="pairs per kind")
    ap.add_argument("--d-min", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=16.0)
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--taps", nargs="*", default=["mid_block", "up_blocks[1]"])
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--experimental", action="store_true",
                    help="use the raw EMDB map instead of simulating -- the OOD\n                          control for CryoFM, which was trained on real maps")
    ap.add_argument("--out", type=Path, default=Path("results/homolog_diagnostic.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.pairs)))
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    pairs = [r for k in ("homolog", "unrelated") for r in by_kind[k][: args.limit]]

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tap_real = CryoFM2Tap(args.ckpt, taps=tuple(args.taps), device=dev,
                          batch_size=args.batch_size)
    tap_rand = None if args.no_random else CryoFM2Tap(
        args.ckpt, taps=tuple(args.taps), device=dev,
        batch_size=args.batch_size, random_weights=True)
    rng = np.random.default_rng(0)
    cube_rots = cube_rotations()

    # Each structure appears in several pairs; compute once.
    need = {}
    for r in pairs:
        for s in ("a", "b"):
            need[(r[f"pdb_{s}"], r[f"chain_{s}"])] = None

    print(f"{len(pairs)} pairs -> {len(need)} unique chains, device={dev}")
    cache: dict[tuple, tuple[dict, dict]] = {}
    for i, key in enumerate(need):
        pdb, chain = key
        try:
            # Two INDEPENDENT orientations: both simulated from scratch, so the
            # difference is discretization, not the resampling of one grid.
            #
            # Note the homolog/unrelated comparisons below use the DEPOSITED
            # orientation of each structure, which is arbitrary and different for
            # every entry. That is deliberate and self-consistent: a sequence
            # model must predict the descriptor whatever the pose, and the
            # residual orientation dependence is exactly what D2_pose measures,
            # so the decomposition subtracts it off.
            d0 = descriptors(pdb, chain, None, args.taps, tap_real, tap_rand, args)
            # The experimental arm can only rotate an ALREADY-SAMPLED volume, so
            # its rotation must come from the cube group -- `rotate_volume` is a
            # transpose+flip and a continuous SO(3) matrix silently degenerates
            # there (duplicate argmax axes, or a valid-looking permutation that
            # puts the coordinates off the density). That makes the experimental
            # pose term optimistic, being CryoFM2's own augmentation group; it
            # does not touch `r2_ceiling`, which has no pose term.
            R = (cube_rots[rng.integers(1, len(cube_rots))] if args.experimental
                 else random_so3(rng))
            d1 = descriptors(pdb, chain, R, args.taps, tap_real, tap_rand, args)
            cache[key] = (d0, d1)
        except Exception as exc:
            print(f"  [{i+1}/{len(need)}] SKIP {Path(pdb).stem}:{chain} "
                  f"{type(exc).__name__}: {exc}")
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(need)}] {len(cache)} ok")

    if not cache:
        raise SystemExit(
            'no chain yielded descriptors -- every one was skipped. Check the '
            'Ca/bulk contrast messages above: a systematic ~0 contrast means a '
            'map origin or coordinate-frame error, not a property of the data.')
    order = list(cache)
    np.savez_compressed(
        args.out.with_suffix(".descriptors.npz"),
        keys=np.array([f"{Path(p).stem}:{c}" for p, c in order]),
        **{f"{n}|{o}": np.stack([cache[k][o][n] for k in order])
           for n in cache[order[0]][0] for o in (0, 1)})

    def partial_out(M0, M1, T):
        """Remove the linear span of the trivial descriptor T from M0 and M1.

        Fitted ONCE on the reference orientation and applied to both, because T
        is a property of the structure (rotation preserves Rg, composition and
        length), so the two orientations must be projected identically or the
        subtraction would itself inject pose variance.
        """
        X = np.concatenate([T, np.ones((len(T), 1))], 1)
        beta, *_ = np.linalg.lstsq(X, M0, rcond=None)
        return M0 - X @ beta, M1 - X @ beta

    keys = sorted({k for v in cache.values() for k in v[0]})
    Tmat = np.stack([cache[k][0]["trivial"] for k in order])
    Tmat = (Tmat - Tmat.mean(0)) / np.where(Tmat.std(0) < 1e-9, 1.0, Tmat.std(0))

    idx_of = {k: i for i, k in enumerate(order)}

    def decompose(M0, M1):
        """Three median squared distances from the two orientation matrices."""
        sd = M0.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z0, Z1 = (M0 - M0.mean(0)) / sd, (M1 - M0.mean(0)) / sd
        d2 = {"pose": ((Z0 - Z1) ** 2).sum(1).tolist(), "homolog": [], "unrelated": []}
        for r in pairs:
            ka, kb = (r["pdb_a"], r["chain_a"]), (r["pdb_b"], r["chain_b"])
            if ka not in idx_of or kb not in idx_of:
                continue
            d2[r["kind"]].append(
                float(((Z0[idx_of[ka]] - Z0[idx_of[kb]]) ** 2).sum()))
        if not all(d2[k] for k in ("pose", "homolog", "unrelated")):
            return None
        m = {k: float(np.median(v)) for k, v in d2.items()}
        denom = m["unrelated"] - m["pose"]
        return {
            "n": {k: len(v) for k, v in d2.items()},
            "D2_pose": m["pose"], "D2_homolog": m["homolog"],
            "D2_unrelated": m["unrelated"],
            # The headline. Fraction of variance tracking sequence similarity at
            # homolog resolution -- a smoothness measure, NOT a strict bound on a
            # trained model (see module docstring). The hard ceiling is the pose
            # term alone: 1 - pose_fraction.
            "r2_ceiling": float((m["unrelated"] - m["homolog"]) / m["unrelated"]),
            # Same numerator on the pose-free spread: how much of the BIOLOGICAL
            # variation homology accounts for, ignoring the nuisance.
            "homolog_signal": float((m["unrelated"] - m["homolog"]) / denom)
            if denom > 0 else None,
            "pose_fraction": float(m["pose"] / m["unrelated"]),
            # The original CLAUDE.md form of the gate: >= 1 means dead.
            "pose_over_homolog": float(m["pose"] / m["homolog"]),
        }

    results = {}
    for name in keys:
        M0 = np.stack([cache[k][0][name] for k in order])
        M1 = np.stack([cache[k][1][name] for k in order])
        raw = decompose(M0, M1)
        if raw is None:
            continue
        results[name] = raw
        # Partial out size + composition. On this project's own history that is
        # the difference between a result and an artifact: two scalars explained
        # 84.5% of a naive pair-distance descriptor, and `trivial` scores highest
        # of anything in the raw table here. Anything CryoFM contributes must
        # survive its removal.
        if name != "trivial":
            R0, R1 = partial_out(M0, M1, Tmat)
            res = decompose(R0, R1)
            if res is not None:
                results[f"{name} [-trivial]"] = res

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 86)
    print(f"{'descriptor':<34}{'R2ceil':>9}{'signal':>9}{'posefrac':>10}"
          f"{'pose/hom':>10}")
    print("-" * 86)
    for name, r in results.items():
        sig = "n/a" if r["homolog_signal"] is None else f"{r['homolog_signal']:.3f}"
        print(f"{name:<34}{r['r2_ceiling']:>9.3f}{sig:>9}"
              f"{r['pose_fraction']:>10.3f}{r['pose_over_homolog']:>10.3f}")
    print("=" * 86)
    print("R2ceil   = (D2unrel - D2hom)/D2unrel -- fraction of variance tracking")
    print("           sequence similarity at homolog resolution. A smoothness")
    print("           measure, NOT a strict bound: the hard ceiling is 1-posefrac.")
    print("signal   = (D2unrel - D2hom)/(D2unrel - D2pose) = 1 - s2_within/s2_between")
    print("           1.0 homologs identical | 0.0 homologs no closer than unrelated")
    print("pose/hom = the CLAUDE.md gate; >= 1.0 means orientation noise swamps")
    print("           the homolog signal and the target is dead")
    print("Compare cryofm vs cryofm_random (learned content) and vs trivial")
    print("(size + composition, the floor any descriptor must clear).")


if __name__ == "__main__":
    main()
