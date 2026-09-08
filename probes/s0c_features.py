"""Stage 0c, part 1 -- the two geometric CONTROL feature blocks.

Both are rivals the density features must beat, and the plan requires them
reported BEFORE any headline additivity number.

* `own_geom` -- what a folded MONOMER already gives you: own-chain neighbour
  counts in shells, distance to the chain centroid, relative sequence position.
  At Stage 0 this is the meaningful substitute for the plan's `docked_geometry`
  arm: with the deposited complex in hand, "interface from docked-complex
  coordinates" IS the label generator and so is degenerate. The honest Stage-0
  rival is own-chain geometry, i.e. what you get from folding one chain.
* `dens_stats` -- the `[N, Rg, composition]` analogue for density: mean/std of
  the PREPROCESSED map in radial shells about the Ca, plus a boundary statistic
  (shell contrast), which is the actual physics of interface-vs-exposed-surface.
  If the CryoFM features cannot beat these ~20 numbers, the foundation model is
  not earning its place.

Output: data/s0c_features.npz
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probes.s0a_clearance_audit import AA3, chain_ca_zyx  # noqa: E402

SHELL_R = (6.0, 8.0, 10.0, 12.0, 15.0)
DENS_R = (3.0, 6.0, 9.0, 12.0, 15.0)


def own_geom(st, chain_id, ca):
    """[n, 13] own-chain-only geometry. No partner atoms are ever touched."""
    from scipy.spatial import cKDTree
    model = st[0]
    heavy = []
    for ch in model:
        if ch.name != chain_id:
            continue
        for res in ch:
            if res.name not in AA3:
                continue
            for a in res:
                if a.element.atomic_number > 1:
                    heavy.append([a.pos.z, a.pos.y, a.pos.x])
    heavy = np.asarray(heavy, float).reshape(-1, 3)
    n = len(ca)
    t_at, t_ca = cKDTree(heavy), cKDTree(ca)
    cols = [t_at.query_ball_point(ca, r, return_length=True).astype(float)
            for r in SHELL_R]
    cols += [t_ca.query_ball_point(ca, r, return_length=True).astype(float)
             for r in SHELL_R]
    cen = ca.mean(0, keepdims=True)
    d_cen = np.linalg.norm(ca - cen, axis=1)
    rg = float(np.sqrt(((ca - cen) ** 2).sum(1).mean())) or 1.0
    cols += [d_cen, d_cen / rg, np.arange(n) / max(n - 1, 1)]
    return np.stack(cols, 1).astype(np.float32)


def dens_stats(vol, origin, ca, voxel=1.5):
    """[n, 12] radial density statistics from the preprocessed volume."""
    cv = (ca - origin[None]) / voxel
    rad = int(np.ceil(max(DENS_R) / voxel))
    ax = np.arange(-rad, rad + 1)
    dz, dy, dx = np.meshgrid(ax, ax, ax, indexing="ij")
    dist = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2) * voxel
    masks = []
    lo = 0.0
    for r in DENS_R:
        masks.append((dist > lo) & (dist <= r))
        lo = r
    shape = np.asarray(vol.shape)
    out = np.zeros((len(ca), 2 * len(DENS_R) + 2), np.float32)
    for i, c in enumerate(cv):
        ci = np.round(c).astype(int)
        lo_i, hi_i = ci - rad, ci + rad + 1
        if np.any(lo_i < 0) or np.any(hi_i > shape):
            out[i] = np.nan
            continue
        cube = np.asarray(vol[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1],
                              lo_i[2]:hi_i[2]], dtype=np.float32)
        vals = [cube[m] for m in masks]
        means = [float(v.mean()) for v in vals]
        stds = [float(v.std()) for v in vals]
        out[i, :len(DENS_R)] = means
        out[i, len(DENS_R):2 * len(DENS_R)] = stds
        # boundary statistics: how fast density falls off outward
        out[i, -2] = means[0] - means[-1]
        out[i, -1] = means[0] / (abs(means[-1]) + 1e-3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="data/o7_coords.npz")
    ap.add_argument("--chains", default="data/alignment_chains.csv")
    ap.add_argument("--labels", default="data/s0b_labels.npz")
    ap.add_argument("--vol-dir", default="data/cleandift_vols")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-dens", action="store_true")
    ap.add_argument("--out", default="data/s0c_features.npz")
    a = ap.parse_args()

    import gemmi

    lab = np.load(a.labels, allow_pickle=True)
    keys = sorted(set(k.split("/")[0] for k in lab.files
                      if k.endswith("/idx")))
    if a.limit:
        keys = keys[:a.limit]
    rows = {r["key"]: r for r in csv.DictReader(open(a.chains))}

    out, bad = {}, {"error": 0, "no_vol": 0}
    for i, key in enumerate(keys):
        r = rows[key]
        try:
            st = gemmi.read_structure(r["pdb"])
            st.setup_entities()
            st.remove_alternative_conformations()
            _, ca = chain_ca_zyx(st, r["chain"])
            idx = lab[f"{key}/idx"].astype(int)
            out[f"{key}/own_geom"] = own_geom(st, r["chain"], ca)[idx]

            if not a.no_dens:
                npy = Path(a.vol_dir) / f"{r['emd']}.npy"
                meta = Path(a.vol_dir) / f"{r['emd']}.json"
                if not (npy.exists() and meta.exists()):
                    bad["no_vol"] += 1
                else:
                    m = json.loads(meta.read_text())
                    vol = np.load(npy, mmap_mode="r")
                    origin = np.asarray(m["origin_zyx"], float)
                    out[f"{key}/dens_stats"] = dens_stats(
                        vol, origin, ca[idx])
                    del vol
        except Exception as e:  # noqa: BLE001
            bad["error"] += 1
            if bad["error"] <= 3:
                print(f"  ! {key}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    ng = sum(1 for k in out if k.endswith("/own_geom"))
    nd = sum(1 for k in out if k.endswith("/dens_stats"))
    nan = sum(int(np.isnan(out[k]).any(1).sum()) for k in out
              if k.endswith("/dens_stats"))
    print(f"\nown_geom {ng} chains | dens_stats {nd} chains"
          f" | rows with NaN density {nan} | issues {bad}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
