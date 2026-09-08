"""Stage 0a -- the clearance audit. GATES all of Stage 0.

`probes/o5_boxes.chain_boxes` keeps only residues whose Ca sits at least
PATCH//2 = 32 voxels = 48 A from every edge of the map volume, so that a full
64^3 box can be cut. CLAUDE.md records this silently discarding *every surface
residue* on simulated density, biasing the sample to buried environments.

Interface and pocket residues ARE surface residues. If the dropped residues are
materially enriched for the interface label, then every additivity read taken on
the existing box-path caches is measured on the wrong population and a null
result would be uninterpretable.

This script measures that enrichment directly. It reads only the volume-cache
JSON sidecars (origin/shape) and the fitted PDBs -- no volumes, no GPU.

Output: results/s0a_clearance_audit.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
NUC3 = {"A", "C", "G", "U", "DA", "DC", "DG", "DT", "I", "DI"}


def chain_ca_zyx(st, chain_id):
    """Ca of residues with a complete N/CA/C backbone -- exactly the residues
    `chain_backbone` keeps, so the audit population matches the cached one."""
    import gemmi  # noqa: F401
    out_idx, out_ca = [], []
    model = st[0]
    for ch in model:
        if ch.name != chain_id:
            continue
        for i, res in enumerate(ch):
            if res.name not in AA3:
                continue
            aN, aCA, aC = (res.find_atom(n, "*") for n in ("N", "CA", "C"))
            if aN is None or aCA is None or aC is None:
                continue
            out_idx.append(i)
            out_ca.append([aCA.pos.z, aCA.pos.y, aCA.pos.x])
    return np.asarray(out_idx, int), np.asarray(out_ca, float).reshape(-1, 3)


def residue_heavy(st, chain_id, keep_idx):
    """Heavy atoms of the target chain grouped by residue -> (xyz[M,3], res[M])."""
    model = st[0]
    pos, rid = [], []
    want = set(int(i) for i in keep_idx)
    order = {int(v): k for k, v in enumerate(keep_idx)}
    for ch in model:
        if ch.name != chain_id:
            continue
        for i, res in enumerate(ch):
            if i not in want:
                continue
            for a in res:
                if a.element.atomic_number <= 1:
                    continue
                pos.append([a.pos.z, a.pos.y, a.pos.x])
                rid.append(order[i])
    return np.asarray(pos, float).reshape(-1, 3), np.asarray(rid, int)


def partner_heavy(st, chain_id):
    """Heavy atoms of every OTHER chain instance, split protein / nucleic."""
    model = st[0]
    prot, nuc = [], []
    for ch in model:
        if ch.name == chain_id:
            continue
        for res in ch:
            tgt = prot if res.name in AA3 else (nuc if res.name in NUC3 else None)
            if tgt is None:
                continue
            for a in res:
                if a.element.atomic_number <= 1:
                    continue
                tgt.append([a.pos.z, a.pos.y, a.pos.x])
    f = lambda L: np.asarray(L, float).reshape(-1, 3)
    return f(prot), f(nuc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default="data/alignment_chains.csv")
    ap.add_argument("--vol-dir", default="data/cleandift_vols")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--voxel", type=float, default=1.5)
    ap.add_argument("--out", default="results/s0a_clearance_audit.json")
    a = ap.parse_args()

    import csv
    import gemmi
    from scipy.spatial import cKDTree

    rows = list(csv.DictReader(open(a.chains)))
    rng = np.random.default_rng(0)
    if a.limit and a.limit < len(rows):
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.limit, replace=False))]

    half = a.patch // 2
    per_chain, skipped = [], {"no_sidecar": 0, "no_residues": 0, "error": 0}

    for r in rows:
        meta = Path(a.vol_dir) / f"{r['emd']}.json"
        if not meta.exists():
            skipped["no_sidecar"] += 1
            continue
        try:
            m = json.loads(meta.read_text())
            origin = np.asarray(m["origin_zyx"], float)
            shape = np.asarray(m["shape"], int)

            st = gemmi.read_structure(r["pdb"])
            st.setup_entities()
            st.remove_alternative_conformations()

            idx, ca = chain_ca_zyx(st, r["chain"])
            if len(idx) < 8:
                skipped["no_residues"] += 1
                continue

            # the exact clearance filter used by the cached box path
            cv = (ca - origin[None]) / a.voxel
            keep = np.all((cv >= half) & (cv < shape[None] - half), axis=1)

            hpos, hres = residue_heavy(st, r["chain"], idx)
            ppos, npos = partner_heavy(st, r["chain"])
            n = len(idx)

            def label(part):
                y = np.zeros(n, bool)
                if len(part) == 0 or len(hpos) == 0:
                    return y
                hit = cKDTree(part).query_ball_point(hpos, a.cutoff, return_length=True) > 0
                if hit.any():
                    y[np.unique(hres[hit])] = True
                return y

            iface = label(ppos)
            ifn = label(npos)

            per_chain.append(dict(
                key=r["key"], emd=r["emd"], split=r["split"], n=int(n),
                n_keep=int(keep.sum()),
                iface_all=float(iface.mean()),
                iface_keep=float(iface[keep].mean()) if keep.any() else float("nan"),
                iface_drop=float(iface[~keep].mean()) if (~keep).any() else float("nan"),
                nuc_all=float(ifn.mean()),
                shape=int(shape[0]),
            ))
        except Exception as e:  # noqa: BLE001
            skipped["error"] += 1
            if skipped["error"] <= 3:
                print(f"  ! {r['key']}: {type(e).__name__}: {e}", flush=True)

        if len(per_chain) % 50 == 0 and per_chain:
            print(f"  {len(per_chain)} chains", flush=True)

    if not per_chain:
        raise SystemExit("no chains processed")

    # POOLED over residues -- the population an additivity probe actually sees.
    N = np.array([c["n"] for c in per_chain])
    K = np.array([c["n_keep"] for c in per_chain])
    ia = np.array([c["iface_all"] for c in per_chain])
    ik = np.array([c["iface_keep"] for c in per_chain])
    idr = np.array([c["iface_drop"] for c in per_chain])
    D = N - K
    pooled_keep = float(np.nansum(ik * K) / max(K.sum(), 1))
    pooled_drop = float(np.nansum(idr * D) / max(D.sum(), 1))
    pooled_all = float(np.nansum(ia * N) / max(N.sum(), 1))

    res = dict(
        _meta=dict(chains=len(per_chain), skipped=skipped, cutoff_A=a.cutoff,
                   clearance_A=half * a.voxel, limit=a.limit),
        residues=dict(total=int(N.sum()), kept=int(K.sum()), dropped=int(D.sum()),
                      drop_fraction=float(D.sum() / N.sum())),
        interface_rate=dict(all=pooled_all, kept=pooled_keep, dropped=pooled_drop,
                            enrichment_drop_over_keep=float(pooled_drop / pooled_keep)
                            if pooled_keep > 0 else None),
        chains_fully_kept=int((D == 0).sum()),
        chains_fully_dropped=int((K == 0).sum()),
        nucleic_contact_rate=float(np.nansum(
            np.array([c["nuc_all"] for c in per_chain]) * N) / N.sum()),
        per_chain=per_chain,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    print("\n=== STAGE 0a: CLEARANCE AUDIT ===")
    print(f"chains {len(per_chain)}  skipped {skipped}")
    print(f"residues total {N.sum()}  kept {K.sum()}  dropped {D.sum()}"
          f"  ({100*D.sum()/N.sum():.1f}% dropped)")
    print(f"chains fully kept {res['chains_fully_kept']}"
          f" / fully dropped {res['chains_fully_dropped']}")
    print(f"interface rate  all {pooled_all:.4f} |"
          f" KEPT {pooled_keep:.4f} | DROPPED {pooled_drop:.4f}")
    e = res["interface_rate"]["enrichment_drop_over_keep"]
    print(f"enrichment (dropped/kept) = {e:.3f}" if e else "enrichment undefined")
    print(f"nucleic-contact rate {res['nucleic_contact_rate']:.4f}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
