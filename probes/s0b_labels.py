"""Stage 0b -- per-residue task labels for the CACHED chains, per chain INSTANCE.

Emits interface and ligand-pocket labels aligned row-for-row with the cached
latents in `data/o5_multitap_parts/` and `data/o7_alltap_parts/`, so the Stage 0c
additivity read needs no GPU.

Design decisions this file enforces (all from the Stage 0 plan):

* **Per chain INSTANCE, not per (sequence, position).** `data/map_chains/` keys
  residues by sequence, so labels there are OR'd over every symmetry copy --
  which inflates the positive rate and makes the label partly a function of
  oligomeric state. Labels here come from gemmi chain instances.
* **Both interface definitions.** heavy-atom 5 A (ours) AND 6 A Ca-Ca
  (ProteinShake's), without which the external reference points
  (raw-ESM MCC 0.236, best pyramid band 0.203) are not comparable.
* **homo vs hetero split**, plus `n_chains` as a protein-level covariate, so an
  assembly-level shortcut cannot masquerade as a per-residue result.
* **Two integrity checks, both hard failures.** The observed-Ca array must match
  `o7_coords.npz::<key>/ca_all` to 1e-2 A (ordering), and the amino-acid codes at
  the cached rows must match the cached `aa` exactly (row alignment).

Output: data/s0b_labels.npz + results/s0b_labels.json
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probes.s0a_clearance_audit import AA3, NUC3, chain_ca_zyx  # noqa: E402

AA1 = "ARNDCQEGHILKMFPSTWYV"
AA3to1 = dict(zip(
    ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"],
    "ARNDCQEGHILKMFPSTWYV"))
WATER = {"HOH", "WAT", "DOD", "H2O"}


def chain_seq(ch):
    return "".join(AA3to1[r.name] for r in ch if r.name in AA3)


def collect(st, chain_id, same_seq=None):
    """Target-chain per-residue heavy atoms + every partner group.

    `same_seq(chain_name) -> bool` decides homo vs hetero. Prefer the project's
    own `map_chains.csv::seq_idx`; observed-sequence string equality
    under-counts copies whose observed residues differ (same SEQRES, different
    gaps), which would silently push homo contacts into the hetero bucket.
    """
    model = st[0]
    tgt_pos, tgt_rid, tgt_ca = [], [], []
    tseq = None
    n_poly, seqs = 0, []

    for ch in model:
        s = chain_seq(ch)
        if len(s) >= 1:
            n_poly += 1
            seqs.append(s)
        if ch.name == chain_id:
            tseq = s

    # target chain, in the SAME filtered order as chain_ca_zyx
    k = 0
    for ch in model:
        if ch.name != chain_id:
            continue
        for res in ch:
            if res.name not in AA3:
                continue
            aN, aCA, aC = (res.find_atom(n, "*") for n in ("N", "CA", "C"))
            if aN is None or aCA is None or aC is None:
                continue
            tgt_ca.append([aCA.pos.z, aCA.pos.y, aCA.pos.x])
            for a in res:
                if a.element.atomic_number <= 1:
                    continue
                tgt_pos.append([a.pos.z, a.pos.y, a.pos.x])
                tgt_rid.append(k)
            k += 1

    groups = {"homo": [], "hetero": [], "nuc": [], "het": [], "het_nonion": []}
    ca_partner = []
    het_names = set()
    for ch in model:
        if ch.name == chain_id:
            continue
        s = chain_seq(ch)
        for res in ch:
            if res.name in WATER:
                continue
            heavy = [[a.pos.z, a.pos.y, a.pos.x] for a in res
                     if a.element.atomic_number > 1]
            if not heavy:
                continue
            if res.name in AA3:
                homo = (same_seq(ch.name) if same_seq is not None
                        else bool(s) and s == tseq)
                groups["homo" if homo else "hetero"].extend(heavy)
                aCA = res.find_atom("CA", "*")
                if aCA is not None:
                    ca_partner.append([aCA.pos.z, aCA.pos.y, aCA.pos.x])
            elif res.name in NUC3:
                groups["nuc"].extend(heavy)
            else:
                groups["het"].extend(heavy)
                het_names.add(res.name)
                if len(heavy) > 1:
                    groups["het_nonion"].extend(heavy)

    f = lambda L: np.asarray(L, float).reshape(-1, 3)
    return (f(tgt_pos), np.asarray(tgt_rid, int), f(tgt_ca),
            {k2: f(v) for k2, v in groups.items()}, f(ca_partner),
            n_poly, len(set(seqs)), sorted(het_names), tseq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="data/o7_coords.npz")
    ap.add_argument("--chains", default="data/alignment_chains.csv")
    ap.add_argument("--parts", default="data/o5_multitap_parts")
    ap.add_argument("--map-chains", default="data/map_chains.csv")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--ca-cutoff", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/s0b_labels.npz")
    ap.add_argument("--summary", default="results/s0b_labels.json")
    a = ap.parse_args()

    import gemmi
    from scipy.spatial import cKDTree

    co = np.load(a.coords, allow_pickle=True)
    keys = sorted(set(k.split("/")[0] for k in co.keys()))
    if a.limit:
        keys = keys[:a.limit]
    rows = {r["key"]: r for r in csv.DictReader(open(a.chains))}
    seqidx = {}
    if Path(a.map_chains).exists():
        for r in csv.DictReader(open(a.map_chains)):
            seqidx[(r["emd"], r["chain"])] = r["seq_idx"]
    n_fallback = 0

    out, per_chain = {}, []
    bad = {"no_row": 0, "ca_mismatch": 0, "aa_mismatch": 0, "error": 0}
    checked_aa = 0

    for ci, key in enumerate(keys):
        r = rows.get(key)
        if r is None:
            bad["no_row"] += 1
            continue
        try:
            st = gemmi.read_structure(r["pdb"])
            st.setup_entities()
            st.remove_alternative_conformations()

            _, ca_ref = chain_ca_zyx(st, r["chain"])
            cal = co[f"{key}/ca_all"]
            if len(ca_ref) != len(cal) or np.abs(ca_ref - cal).max() > 1e-2:
                bad["ca_mismatch"] += 1
                continue

            mine = seqidx.get((r["emd"], r["chain"]))
            if mine is None:
                n_fallback += 1
                same_seq = None
            else:
                same_seq = (lambda nm, e=r["emd"], m=mine:
                            seqidx.get((e, nm)) == m)
            tp, trid, tca, grp, capart, n_poly, n_uniq, hets, tseq = collect(
                st, r["chain"], same_seq)
            n = len(tca)

            def lab(part, cutoff=a.cutoff, src=None, sid=None):
                y = np.zeros(n, bool)
                src = tp if src is None else src
                sid = trid if sid is None else sid
                if len(part) == 0 or len(src) == 0:
                    return y
                hit = cKDTree(part).query_ball_point(
                    src, cutoff, return_length=True) > 0
                if hit.any():
                    y[np.unique(sid[hit])] = True
                return y

            prot = np.concatenate([grp["homo"], grp["hetero"]]) \
                if len(grp["homo"]) or len(grp["hetero"]) else np.zeros((0, 3))
            L = {
                "iface5_heavy": lab(prot),
                "iface_homo": lab(grp["homo"]),
                "iface_hetero": lab(grp["hetero"]),
                "iface_nuc": lab(grp["nuc"]),
                "lig5_any": lab(grp["het"]),
                "lig5_nonion": lab(grp["het_nonion"]),
                "iface6_ca": lab(capart, a.ca_cutoff,
                                 src=tca, sid=np.arange(n)),
            }

            idx = co[f"{key}/idx"].astype(int)
            aa_mine = np.array([AA1.index(c) if c in AA1 else -1
                                for c in tseq])
            p = Path(a.parts) / f"{key}.npz"
            if p.exists():
                pa = np.load(p, allow_pickle=True)["aa"]
                if len(pa) != len(idx) or not np.array_equal(pa, aa_mine[idx]):
                    bad["aa_mismatch"] += 1
                    continue
                checked_aa += 1

            for k2, v in L.items():
                out[f"{key}/{k2}"] = v[idx]
                out[f"{key}/{k2}_all"] = v
            out[f"{key}/idx"] = idx
            out[f"{key}/meta"] = np.array(
                [n_poly, n_uniq, n, len(idx)], dtype=np.int32)

            per_chain.append(dict(
                key=key, emd=r["emd"], chain=r["chain"], split=r["split"],
                cluster=r["cluster"], n_obs=int(n), n_cached=int(len(idx)),
                n_chains=int(n_poly), n_unique_seq=int(n_uniq),
                het=hets,
                **{f"rate_{k2}": float(v[idx].mean()) for k2, v in L.items()}))
        except Exception as e:  # noqa: BLE001
            bad["error"] += 1
            if bad["error"] <= 3:
                print(f"  ! {key}: {type(e).__name__}: {e}", flush=True)
        if (ci + 1) % 250 == 0:
            print(f"  {ci+1}/{len(keys)} ({len(per_chain)} ok)", flush=True)

    if not per_chain:
        raise SystemExit("no chains labelled")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)

    def pooled(name):
        w = np.array([c["n_cached"] for c in per_chain], float)
        v = np.array([c[f"rate_{name}"] for c in per_chain], float)
        return float((v * w).sum() / w.sum())

    names = ["iface5_heavy", "iface6_ca", "iface_homo", "iface_hetero",
             "iface_nuc", "lig5_any", "lig5_nonion"]
    nch = np.array([c["n_chains"] for c in per_chain])
    summary = dict(
        _meta=dict(chains=len(per_chain), integrity=bad,
                   homo_calls_by_seqidx=len(per_chain) - n_fallback,
                   homo_calls_fallback=n_fallback,
                   aa_rows_verified=checked_aa,
                   cutoff_A=a.cutoff, ca_cutoff_A=a.ca_cutoff),
        residues_cached=int(sum(c["n_cached"] for c in per_chain)),
        pooled_positive_rate={n2: pooled(n2) for n2 in names},
        chains_with_any={n2: int(sum(1 for c in per_chain
                                     if c[f"rate_{n2}"] > 0))
                         for n2 in names},
        n_chains_per_map=dict(median=float(np.median(nch)),
                              mean=float(nch.mean()), max=int(nch.max()),
                              monomer_fraction=float((nch == 1).mean())),
        splits={s: int(sum(1 for c in per_chain if c["split"] == s))
                for s in sorted(set(c["split"] for c in per_chain))},
        per_chain=per_chain,
    )
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summary, indent=2))

    print("\n=== STAGE 0b: LABELS ===")
    print(f"chains {len(per_chain)}  integrity {bad}"
          f"  aa-verified {checked_aa}")
    print(f"cached residues {summary['residues_cached']}")
    for n2 in names:
        print(f"  {n2:14s} rate {pooled(n2):.4f}"
              f"   chains with any {summary['chains_with_any'][n2]}")
    print(f"n_chains/map median {np.median(nch):.0f}"
          f"  monomers {100*(nch==1).mean():.1f}%")
    print(f"\n-> {a.out}\n-> {a.summary}")


if __name__ == "__main__":
    main()
