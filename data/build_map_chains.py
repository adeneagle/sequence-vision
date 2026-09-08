"""Per-map chain inventory -- the data build the voxel-level objective requires.

WHY THIS EXISTS. `data/alignment_chains.csv` carries **1.3 chains per map**, but
the maps contain a median of **12 chains / 9.5 distinct sequences** (measured
2026-08-31 over 150 random maps; max 56). `build_alignment_set.py` dedupes to one
row per (entry, sequence), shuffles, and then takes the first `--max-chains 2000`
of ~70k candidates -- so each map contributed ~1.3 randomly chosen chains and the
rest of its density is UNLABELLED in every existing target.

For a Ca-centred per-residue target that is mostly benign (the `up_blocks[1]`
receptive field is ~4-5 A, so a residue's feature is mostly about its own chain).
For a VOXEL-level objective it is fatal: a voxel sitting on chain B is not a
positive for chain A's sequence, not background, and not a valid same-map
negative. This module enumerates every chain so the target can be built over all
of them.

RESIDUES ARE INDEXED BY (sequence, position), NOT (chain, position). That is the
design decision that makes the voxel-anchored direction single-valued under
homo-oligomer symmetry: a voxel on copy 3 of a C12 ring has exactly one correct
target, position p of sequence s. Indexing by chain instance would make it
12-valued. The sequence tower embeds sequences, not chain instances, so this is
also the only indexing the loss can actually consume.

FILTERING IS COPIED FROM `backbone_with_resnum`, DELIBERATELY. Observed residues
are those with a complete N/CA/C backbone and a known amino acid, which is
exactly what `extract_esmc_chains.py` embeds -- so index i of an ESM-C embedding
is residue i of this inventory BY CONSTRUCTION, with no alignment step that can
drift. Any divergence in filtering would silently misalign the two modalities,
which is the failure mode that has cost this project the most time.

COORDINATES ARE ZYX AND IN ANGSTROMS, matching `backbone_with_resnum`. The voxel
grid conversion (`(xyz - origin_zyx) / MODEL_VOXEL_SIZE`) is deliberately NOT done
here: origin lives in the volume cache metadata, and baking it in would couple
this inventory to one particular preprocessing run.

Outputs:
  data/map_chains.csv            one row per (emd, chain)
  data/map_chains/<emd>.npz      atoms, residue table, sequences
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def observed_chains(pdb: str, min_len: int) -> dict:
    """{chain_name: (seq, ca_zyx [L,3], heavy_zyx [M,3], heavy_res [M])}.

    `heavy_res` indexes into this chain's own observed-residue list, so it is
    local to the chain; `main` remaps it to the per-map residue table.
    """
    import gemmi

    from probes.homolog_diagnostic_residue import AA3to1

    st = gemmi.read_structure(pdb)
    st.setup_entities()
    st.remove_alternative_conformations()

    out: dict = {}
    for model in st:
        for ch in model:
            seq, ca, heavy, heavy_res = [], [], [], []
            for res in ch:
                aN, aCA, aC = (res.find_atom(n, "*") for n in ("N", "CA", "C"))
                if aN is None or aCA is None or aC is None:
                    continue
                if res.name not in AA3to1:
                    continue
                i = len(seq)
                seq.append(AA3to1[res.name])
                ca.append([aCA.pos.z, aCA.pos.y, aCA.pos.x])
                for a in res:
                    if a.element.is_hydrogen:
                        continue
                    heavy.append([a.pos.z, a.pos.y, a.pos.x])
                    heavy_res.append(i)
            if len(seq) < min_len:
                continue
            out[ch.name] = (
                "".join(seq),
                np.asarray(ca, dtype=np.float32),
                np.asarray(heavy, dtype=np.float32),
                np.asarray(heavy_res, dtype=np.int32),
            )
        break  # model 0 only, as everywhere else in this project
    return out


def build_one(pdb: str, min_len: int) -> dict | None:
    """Per-map inventory keyed by (sequence, position)."""
    chains = observed_chains(pdb, min_len)
    if not chains:
        return None

    # Distinct sequences, in first-seen order so the mapping is deterministic.
    seqs: list[str] = []
    seq_idx: dict[str, int] = {}
    for name in sorted(chains):
        s = chains[name][0]
        if s not in seq_idx:
            seq_idx[s] = len(seqs)
            seqs.append(s)

    # Residue table: every position of every distinct sequence.
    res_seq, res_pos = [], []
    res_base = []           # first row of each sequence's block
    for si, s in enumerate(seqs):
        res_base.append(len(res_seq))
        res_seq.extend([si] * len(s))
        res_pos.extend(range(len(s)))

    atom_xyz, atom_res, atom_chain = [], [], []
    rows = []
    for ci, name in enumerate(sorted(chains)):
        s, ca, heavy, hres = chains[name]
        si = seq_idx[s]
        atom_xyz.append(heavy)
        # chain-local residue index -> per-map residue table row
        atom_res.append(hres.astype(np.int64) + res_base[si])
        atom_chain.append(np.full(len(hres), ci, dtype=np.int32))
        rows.append({
            "chain": name,
            "seq_idx": si,
            "n_obs": len(s),
            "n_heavy": int(len(hres)),
            "seq_sha1": hashlib.sha1(s.encode()).hexdigest()[:12],
        })

    return {
        "seqs": seqs,
        "rows": rows,
        "atom_xyz": np.concatenate(atom_xyz).astype(np.float32),
        "atom_res": np.concatenate(atom_res).astype(np.int32),
        "atom_chain": np.concatenate(atom_chain).astype(np.int32),
        "res_seq": np.asarray(res_seq, dtype=np.int32),
        "res_pos": np.asarray(res_pos, dtype=np.int32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/map_chains.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/map_chains"))
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    src = list(csv.DictReader(open(args.chains)))
    # One inventory per EMDB entry.
    #
    # SPLITS ARE PER-CHAIN IN THE CSV AND A MAP CAN SPAN SEVERAL. Measured on
    # this file: 64 entries host both train and test chains, 50 host train and
    # val. That is NOT a bug and D11 does not forbid it -- `split_map_lists`
    # resolves it by DROPPING shared maps from train (`clean_train = train - val
    # - test`), so the encoder never sees density that also contains val/test
    # residues. A first version of this script asserted one split per map and
    # died on EMD-0322; the assertion was wrong, not the data.
    #
    # Resolve to a map-level split by the same conservative precedence:
    # test > val > train. A map contributing any test chain is wholly test.
    entries: dict[str, dict] = {}
    for r in src:
        e = entries.setdefault(r["emd"], {"pdb": r["pdb"], "splits": set(),
                                          "clusters": set(), "listed": set()})
        e["splits"].add(r["split"])
        e["clusters"].add(r["cluster"])
        e["listed"].add(r["chain"])
    shared = 0
    for emd, e in entries.items():
        s = e["splits"]
        e["split"] = "test" if "test" in s else ("val" if "val" in s else "train")
        if len(s) > 1:
            shared += 1
    print(f"{shared}/{len(entries)} maps span >1 chain-level split; resolved by "
          f"precedence test>val>train (matches split_map_lists' clean_train)",
          flush=True)

    keys = sorted(entries, key=int)
    if args.limit:
        keys = keys[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(src)} listed chains -> {len(keys)} maps -> {args.out_dir}", flush=True)

    out_rows, done, skipped = [], 0, 0
    n_chain, n_seq = [], []
    t0 = time.time()
    for i, emd in enumerate(keys):
        e = entries[emd]
        npz = args.out_dir / f"{emd}.npz"
        try:
            inv = build_one(e["pdb"], args.min_len)
            if inv is None:
                raise ValueError(f"no chain >= {args.min_len} observed residues")
        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {emd} {type(exc).__name__}: {exc}", flush=True)
            continue

        tmp = npz.with_suffix(".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                atom_xyz=inv["atom_xyz"], atom_res=inv["atom_res"],
                atom_chain=inv["atom_chain"],
                res_seq=inv["res_seq"], res_pos=inv["res_pos"],
                seqs=np.array(inv["seqs"], dtype=object), allow_pickle=True,
            )
        tmp.replace(npz)

        split = e["split"]
        # multiplicity: how many chains carry each distinct sequence
        mult: dict[int, int] = {}
        for r in inv["rows"]:
            mult[r["seq_idx"]] = mult.get(r["seq_idx"], 0) + 1
        for r in inv["rows"]:
            out_rows.append({
                "emd": emd, "chain": r["chain"], "seq_idx": r["seq_idx"],
                "n_obs": r["n_obs"], "n_heavy": r["n_heavy"],
                "seq_sha1": r["seq_sha1"], "n_copies": mult[r["seq_idx"]],
                "split": split,
                "in_alignment_set": int(r["chain"] in e["listed"]),
                "key": f"{emd}_{r['chain']}",
            })
        n_chain.append(len(inv["rows"]))
        n_seq.append(len(inv["seqs"]))
        done += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(keys)}] {done} done, {skipped} skipped, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    nc, ns = np.array(n_chain), np.array(n_seq)
    print(f"\nmaps: {done} done, {skipped} skipped | {(time.time()-t0)/60:.1f} min")
    print(f"chains: {len(out_rows)} total, median {np.median(nc):.0f}/map, max {nc.max()}")
    print(f"distinct seqs: median {np.median(ns):.1f}/map, max {ns.max()}, "
          f"{100*np.mean(ns >= 2):.0f}% of maps have >=2")
    listed = sum(r["in_alignment_set"] for r in out_rows)
    print(f"of which already in alignment_chains.csv: {listed} "
          f"({100*listed/len(out_rows):.1f}%) -- the rest have no ESM-C yet")
    print(f"-> {args.out_csv}")

    if done == 0:
        raise SystemExit("FAILED: 0 maps inventoried.")
    if skipped > 0.5 * len(keys):
        raise SystemExit(f"FAILED: {skipped}/{len(keys)} maps skipped -- systematic.")


if __name__ == "__main__":
    main()
