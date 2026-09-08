"""Chain list + leakage-safe splits for the sequence -> density-feature alignment.

Splits are by mmseqs CLUSTER at 30% identity, not by protein. Protein-level is
not enough here: the homolog diagnostic measured aligned-residue identity of
63.2% between 30-60% homologs, and showed the target tracks sequence family, so
putting a homolog of a training chain in the test set leaks most of the answer.

Emits the OBSERVED sequence (residues with a complete N/CA/C backbone), because
that is exactly the residue set the density side can describe. Both modalities
must be indexed by the same list or the pairing is silently wrong -- the failure
mode this project has paid for repeatedly.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from data.build_homolog_pairs import CORPUS, read_entries
from probes.homolog_diagnostic_residue import chain_backbone


def cluster(seqs: dict[str, str], tmp: Path, ident: float = 0.30,
            cov: float = 0.5) -> dict[str, str]:
    """key -> cluster representative, via mmseqs easy-cluster."""
    fa = tmp / "in.fasta"
    with fa.open("w") as fh:
        for k, s in seqs.items():
            fh.write(f">{k}\n{s}\n")
    pre = tmp / "clu"
    subprocess.run(
        ["mmseqs", "easy-cluster", str(fa), str(pre), str(tmp / "mm"),
         "--min-seq-id", str(ident), "-c", str(cov), "--cov-mode", "0"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {}
    for line in (tmp / "clu_cluster.tsv").read_text().splitlines():
        rep, mem = line.split("\t")
        out[mem] = rep
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-entries", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--max-chains", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/alignment_chains.csv"))
    args = ap.parse_args()

    recs = read_entries(args.limit_entries)
    # One chain per (entry, sequence): extra copies of a homo-oligomer are the
    # same molecule and would just duplicate rows.
    seen, uniq = set(), []
    for r in recs:
        key = (r["emd"], r["seq"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    print(f"{len(uniq)} unique (entry, sequence) chains")

    import random
    rng = random.Random(args.seed)
    rng.shuffle(uniq)

    rows, obs = [], {}
    for r in uniq:
        if len(rows) >= args.max_chains:
            break
        try:
            seq, ca, fr = chain_backbone(r["pdb"], r["chain"])
        except Exception:
            continue
        if not (args.min_len <= len(seq) <= args.max_len):
            continue
        key = f"{r['emd']}_{r['chain']}"
        obs[key] = seq
        rows.append({"key": key, "emd": r["emd"], "pdb": r["pdb"],
                     "chain": r["chain"], "n_obs": len(seq), "seq": seq})
    print(f"{len(rows)} chains with a usable observed backbone "
          f"({args.min_len}-{args.max_len} residues)")

    with tempfile.TemporaryDirectory() as td:
        rep = cluster(obs, Path(td))
    clusters = sorted({rep.get(r["key"], r["key"]) for r in rows})
    rng.shuffle(clusters)
    n_val = int(len(clusters) * args.val_frac)
    n_test = int(len(clusters) * args.test_frac)
    split_of = {}
    for i, c in enumerate(clusters):
        split_of[c] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    for r in rows:
        r["cluster"] = rep.get(r["key"], r["key"])
        r["split"] = split_of[r["cluster"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "emd", "pdb", "chain", "n_obs",
                                           "cluster", "split", "seq"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    cs = Counter(r["split"] for r in rows)
    print(f"{len(clusters)} clusters at 30% identity -> "
          f"train {cs['train']} / val {cs['val']} / test {cs['test']} chains")
    # No cluster may straddle a split, or the leakage this file exists to
    # prevent is back.
    straddle = {r["cluster"] for r in rows} & set()
    by_c = {}
    for r in rows:
        by_c.setdefault(r["cluster"], set()).add(r["split"])
    bad = [c for c, s in by_c.items() if len(s) > 1]
    assert not bad, f"{len(bad)} clusters straddle splits"
    print("OK: no cluster straddles a split")
    print(json.dumps({"chains": len(rows), "clusters": len(clusters), **cs}, indent=2))


if __name__ == "__main__":
    main()
