"""Build homolog and length-matched unrelated chain pairs from Cryo2StructData.

Input for the homolog diagnostic: does a density descriptor vary *smoothly* with
sequence? That needs three comparison levels --

    same structure, two orientations   -> pose noise (the nuisance floor)
    homologs at 30-60% identity        -> does the feature track sequence?
    unrelated, LENGTH-MATCHED          -> the full spread

The length matching is not optional. Homologs have similar lengths and unrelated
proteins do not, so an unmatched "unrelated" pool makes size alone look like
homology signal -- the same artifact that inflated this project's contact probe
to AUC 0.997.

Reads the corpus in place (1.4 TB, world-readable, do NOT copy):
    /mnt/main0/projects/hypernetworks-for-cryo-em/cryo2structdata-3A/full/<NNNN>/
Each entry holds `<pdbid>.fasta` (PDB-style, one record per ENTITY with a
"Chains A, J[auth L]" header) and `<pdbid>.pdb` (the fitted model).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

CORPUS = Path("/mnt/main0/projects/hypernetworks-for-cryo-em/cryo2structdata-3A/full")

# ">6GIQ_1|Chains A, J[auth L]|desc|organism"  or  ">6GH5_2|Chain C|..."
_CHAINS = re.compile(r"Chains?\s+(.+)", re.I)


def _chain_ids(header: str) -> list[str]:
    """Chain IDs from a PDB-style FASTA header.

    Prefers the author chain ID in `J[auth L]` -- that is what appears in the
    coordinate file, so it is the one that can be matched to atoms.
    """
    parts = header.split("|")
    if len(parts) < 2:
        return []
    m = _CHAINS.match(parts[1].strip())
    if not m:
        return []
    out = []
    for tok in m.group(1).split(","):
        tok = tok.strip()
        auth = re.search(r"\[auth\s+([^\]]+)\]", tok)
        out.append(auth.group(1).strip() if auth else tok.split("[")[0].strip())
    return [c for c in out if c]


def read_entries(limit: int | None = None) -> list[dict]:
    """One record per (entry, chain): id, pdb path, chain, sequence."""
    recs: list[dict] = []
    dirs = sorted(p for p in CORPUS.iterdir() if p.is_dir())
    if limit:
        dirs = dirs[:limit]
    for d in dirs:
        fastas = list(d.glob("*.fasta"))
        pdbs = list(d.glob("*.pdb"))
        # coil/helix/strand.pdb are SS decompositions, not the fitted model
        pdbs = [p for p in pdbs if p.stem not in {"coil", "helix", "strand"}]
        if not fastas or not pdbs:
            continue
        header, seq = None, []
        for line in fastas[0].read_text().splitlines():
            if line.startswith(">"):
                if header and seq:
                    for c in _chain_ids(header):
                        recs.append({"emd": d.name, "pdb": str(pdbs[0]),
                                     "chain": c, "seq": "".join(seq)})
                header, seq = line[1:], []
            elif line.strip():
                seq.append(line.strip())
        if header and seq:
            for c in _chain_ids(header):
                recs.append({"emd": d.name, "pdb": str(pdbs[0]),
                             "chain": c, "seq": "".join(seq)})
    return recs


def mmseqs_pairs(recs: list[dict], tmp: Path, min_cov: float = 0.7) -> list[tuple]:
    """All-vs-all identity. Returns (i, j, fident, alnlen) with i < j."""
    fa = tmp / "all.fasta"
    with fa.open("w") as fh:
        for k, r in enumerate(recs):
            fh.write(f">{k}\n{r['seq']}\n")
    res = tmp / "hits.tsv"
    subprocess.run(
        ["mmseqs", "easy-search", str(fa), str(fa), str(res), str(tmp / "mm"),
         "-s", "7.5", "--max-seqs", "4000", "-e", "1e-3",
         "-c", str(min_cov), "--cov-mode", "0",
         "--format-output", "query,target,fident,alnlen"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = []
    for line in res.read_text().splitlines():
        q, t, fid, aln = line.split("\t")
        qi, ti = int(q), int(t)
        if qi < ti:
            out.append((qi, ti, float(fid), int(aln)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-entries", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=600,
                    help="cap simulation cost; large complexes blow up the box")
    ap.add_argument("--hom-lo", type=float, default=0.30)
    ap.add_argument("--hom-hi", type=float, default=0.60)
    ap.add_argument("--unrel-hi", type=float, default=0.20)
    ap.add_argument("--len-tol", type=float, default=0.10,
                    help="unrelated pairs must match a homolog pair's mean "
                         "length to within this fraction")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/homolog_pairs.csv"))
    args = ap.parse_args()

    recs = [r for r in read_entries(args.limit_entries)
            if args.min_len <= len(r["seq"]) <= args.max_len]
    # One chain per (entry, sequence): identical chains of a homo-oligomer are
    # the same molecule, and pairing them would measure nothing.
    seen, uniq = set(), []
    for r in recs:
        key = (r["emd"], r["seq"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    print(f"{len(uniq)} unique (entry, sequence) chains "
          f"from {len({r['emd'] for r in uniq})} entries")

    with tempfile.TemporaryDirectory() as td:
        hits = mmseqs_pairs(uniq, Path(td))
    print(f"{len(hits)} aligned pairs above coverage threshold")

    rng = random.Random(args.seed)
    # Homologs must be DIFFERENT entries -- the same entry at 100% identity is
    # the pose comparison, not the homolog comparison.
    hom = [(i, j, f) for i, j, f, _ in hits
           if args.hom_lo <= f <= args.hom_hi and uniq[i]["emd"] != uniq[j]["emd"]]
    rng.shuffle(hom)
    hom = hom[: args.n_pairs]
    print(f"{len(hom)} homolog pairs in [{args.hom_lo}, {args.hom_hi}]")

    # Anything that aligned at all is excluded from the unrelated pool, however
    # weakly -- a 25% hit is not "unrelated", it is a weak homolog.
    related = {(i, j) for i, j, _, _ in hits}
    by_len: dict[int, list[int]] = {}
    for k, r in enumerate(uniq):
        by_len.setdefault(len(r["seq"]), []).append(k)
    lengths = sorted(by_len)

    unrel = []
    for i, j, _ in hom:
        target = 0.5 * (len(uniq[i]["seq"]) + len(uniq[j]["seq"]))
        lo, hi = target * (1 - args.len_tol), target * (1 + args.len_tol)
        cand = [k for L in lengths if lo <= L <= hi for k in by_len[L]]
        rng.shuffle(cand)
        for a in cand:
            for b in cand:
                if a >= b or uniq[a]["emd"] == uniq[b]["emd"]:
                    continue
                if (a, b) in related or (b, a) in related:
                    continue
                unrel.append((a, b, 0.0))
                break
            else:
                continue
            break
    print(f"{len(unrel)} length-matched unrelated pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "emd_a", "pdb_a", "chain_a", "len_a",
                    "emd_b", "pdb_b", "chain_b", "len_b", "fident"])
        for kind, pairs in (("homolog", hom), ("unrelated", unrel)):
            for i, j, f in pairs:
                a, b = uniq[i], uniq[j]
                w.writerow([kind, a["emd"], a["pdb"], a["chain"], len(a["seq"]),
                            b["emd"], b["pdb"], b["chain"], len(b["seq"]), f"{f:.3f}"])

    ha = [0.5 * (len(uniq[i]["seq"]) + len(uniq[j]["seq"])) for i, j, _ in hom]
    ua = [0.5 * (len(uniq[i]["seq"]) + len(uniq[j]["seq"])) for i, j, _ in unrel]
    summary = {
        "n_chains": len(uniq),
        "n_homolog": len(hom),
        "n_unrelated": len(unrel),
        "mean_len_homolog": sum(ha) / max(len(ha), 1),
        "mean_len_unrelated": sum(ua) / max(len(ua), 1),
    }
    print(json.dumps(summary, indent=2))
    # If these means diverge the matching failed and every downstream number is
    # confounded by size.
    if ha and ua:
        rel = abs(summary["mean_len_homolog"] - summary["mean_len_unrelated"]) / \
            summary["mean_len_homolog"]
        print(f"length-match residual: {rel:.1%} (want << {args.len_tol:.0%})")


if __name__ == "__main__":
    main()
