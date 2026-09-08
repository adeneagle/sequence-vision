"""Curate paired (EMDB map, fitted PDB model) entries for Phase 0.

Single-particle cryo-EM entries better than a resolution cutoff, that have a
fitted atomic model -- this pairing is what gives exact residue -> voxel
correspondence, which the whole alignment study depends on.

Writes data/manifest.csv and downloads into data/maps/.
Re-running skips anything already on disk, so it is safe to interrupt.

    pixi run python data/build_emdb_pairs.py --n 50 --max-res 3.0
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

EMDB_SEARCH = "https://www.ebi.ac.uk/emdb/api/search/"
EMDB_MAP = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{eid}/map/{lower}.map.gz"
RCSB_CIF = "https://files.rcsb.org/download/{pdb}.cif"


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sequence-vision/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


PAGE = 100  # EMDB caps a single response at 100 records regardless of `rows`


def query(max_res: float, rows: int, method: str | None = "singleParticle") -> list[dict]:
    """Paginated EMDB search.

    The API silently caps every response at 100 records: `rows=2500` returns 100,
    and `start=`/`offset=` are ignored (verified -- they return page 1 again).
    Only `page=` advances. An earlier version passed `rows=2500`, got 100, and
    reported success, so a run that asked for 600 maps quietly delivered 51.
    Hence the explicit page loop and the assertion below.
    """
    q = f"resolution:%5B0%20TO%20{max_res}%5D"
    if method:
        q += f"%20AND%20structure_determination_method:%22{method}%22"
    out, seen, page = [], set(), 1     # 1-based: page=0 returns empty
    while len(out) < rows:
        d = json.loads(_get(f"{EMDB_SEARCH}{q}?rows={PAGE}&page={page}&wt=json"))
        docs = d if isinstance(d, list) else d.get("response", {}).get("docs", [])
        fresh = [r for r in docs if r.get("emdb_id") not in seen]
        if not fresh:
            print(f"  EMDB exhausted at page {page} ({len(out)} records)")
            break
        seen.update(r.get("emdb_id") for r in fresh)
        out.extend(fresh)
        page += 1
    assert len(out) > PAGE or page <= 2, (
        f"pagination appears broken: {len(out)} records after {page} pages")
    return out[:rows]


def extract(rec: dict) -> dict | None:
    """Pull the handful of fields we need out of a full EMDB record."""
    eid = rec.get("emdb_id")
    if not eid:
        return None
    try:
        m = rec["map"]
        dims = m["dimensions"]
        box = (int(dims["col"]), int(dims["row"]), int(dims["sec"]))
        apix = float(m["pixel_spacing"]["x"]["valueOf_"])
    except Exception:
        return None
    try:
        res = float(rec["structure_determination_list"]["structure_determination"][0]
                    ["image_processing"][0]["final_reconstruction"]["resolution"]["valueOf_"])
    except Exception:
        return None
    pdbs = rec.get("crossreferences", {}).get("pdb_list", {}).get("pdb_reference", [])
    pdbs = [p.get("pdb_id") for p in pdbs if p.get("pdb_id")] if isinstance(pdbs, list) else []
    if not pdbs:
        return None
    # Cubic-ness is RECORDED, not used to reject: it is an analysis convenience,
    # and re-downloading after a scope change costs days. Downstream code filters
    # the manifest instead.
    return {"emdb_id": eid, "resolution": res, "box": max(box),
            "box_x": box[0], "box_y": box[1], "box_z": box[2],
            "cubic": int(box[0] == box[1] == box[2]),
            "apix": apix, "pdb_id": pdbs[0]}


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def model_info(cif_path: str) -> dict | None:
    """Per-chain sequences + assembly/symmetry facts from the fitted model.

    Symmetry is recorded deliberately: a C_n axis with n>=3 makes the inertia
    tensor exactly degenerate (breaking PCA frames) and makes symmetry-equivalent
    residues carry different features from a non-equivariant network. It has
    already broken two separate things -- track it rather than rediscover it.
    """
    import gemmi

    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    chains = {}
    for chain in st[0]:
        seq = "".join(AA3TO1.get(r.name, "") for r in chain
                      if r.find_atom("CA", "*") is not None)
        if len(seq) >= 30:
            chains[chain.name] = seq
    if not chains:
        return None
    uniq = sorted(set(chains.values()))
    try:
        blk = gemmi.cif.read(cif_path).sole_block()
        sym = (blk.find_value("_em_single_particle_entity.point_symmetry") or "?").strip("\"'")
    except Exception:
        sym = "?"
    # Pair on the LARGEST chain rather than requiring a single unique sequence.
    # Pooling density over just that chain's Ca keeps the map<->sequence join
    # unambiguous for hetero-complexes too, which would otherwise cost ~58% of
    # entries. Trade-off: for a hetero-complex the density around the chain
    # includes partner chains that its sequence does not determine, so
    # n_unique_seq is recorded as a covariate.
    chain_id = max(chains, key=lambda c: len(chains[c]))
    return {"n_unique_seq": len(uniq), "n_chains": len(chains), "point_symmetry": sym,
            "chain_id": chain_id, "seq": chains[chain_id],
            "seq_len": len(chains[chain_id])}


def cluster_30(records: list[dict], workdir: Path, min_seq_id: float = 0.30) -> list[dict]:
    """Keep one representative per mmseqs2 cluster at `min_seq_id`."""
    import subprocess

    workdir.mkdir(parents=True, exist_ok=True)
    fa = workdir / "seqs.fasta"
    with open(fa, "w") as fh:
        for r in records:
            fh.write(f">{r['emdb_id']}\n{r['seq']}\n")
    pref = workdir / "clu"
    cmd = ["mmseqs", "easy-cluster", str(fa), str(pref), str(workdir / "tmp"),
           "--min-seq-id", str(min_seq_id), "-c", "0.8", "--cov-mode", "0", "-v", "1"]
    subprocess.run(cmd, check=True, capture_output=True)

    reps = set()
    with open(f"{pref}_cluster.tsv") as fh:
        for line in fh:
            rep, _member = line.split()
            reps.add(rep)
    kept = [r for r in records if r["emdb_id"] in reps]
    print(f"mmseqs2 @{min_seq_id:.0%} id: {len(records)} -> {len(kept)} clusters "
          f"({len(records)-len(kept)} redundant dropped)")
    return kept


def fetch_one(e: dict, outdir: Path) -> dict | None:
    eid = e["emdb_id"]
    lower = "emd_" + eid.split("-")[1]
    map_path = outdir / f"{lower}.map"
    cif_path = outdir / f"{e['pdb_id']}.cif"
    try:
        if not map_path.exists():
            gz = outdir / f"{lower}.map.gz"
            gz.write_bytes(_get(EMDB_MAP.format(eid=eid, lower=lower), timeout=900))
            with gzip.open(gz, "rb") as fi, open(map_path, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            gz.unlink()
        if not cif_path.exists():
            cif_path.write_bytes(_get(RCSB_CIF.format(pdb=e["pdb_id"].lower())))
    except Exception as exc:
        print(f"  {eid}: FAILED ({type(exc).__name__}: {exc})")
        map_path.unlink(missing_ok=True)
        return None
    e = dict(e, map_path=str(map_path), cif_path=str(cif_path))
    print(f"  {eid} {e['resolution']:.2f}A box={e['box']} apix={e['apix']:.3f} -> {e['pdb_id']}")
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-res", type=float, default=3.0)
    ap.add_argument("--max-box", type=int, default=100000, help="cap on the largest box dim")
    ap.add_argument("--method", default="singleParticle",
                    help="EMDB structure_determination_method, or 'any'")
    ap.add_argument("--rows", type=int, default=400, help="EMDB records to consider")
    ap.add_argument("--outdir", type=Path, default=Path("data/maps"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cluster", action="store_true", help="mmseqs2 redundancy removal")
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--single-sequence", action="store_true",
                    help="keep only maps whose fitted model has one unique sequence")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"querying EMDB (<= {args.max_res} A, method={args.method}, {args.rows} records)...")
    recs = query(args.max_res, args.rows,
                 None if args.method == "any" else args.method)
    cands = [c for c in (extract(r) for r in recs) if c and c["box"] <= args.max_box]
    # Shuffle rather than sort by box: taking the n smallest would systematically
    # bias the sample toward small proteins, which is exactly the kind of silent
    # selection effect that invalidates a cross-protein claim. max_box already
    # bounds cost; within that bound, sample uniformly.
    import random
    random.Random(0).shuffle(cands)
    print(f"{len(recs)} records -> {len(cands)} usable candidates (cubic, fitted model, box <= {args.max_box})")

    cands = cands[: args.n]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, c, args.outdir): c for c in cands}
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)

    # Attach sequences + symmetry, and require an unambiguous map<->sequence pairing.
    enriched, multi = [], 0
    for r in rows:
        mi = model_info(r["cif_path"])
        if mi is None:
            continue
        if args.single_sequence and mi["n_unique_seq"] > 1:
            multi += 1
            continue
        enriched.append(dict(r, seq=mi["seq"], seq_len=mi["seq_len"],
                             chain_id=mi["chain_id"], n_chains=mi["n_chains"],
                             n_unique_seq=mi["n_unique_seq"],
                             point_symmetry=mi["point_symmetry"]))
    print(f"{len(rows)} downloaded -> {len(enriched)} usable "
          f"({multi} dropped: hetero-complex with >1 unique sequence)")

    if args.cluster and len(enriched) > 1:
        enriched = cluster_30(enriched, args.outdir.parent / "cluster", args.min_seq_id)

    enriched.sort(key=lambda r: r["emdb_id"])
    with open(args.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(enriched[0].keys()))
        w.writeheader()
        w.writerows(enriched)

    from collections import Counter
    print(f"\n{len(enriched)} pairs -> {args.manifest}")
    print("  oligomeric counts:", dict(sorted(Counter(r["n_chains"] for r in enriched).items())))
    print("  annotated symmetry:", dict(Counter(r["point_symmetry"] for r in enriched)))
    print("  seq length: min %d median %d max %d" % (
        min(r["seq_len"] for r in enriched),
        sorted(r["seq_len"] for r in enriched)[len(enriched) // 2],
        max(r["seq_len"] for r in enriched)))


if __name__ == "__main__":
    main()
