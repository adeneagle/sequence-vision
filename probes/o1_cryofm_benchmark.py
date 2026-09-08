"""O1: are CryoFM2 features actually USEFUL? A density-side representation benchmark.

The question `PLAN.md` promised and never asked. It said "Teacher choice is open;
CryoFM is the starting candidate but gets validated before anything is built on
it." What got validated was STABILITY (pose invariance 0.83, random-weight
control 4x). USEFULNESS was never validated -- and per CLAUDE.md, "No feature
API, no representation benchmark in either paper -- whether the activations are
usable at all is the open question." We built an alignment target on a teacher
never shown to be good for anything.

No sequence side here, so this cannot inherit the degeneracy that sank the
alignment objective (R2 against a chosen feature configuration is maximised by
choosing a LESS informative target). The metric is external: classification
accuracy on labels we did not invent.

TASKS (labels shipped with Cryo2StructData / derivable from the fitted model):
  aa   per-residue amino-acid identity, 20-class. External reference point:
       ModelAngelo ~49% top-1 at 4-5 A (our corpus is <=3 A, where its bar is
       higher). This is the canonical "can you read the sequence off a map" task.
  ss   per-residue secondary structure, 3-class, from the shipped
       helix/strand/coil.pdb residue membership.

ARMS -- the controls are the point, not the CryoFM number:
  cryofm   pretrained CryoFM2, `up_blocks[0]` centre feature in the residue's
           N-CA-C frame (this project's best per-residue construction: pose
           invariance 0.826).
  raw      the central 8^3 = 512 RAW voxels of the identical frame-aligned box.
           Same dimensionality as up_blocks[0] (512), same framing, same
           information locality, no network. **If CryoFM cannot beat raw voxels
           its features are worthless.** This project is a graveyard of
           representations that lost to their own raw input (raw ESM beat every
           learned block and every closed-form band).
  random   identical architecture, no checkpoint. Separates "the trained network
           encodes this" from "any 3D conv over density does". It passed 4x for
           stability, which does NOT transfer to classification.
  prior    majority-class rate.

REAL experimental maps, not simulated. Simulated density is a deterministic
function of the atomic model, so a classifier on it partly re-reads its own
labels; and real maps are the deployment domain. Caveat inherited from the
experimental arm of the homolog diagnostic: the raw map contains EVERY chain, so
a residue's box includes neighbouring chains -- realistic, and identical across
all three arms, so it cannot favour one.

Splits are the EXISTING cluster-level assignments in data/alignment_chains.csv
(672 mmseqs clusters at 30% identity, no cluster straddles a split), so results
are directly comparable to everything else in the project.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

AA1 = "ARNDCQEGHILKMFPSTWYV"
SS3 = ("helix", "strand", "coil")


def to_cubic_even(vol: np.ndarray) -> np.ndarray:
    """Pad to a cubic, even-sided box -- CryoFM's Fourier resampler requires it.

    `normalize_voxel_size_fourier` asserts `iz % 2 == 0` and
    `(ix-1)*2 == iy == iz` on the rfft shape, i.e. a cube of even side. 6.7% of
    Cryo2StructData entries are non-cubic or odd (measured over 120 entries), and
    the earlier experimental arm simply skipped them. Skipping is not neutral:
    non-cubic boxes plausibly track elongated/filamentous particles, so dropping
    them biases the sample. Padding is applied at the HIGH end of each axis only,
    so voxel 0 -- and therefore the origin used for coordinate conversion -- is
    unchanged, and the pad value is the map MEDIAN (solvent background) rather
    than zero, per the logged lesson that a zero pad injects an above-background
    slab once normalisation is applied.
    """
    n = max(vol.shape)
    if n % 2:
        n += 1
    if tuple(vol.shape) == (n, n, n):
        return vol
    return np.pad(vol, [(0, n - s) for s in vol.shape],
                  mode="constant", constant_values=float(np.median(vol)))


def ss_labels(entry_dir: Path) -> dict[tuple[str, int], int]:
    """(chain, resseq) -> 0/1/2 from the shipped helix/strand/coil PDBs."""
    lab: dict[tuple[str, int], int] = {}
    for cls, name in enumerate(SS3):
        p = entry_dir / f"{name}.pdb"
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    try:
                        lab[(line[21], int(line[22:26]))] = cls
                    except ValueError:
                        continue
    return lab


def backbone_with_resnum(pdb: str, chain_id: str):
    """(seq, ca_zyx, frames, resnums) with EXACTLY chain_backbone's filtering.

    Replicated rather than imported because chain_backbone does not return
    residue numbers, which the SS lookup needs. The caller asserts the returned
    `seq` equals chain_backbone's, so any divergence in filtering is caught
    rather than silently misaligning labels against features.
    """
    import gemmi
    from probes.homolog_diagnostic_residue import AA3to1

    st = gemmi.read_structure(pdb)
    st.setup_entities()
    st.remove_alternative_conformations()
    seq, ca, frames, nums = [], [], [], []
    for model in st:
        for ch in model:
            if ch.name != chain_id:
                continue
            for res in ch:
                aN, aCA, aC = (res.find_atom(n, "*") for n in ("N", "CA", "C"))
                if aN is None or aCA is None or aC is None:
                    continue
                if res.name not in AA3to1:
                    continue
                p = lambda a: np.array([a.pos.z, a.pos.y, a.pos.x], dtype=np.float64)
                c = p(aCA)
                v1, v2 = p(aN) - c, p(aC) - c
                e1 = v1 / max(np.linalg.norm(v1), 1e-8)
                v2 = v2 - (v2 @ e1) * e1
                e2 = v2 / max(np.linalg.norm(v2), 1e-8)
                ca.append(c)
                frames.append(np.stack([e1, e2, np.cross(e1, e2)]))
                seq.append(AA3to1[res.name])
                nums.append(res.seqid.num)
        break
    return "".join(seq), np.array(ca), np.array(frames), np.array(nums)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--ckpt", default="weights/cryofm-v2/cryofm2-pretrain")
    ap.add_argument("--tap", default="up_blocks[0]")
    ap.add_argument("--timestep", type=int, default=10)
    ap.add_argument("--settings", default=None,
                    help="comma list of t[:c|d] operating points extracted in ONE "
                         "pass, e.g. '10:d,100:d,261:d,500:d,100:c,261:c'. "
                         "d=DECOUPLED (clean input, told t; off-manifold, what a "
                         "CleanDIFT student does), c=COUPLED (x_t=(1-t/1000)x_0 + "
                         "(t/1000)eps; the trained pairing). Map loading and box "
                         "cutting dominate cost, so extra settings only multiply "
                         "the forward over boxes already on the GPU. Every prior "
                         "number in this project is t=10 DECOUPLED -- one point of "
                         "a 2-D space that moves features as much as rotation does "
                         "(noise_shift cos 0.31-0.55), and t=10 sits at the "
                         "near-clean end while DIFT/CleanDIFT put the semantic "
                         "optimum near t=261.")
    ap.add_argument("--per-chain", type=int, default=50,
                    help="residues sampled per chain (cost control; 1500 chains "
                         "x 50 = 75k samples, ample for these probes)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--box-chunk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache", type=Path, default=Path("data/o1_features.npz"))
    ap.add_argument("--feat-dir", type=Path, default=None,
                    help="per-chain feature cache. This partition preempts hard "
                         "(the up_blocks[0] run logged Restarts=22, and a whole "
                         "5-job sweep died at once because Slurm packed it onto "
                         "one node), and extraction is ~2-3 h. Per-chain files "
                         "make a restart cost ONE chain instead of everything.")
    ap.add_argument("--out", type=Path, default=Path("results/o1_cryofm_benchmark.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.settings:
        SETTINGS = []
        for tok in args.settings.split(","):
            tt, _, mode = tok.strip().partition(":")
            SETTINGS.append((int(tt), (mode or "d").lower()))
    else:
        SETTINGS = [(args.timestep, "d")]
    # Keep the legacy key when no --settings is given: the five tap-sweep jobs
    # already on the cluster wrote parts under "cryofm", and a requeue must not
    # find a renamed key and fail to aggregate.
    names = ([f"cryofm_t{t}{m}" for t, m in SETTINGS] if args.settings
             else ["cryofm"])
    print("operating points: " + ", ".join(names))

    if args.cache.exists():
        print(f"loading cached features {args.cache}", flush=True)
        z = np.load(args.cache, allow_pickle=True)
        F = {k: z[k] for k in z.files}
    else:
        from teachers.cryofm_tap import MODEL_VOXEL_SIZE, PATCH, CryoFM2Tap, preprocess
        from probes.local_frame_stability import centre_features, extract_local_boxes
        from probes.homolog_diagnostic_residue import chain_backbone
        from probes.stability import load_map

        rows = list(csv.DictReader(open(args.chains)))
        if args.limit:
            rows = rows[: args.limit]
        rows.sort(key=lambda r: r["emd"])          # group chains of one entry
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fdir = args.feat_dir or Path(
            f"data/o1_feat_{args.tap.replace('[','').replace(']','')}_parts")
        fdir.mkdir(parents=True, exist_ok=True)
        print(f"{len(rows)} chains | tap={args.tap} | t={args.timestep} | {dev}\n"
              f"per-chain cache: {fdir}", flush=True)

        # Built on first use: a requeued job whose parts are all present must not
        # pay for two 168 M-param model loads just to aggregate.
        _taps: dict[str, object] = {}

        def taps():
            if not _taps:
                _taps["real"] = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                                           batch_size=args.batch_size)
                _taps["rand"] = CryoFM2Tap(args.ckpt, taps=(args.tap,), device=dev,
                                           batch_size=args.batch_size,
                                           random_weights=True)
            return _taps["real"], _taps["rand"]

        rng = np.random.default_rng(args.seed)
        cache_key, cache_vol = None, None
        done = skipped = 0
        for i, r in enumerate(rows):
            part = fdir / f"{r['key']}.npz"
            if part.exists():
                done += 1
                continue
            try:
                d = Path(r["pdb"]).parent
                if cache_key != d.name:                  # reuse across chains
                    vol, vs, origin = load_map(str(d / f"{d.name}_raw_emd.map"))
                    cache_vol = (preprocess(to_cubic_even(vol), vs), origin)
                    cache_key = d.name
                norm, origin = cache_vol

                seq, ca, fr, nums = backbone_with_resnum(r["pdb"], r["chain"])
                seq_ref, _, _ = chain_backbone(r["pdb"], r["chain"])
                if seq != seq_ref:
                    raise ValueError("resnum reader diverged from chain_backbone")
                if seq != r["seq"]:
                    raise ValueError("observed sequence differs from the chain list")

                coords = (ca - np.asarray(origin)[None]) / MODEL_VOXEL_SIZE
                shape = np.array(norm.shape)
                ok = np.all((coords >= PATCH // 2)
                            & (coords < shape[None] - PATCH // 2), axis=1)
                ssl = ss_labels(d)
                aa = np.array([AA1.index(c) if c in AA1 else -1 for c in seq])
                ss = np.array([ssl.get((r["chain"], int(n)), -1) for n in nums])
                ok &= (aa >= 0) & (ss >= 0)
                idx = np.nonzero(ok)[0]
                if len(idx) < 10:
                    raise ValueError(f"only {len(idx)} usable residues")
                if len(idx) > args.per_chain:
                    idx = rng.choice(idx, args.per_chain, replace=False)

                tap_real, tap_rand = taps()
                boxes = extract_local_boxes(torch.from_numpy(norm), coords[idx],
                                            fr[idx], device=dev, chunk=args.box_chunk)
                feats_by_setting = {}
                for (tt, mode), nm in zip(SETTINGS, names):
                    nl = (tt / 1000.0) if mode == "c" else None
                    feats_by_setting[nm] = centre_features(
                        tap_real, boxes, tt, args.batch_size,
                        noise_level=nl)[args.tap].astype(np.float32)
                    tap_real._buf.clear()
                # one random-weight control, at the reference point
                f_rand = centre_features(tap_rand, boxes, SETTINGS[0][0],
                                         args.batch_size)[args.tap]
                tap_rand._buf.clear()
                c = PATCH // 2
                raw = boxes[:, 0, c - 4:c + 4, c - 4:c + 4, c - 4:c + 4]
                raw = raw.reshape(len(idx), -1).float().cpu().numpy()
                del boxes
                if dev == "cuda":
                    torch.cuda.empty_cache()

                np.savez(part,
                         random=f_rand.astype(np.float32),
                         **feats_by_setting,
                         raw=raw.astype(np.float32),
                         aa=aa[idx], ss=ss[idx],
                         split=np.array([r["split"]] * len(idx)),
                         cluster=np.array([r["cluster"]] * len(idx)))
                done += 1
            except Exception as exc:
                skipped += 1
                if skipped <= 15:
                    print(f"  SKIP {r['key']} {type(exc).__name__}: {exc}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] {done} ok, {skipped} skipped",
                      flush=True)
        print(f"chains: {done} ok, {skipped} skipped", flush=True)
        keys = tuple(names) + ("random", "raw", "aa", "ss", "split", "cluster")
        parts = [fdir / f"{r['key']}.npz" for r in rows]
        parts = [q for q in parts if q.exists()]
        if not parts:
            raise SystemExit("no per-chain parts were produced")
        print(f"aggregating {len(parts)} per-chain parts", flush=True)
        buf: dict[str, list] = {k: [] for k in keys}
        for q in parts:
            z = np.load(q, allow_pickle=True)
            for k in keys:
                buf[k].append(z[k])
        F = {k: np.concatenate(v) for k, v in buf.items()}
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, **F)
        print(f"cached -> {args.cache}", flush=True)

    # ---------------- probes ----------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    split = F["split"].astype(str)
    tr, te = split == "train", split == "test"
    print(f"\nresidues: train {tr.sum()}  test {te.sum()}  "
          f"({len(np.unique(F['cluster'][tr]))} train clusters, "
          f"{len(np.unique(F['cluster'][te]))} test clusters)")

    if tr.sum() == 0 or te.sum() == 0:
        # Cheap insurance: without this an otherwise-complete 2-3 h extraction
        # dies at the very last step (np.bincount over an empty test split).
        raise SystemExit(
            f"empty split: train={tr.sum()} test={te.sum()}. With --limit the "
            "sampled chains can all land in one split; the features are cached, "
            "so re-run the probes with more chains.")

    res: dict[str, dict] = {}
    for task in ("ss", "aa"):
        y = F[task]
        maj = float(np.bincount(y[te]).max() / te.sum())
        res.setdefault(task, {})["prior"] = {"acc": maj, "macro_f1": None}
        print(f"\n=== task {task} ({len(np.unique(y))} classes) ===")
        print(f"  {'arm':10s} {'dim':>5s} {'acc':>8s} {'macroF1':>9s}   (prior acc {maj:.4f})")
        arms = ["raw", "random"] + [n for n in names if n in F]
        for arm in arms:
            X = F[arm]
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict(sc.transform(X[te]))
            a = float(accuracy_score(y[te], p))
            f1 = float(f1_score(y[te], p, average="macro"))
            res[task][arm] = {"acc": a, "macro_f1": f1, "dim": int(X.shape[1])}
            print(f"  {arm:10s} {X.shape[1]:5d} {a:8.4f} {f1:9.4f}", flush=True)

    res["_meta"] = {
        "tap": args.tap, "timestep": args.timestep, "per_chain": args.per_chain,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "density": "REAL experimental maps (Cryo2StructData raw_emd)",
        "external_reference": "ModelAngelo ~49% top-1 AA at 4-5 A",
        "kill_criterion": "cryofm must beat raw voxels, else the teacher is out",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
