"""Stage 1a / 1b: does an ESMFold2-style PAIR channel over ESM-C beat the
per-residue readout on the density target?

Stage 0 measured the ceilings first (results/stage0_oracle.json):
  * `shellcomp` = what ONE outer product + row pooling can express given perfect
    ground-truth distances -> 0.175 alone, and `esmc+shellcomp` = 0.245. So
    Stage 1a's ceiling is 0.245 vs the 0.187 arm to beat: at most +0.058, and
    only if the pair channel infers distances as well as ground truth does.
  * `dirmom` (frame-directional geometry) = 0.401. NOT expressible as a sum of
    pairwise terms -- recovering it needs triple-wise reasoning, which is what
    the triangle multiplicative updates of Stage 1b are for.
So 1a is expected to gain little and 1b is the interesting arm. Running both
because a ceiling is not a measurement.

ARCHITECTURE mirrors ESMFold2's real ESM-C pathway (verified against the
released checkpoint; see ESMFOLD2_LM_PATHWAY.md):
    single -> SingleToPair: concat[x_i*x_j, x_i-x_j] -> MLP        (Stage 1a)
           -> N x PairUpdateBlock: TriMul(out) + TriMul(in) + Transition  (1b)
           -> RowAttentionPooling -> per-residue vector
The head is EXACTLY the baseline MLP with the pooled vector concatenated, so
with the pair channel off the model is the 0.187 baseline architecture and any
delta is attributable to the pair channel alone.

Two init choices, both from this project's own logged lessons:
  * every residual branch (`proj_emit`, transition `w3`) is ZERO-init, so each
    PairUpdateBlock is identity at init -- a 16-block residual decoder here once
    diverged to R2 -371 without this.
  * `row_pool.out_proj` is ZERO-init, so training STARTS exactly at the baseline
    and the pair channel can only earn its way in.

CONTROL ARMS, all mandatory:
  none          pair channel off -> must reproduce ~0.187, else the harness is
                broken and no other number here means anything.
  relpos        pair channel fed ONLY relative sequence position, no ESM-C. The
                sibling project found >50% of short-range predictability was
                backbone adjacency, so |i-j| alone in a pair rep could
                manufacture most of a positive.
  esm+relpos    both, to see whether they are complementary.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

RELPOS_CLIP = 32


# ---------------------------------------------------------------- modules

class SingleToPair(nn.Module):
    """ESMFold2 SingleToPair: downproject -> concat[prod, diff] -> MLP."""

    def __init__(self, d_s: int, d0: int, d_pair: int):
        super().__init__()
        self.downproject = nn.Linear(d_s, d0)
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * d0, d_pair), nn.GELU(), nn.Linear(d_pair, d_pair))
        self.norm = nn.LayerNorm(d_pair)

    def forward(self, x):                                  # [B, L, d_s]
        x = self.downproject(x)
        outer = torch.cat([x[:, :, None, :] * x[:, None, :, :],
                           x[:, :, None, :] - x[:, None, :, :]], dim=-1)
        return self.norm(self.output_mlp(outer))           # [B, L, L, d_pair]


class TriMul(nn.Module):
    """ESMFold2 TriangleMultiplicativeBlock. latent == d_pair, as in the ckpt."""

    def __init__(self, d: int, flow: str):
        super().__init__()
        self.norm_start = nn.LayerNorm(d)
        self.norm_mix = nn.LayerNorm(d)
        self.proj_bundle = nn.Linear(d, 4 * d, bias=False)
        self.proj_emit = nn.Linear(d, d, bias=False)
        self.proj_gate = nn.Linear(d, d, bias=False)
        self.flow = flow
        nn.init.zeros_(self.proj_emit.weight)              # identity at init

    def forward(self, z):                                  # [B, L, L, d]
        n = self.norm_start(z)
        signal, gate = self.proj_bundle(n).chunk(2, dim=-1)
        routed = signal * torch.sigmoid(gate)
        left, right = routed.chunk(2, dim=-1)
        if self.flow == "out":
            c = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            c = torch.einsum("bkid,bkjd->bijd", left, right)
        return self.proj_emit(self.norm_mix(c)) * torch.sigmoid(self.proj_gate(n))


class PairUpdateBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.tri_out = TriMul(d, "out")
        self.tri_in = TriMul(d, "in")
        self.norm = nn.LayerNorm(d)
        self.w12 = nn.Linear(d, 8 * d, bias=False)         # SwiGLU, hidden 4d
        self.w3 = nn.Linear(4 * d, d, bias=False)
        nn.init.zeros_(self.w3.weight)                     # identity at init

    def forward(self, z):
        z = z + self.tri_out(z)
        z = z + self.tri_in(z)
        h = self.norm(z)
        a, b = self.w12(h).chunk(2, dim=-1)
        return z + self.w3(F.silu(a) * b)


class RowAttnPool(nn.Module):
    """ESMFold2 RowAttentionPooling: scalar score per pair, softmax over j."""

    def __init__(self, d: int):
        super().__init__()
        self.attn_proj = nn.Linear(d, 1)
        self.out_proj = nn.Linear(d, d)
        nn.init.zeros_(self.out_proj.weight)               # start AT baseline
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z):                                  # [B, L, L, d]
        w = self.attn_proj(z).squeeze(-1).softmax(dim=-1)   # [B, L, L]
        return self.out_proj(torch.einsum("bnm,bnmd->bnd", w, z))


class Stage1(nn.Module):
    def __init__(self, d_in=1152, d_out=512, pair="esm", n_tri=0,
                 d_s=256, d0=64, d_pair=64, hidden=2048, ckpt_blocks=True):
        super().__init__()
        self.pair, self.n_tri, self.ckpt_blocks = pair, n_tri, ckpt_blocks
        self.use_esm_pair = pair in ("esm", "esm+relpos")
        self.use_relpos = pair in ("relpos", "esm+relpos")
        if self.use_esm_pair:
            self.single_proj = nn.Sequential(nn.Linear(d_in, d_s), nn.GELU())
            self.to_pair = SingleToPair(d_s, d0, d_pair)
        if self.use_relpos:
            self.relpos = nn.Linear(2 * RELPOS_CLIP + 1, d_pair, bias=False)
        on = pair != "none"
        self.blocks = nn.ModuleList([PairUpdateBlock(d_pair)
                                     for _ in range(n_tri)]) if on else None
        self.pool = RowAttnPool(d_pair) if on else None
        head_in = d_in + (d_pair if on else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_out))

    def forward(self, xs, relpos_oh=None):                 # xs [B, L, 1152]
        if self.pair == "none":
            return self.head(xs)
        z = 0.0
        if self.use_esm_pair:
            z = z + self.to_pair(self.single_proj(xs))
        if self.use_relpos:
            z = z + self.relpos(relpos_oh)
        for blk in (self.blocks or []):
            z = (checkpoint(blk, z, use_reentrant=False)
                 if (self.ckpt_blocks and self.training) else blk(z))
        return self.head(torch.cat([xs, self.pool(z)], dim=-1))


# ---------------------------------------------------------------- data

def load_all(chains: Path, esmc_dir: Path, tgt_dir: Path):
    rows = list(csv.DictReader(open(chains)))
    data = {"train": [], "val": [], "test": []}
    for r in rows:
        e, t = esmc_dir / f"{r['key']}.npy", tgt_dir / f"{r['key']}.npz"
        if not (e.exists() and t.exists()):
            continue
        d = np.load(t, allow_pickle=True)
        feats, keep, seq = d["feats"], d["keep"], str(d["seq"])
        emb = np.load(e)
        if len(emb) != len(seq) or len(keep) != len(seq) or int(keep.sum()) != len(feats):
            continue
        if not keep.all():          # would need a target mask; never happens here
            continue
        data[r["split"]].append((emb.astype(np.float32), feats.astype(np.float32)))
    return data


def relpos_onehot(L: int, device):
    i = torch.arange(L, device=device)
    b = (i[:, None] - i[None, :]).clamp(-RELPOS_CLIP, RELPOS_CLIP) + RELPOS_CLIP
    return F.one_hot(b, 2 * RELPOS_CLIP + 1).float()[None]


# ---------------------------------------------------------------- train/eval

def evaluate(net, chains, mx, sx, my, sy, dev, need_relpos):
    """Predictions -> the SAME pooled-R2 / centred-cosine as align_esmc_density."""
    net.eval()
    P, Y = [], []
    with torch.no_grad():
        for emb, feats in chains:
            xs = torch.from_numpy((emb - mx) / sx).to(dev)[None]
            rp = relpos_onehot(len(emb), dev) if need_relpos else None
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                p = net(xs, rp)
            P.append(p[0].float().cpu().numpy() * sy + my)
            Y.append(feats)
    P, Y = np.concatenate(P), np.concatenate(Y)
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - my) ** 2).sum(0)
    r2 = float(1.0 - ss_res.sum() / ss_tot.sum())
    cos = float(np.mean(np.sum((Y - my) * (P - my), 1) /
                        np.maximum(np.linalg.norm(Y - my, axis=1)
                                   * np.linalg.norm(P - my, axis=1), 1e-9)))
    return r2, cos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, default=Path("data/alignment_chains.csv"))
    ap.add_argument("--esmc-dir", type=Path, default=Path("data/esmc_chains"))
    ap.add_argument("--target-dir", type=Path, default=Path("data/density_targets"))
    ap.add_argument("--pair", choices=["none", "esm", "relpos", "esm+relpos"],
                    default="esm")
    ap.add_argument("--n-tri", type=int, default=0, help="0 = Stage 1a, >0 = 1b")
    ap.add_argument("--d-pair", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--accum", type=int, default=16, help="chains per opt step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/stage1_pair.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"{args.pair}_tri{args.n_tri}"
    print(f"[{tag}] device={dev} pair={args.pair} n_tri={args.n_tri} "
          f"d_pair={args.d_pair}", flush=True)

    D = load_all(args.chains, args.esmc_dir, args.target_dir)
    print(f"chains train/val/test: {len(D['train'])}/{len(D['val'])}/{len(D['test'])}",
          flush=True)
    Xtr = np.concatenate([e for e, _ in D["train"]])
    Ytr = np.concatenate([f for _, f in D["train"]])
    mx, sx = Xtr.mean(0), Xtr.std(0)
    sx[sx < 1e-8] = 1.0
    my, sy = Ytr.mean(0), Ytr.std(0)
    sy[sy < 1e-8] = 1.0
    print(f"residues train/val/test: {len(Ytr)}/"
          f"{sum(len(f) for _, f in D['val'])}/"
          f"{sum(len(f) for _, f in D['test'])}", flush=True)
    del Xtr, Ytr

    need_rp = args.pair in ("relpos", "esm+relpos")
    net = Stage1(pair=args.pair, n_tri=args.n_tri, d_pair=args.d_pair).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    n_pair_par = sum(p.numel() for n, p in net.named_parameters()
                     if not n.startswith("head."))
    print(f"params: {n_par/1e6:.2f} M total, {n_pair_par/1e6:.2f} M in the "
          f"pair channel", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)

    best, best_state, bad, step = np.inf, None, 0, 0
    for ep in range(args.epochs):
        net.train()
        order = np.random.permutation(len(D["train"]))
        opt.zero_grad(set_to_none=True)
        t0, run = time.time(), 0.0
        for c, ci in enumerate(order):
            emb, feats = D["train"][ci]
            xs = torch.from_numpy((emb - mx) / sx).to(dev)[None]
            y = torch.from_numpy((feats - my) / sy).to(dev)[None]
            rp = relpos_onehot(len(emb), dev) if need_rp else None
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                loss = F.mse_loss(net(xs, rp), y)
            (loss / args.accum).backward()
            run += loss.item()
            if (c + 1) % args.accum == 0 or c + 1 == len(order):
                step += 1
                if args.warmup:
                    for g in opt.param_groups:
                        g["lr"] = args.lr * min(1.0, step / args.warmup)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        # val MSE in standardized space
        net.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for emb, feats in D["val"]:
                xs = torch.from_numpy((emb - mx) / sx).to(dev)[None]
                y = torch.from_numpy((feats - my) / sy).to(dev)[None]
                rp = relpos_onehot(len(emb), dev) if need_rp else None
                with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                    vl += F.mse_loss(net(xs, rp).float(), y,
                                     reduction="sum").item()
                vn += y.numel()
        vl /= vn
        print(f"  ep {ep:2d} train {run/len(order):.4f} val {vl:.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stop", flush=True)
                break

    net.load_state_dict(best_state)
    r2, cos = evaluate(net, D["test"], mx, sx, my, sy, dev, need_rp)
    print(f"[{tag}] TEST R2 {r2:+.4f}  cos {cos:+.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    res[tag] = {"r2": r2, "cos": cos, "pair": args.pair, "n_tri": args.n_tri,
                "d_pair": args.d_pair, "params_M": round(n_par / 1e6, 3),
                "pair_params_M": round(n_pair_par / 1e6, 3),
                "best_val_mse": best, "epochs_run": ep + 1, "seed": args.seed}
    res["_meta"] = {"arm_to_beat_esmc_mlp": 0.187, "esmc_ridge": 0.1694,
                    "stage0_1a_ceiling_esmc_plus_shellcomp": 0.2446,
                    "stage0_dirmom": 0.4009, "stage0_full_oracle": 0.4099,
                    "sequence_tracked": 0.447, "pose_ceiling": 0.849}
    args.out.write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
