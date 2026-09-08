"""Feature tap for CryoFM2 (UNet3D).

CryoFM has no encoder -- it is a pure flow/velocity denoiser v(x_t, t). "Features"
here means internal activations captured by forward hooks at a chosen noise level,
the standard diffusion-features technique. This is NOT validated by the CryoFM
papers; establishing whether the features are usable is exactly what Phase 0 does.

Preprocessing mirrors CryoFM2's own inference path exactly
(third_party/cryofm/src/cryofm/projects/cryofm2/sampling_helper.py:318):

    resample to 1.5 A/voxel (Fourier, energy-preserving)
    max_value = percentile(vol, 99.999)
    vol = vol / max_value
    vol = (vol - 0.04) / 0.09

Note there is NO negative clipping -- that belongs to CryoFM1's ClipNorm3D, not
CryoFM2. Getting this wrong silently biases every feature.

Tap points and their granularity at 1.5 A/voxel:

    mid_block      512 tok x 512 ch   12.0 A/token   coarse / most semantic
    up_blocks[0]  4096 tok x 512 ch    6.0 A/token   ~1-2 residues  <- primary
    up_blocks[1] 32768 tok x 256 ch    3.0 A/token   fine detail
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from cryofm.core.datasets.transforms.patchify import GridPatches3D
from cryofm.core.models.unet3d.unet import UNet3DModel
from cryofm.projects.cryofm2.utils.infer_relion_utils import resize_by_voxel_size

MODEL_VOXEL_SIZE = 1.5
CRYOEM_DENSITY_MEAN = 0.04
CRYOEM_DENSITY_STD = 0.09
PATCH = 64

# Spatial downsample factor of each tap relative to the input voxel grid.
TAP_STRIDES = {
    "conv_in": 1,
    "down_blocks[0]": 2,
    "down_blocks[1]": 4,
    "down_blocks[2]": 8,
    "down_blocks[3]": 8,
    "mid_block": 8,
    "up_blocks[0]": 4,
    "up_blocks[1]": 2,
    "up_blocks[2]": 1,
    "up_blocks[3]": 1,
}
DEFAULT_TAPS = ("mid_block", "up_blocks[0]", "up_blocks[1]")


# --------------------------------------------------------------------------
# Cube rotations (octahedral group, 24 elements)
# --------------------------------------------------------------------------

def cube_rotations() -> list[np.ndarray]:
    """The 24 proper rotations of the cube, as integer 3x3 matrices.

    Enumerates signed permutation matrices and keeps det == +1. Applying these
    to a volume is lossless (transpose + flip only, no interpolation), which
    matters for a pose-stability test -- any interpolation would confound the
    measurement with resampling error.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3), dtype=np.int64)
            for axis, (p, s) in enumerate(zip(perm, signs)):
                R[axis, p] = s
            if round(float(np.linalg.det(R))) == 1:
                mats.append(R)
    assert len(mats) == 24, len(mats)
    return mats


def rotate_volume(vol: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a cube rotation to a volume indexed [x, y, z].

    Convention: for a voxel at integer index v (relative to the volume centre),
    the rotated volume satisfies  out[R @ v] == vol[v].
    """
    perm = [int(np.argmax(np.abs(R[a]))) for a in range(3)]
    out = np.transpose(vol, axes=perm)
    for a in range(3):
        if R[a, perm[a]] < 0:
            out = np.flip(out, axis=a)
    return np.ascontiguousarray(out)


def rotate_coords(coords: np.ndarray, R: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Rotate voxel-index coordinates [N, 3] consistently with `rotate_volume`."""
    centre = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    rotated_shape = np.abs(R) @ np.asarray(shape, dtype=np.float64)
    new_centre = (rotated_shape - 1.0) / 2.0
    return (coords - centre) @ R.T + new_centre


# --------------------------------------------------------------------------
# Model loading and preprocessing
# --------------------------------------------------------------------------

def load_cryofm2(ckpt_dir: str, device: str = "cuda",
                 random_weights: bool = False,
                 trainable: bool = False) -> UNet3DModel:
    """Load CryoFM2. `random_weights=True` keeps the architecture but skips the
    checkpoint -- the control that separates "the trained network encodes this"
    from "any 3D conv over density does".

    That control is not hypothetical here: an untrained UNet still reaches pooled
    retrieval 0.58-0.68 against a chance of 0.10, carried by map size and gross
    density statistics alone. It is the pretrained/random *gap* that is evidence
    (4x per-residue, 2-3x on pooled d'), never the absolute number.
    """
    cfg = yaml.unsafe_load(open(f"{ckpt_dir}/config.yaml"))
    model = UNet3DModel(**cfg["model"])
    if not random_weights:
        missing, unexpected = model.load_state_dict(
            load_file(f"{ckpt_dir}/model.safetensors"), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.eval().to(device)
    if trainable:
        # `.eval()` is deliberately kept even for the student: dropout is 0.0 and
        # the norms are GroupNorm (no running stats), so eval() changes nothing
        # about gradients, and keeping it means the student and the teacher differ
        # in exactly one respect -- requires_grad.
        model.requires_grad_(True)
    return model


def preprocess(vol: np.ndarray, src_voxel_size: float) -> np.ndarray:
    """Resample to 1.5 A/voxel and apply CryoFM2's normalization."""
    vol = np.asarray(vol, dtype=np.float32)
    if abs(src_voxel_size - MODEL_VOXEL_SIZE) > 1e-6:
        vol = np.asarray(resize_by_voxel_size(vol, src_voxel_size, MODEL_VOXEL_SIZE), dtype=np.float32)
    max_value = np.percentile(vol, 99.999)
    if max_value <= 0:
        raise ValueError(f"non-positive 99.999th percentile ({max_value}); map likely empty or inverted")
    vol = vol / max_value
    return ((vol - CRYOEM_DENSITY_MEAN) / CRYOEM_DENSITY_STD).astype(np.float32)


class StopForward(Exception):
    """Raised from a capture hook to abandon the rest of the forward pass.

    The tail past `up_blocks[1]` runs at 64^3 and is roughly half the network's
    FLOPs while contributing nothing to any tap we read. Catch it ONLY as
    `StopForward` -- a bare `except Exception` here would turn a real CUDA or
    shape error into a silently short feature dict.
    """


@dataclass
class FeatureVolume:
    """A tap's activations stitched back over the whole (resampled) map grid."""
    data: np.ndarray          # [C, D/s, H/s, W/s]
    stride: int               # voxels of the 1.5 A grid per feature cell
    grid_shape: tuple         # shape of the 1.5 A voxel grid it came from

    @property
    def angstroms_per_cell(self) -> float:
        return self.stride * MODEL_VOXEL_SIZE


# --------------------------------------------------------------------------
# The tap
# --------------------------------------------------------------------------

class CryoFM2Tap:
    def __init__(
        self,
        ckpt_dir: str,
        taps: tuple[str, ...] = DEFAULT_TAPS,
        device: str = "cuda",
        batch_size: int = 8,
        random_weights: bool = False,
        detach: bool = True,
        stop_after: str | None = None,
        trainable: bool = False,
        student_ckpt: str | None = None,
    ):
        self.model = load_cryofm2(ckpt_dir, device, random_weights=random_weights,
                                  trainable=trainable)
        self.taps = tuple(taps)
        self.device = device
        self.batch_size = batch_size
        self.detach = detach
        self.stop_after = stop_after
        self._buf: dict[str, torch.Tensor] = {}
        self._register(taps)
        if student_ckpt is not None:
            self.load_student(student_ckpt)

    def load_student(self, path: str) -> None:
        """Load a CleanDIFT student checkpoint.

        ORDER MATTERS. The student swaps `time_embedding` for a `LearnedTimeEmb`
        (a single `p` parameter), so its keys are `time_embedding.p` and NOT
        `time_embedding.linear_1.*`. The swap must therefore happen BEFORE the
        load. `strict=True` is deliberate: with `strict=False` a mismatch would
        silently leave the learned conditioning vector at its initialisation and
        the student would be evaluated at whatever `t_init` the constructor had.
        """
        from teachers.cryofm_student import attach_const_time_embedding
        sd = torch.load(path, map_location=self.device, weights_only=True)
        sd = sd.get("student", sd)
        if any(k.startswith("time_embedding.p") for k in sd):
            attach_const_time_embedding(
                self.model, torch.zeros(sd["time_embedding.p"].shape,
                                        device=self.device))
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

    def resolve_tap(self, name: str):
        """Module object for a tap name. Extracted from `_register` unchanged so
        the student path can reach the same modules without duplicating the
        parsing."""
        if name.startswith("up_blocks["):
            return self.model.up_blocks[int(name[10:-1])]
        if name.startswith("down_blocks["):
            return self.model.down_blocks[int(name[12:-1])]
        if name == "mid_block":
            return self.model.mid_block
        if name == "conv_in":
            return self.model.conv_in
        raise ValueError(f"unknown tap: {name}")

    def _register(self, taps) -> None:
        """Attach capture hooks.

        `self.detach` False is the whole of the CleanDIFT gradient plumbing: the
        hook then stores the live tensor, so a loss on `_buf` backpropagates into
        the trunk. It is off by default, so every existing probe is byte-for-byte
        unaffected. Note `@torch.no_grad()` on `feature_volumes` /
        `sample_frame_averaged` can stay: the student path calls `self.model`
        directly (as `local_frame_stability.centre_features` already does) rather
        than through those wrappers.

        `self.stop_after` truncates the forward once the deepest wanted tap has
        fired, by raising `StopForward` from the hook. Safe because forward hooks
        run AFTER the module returns, so no `torch.utils.checkpoint` frame is
        mid-flight and the graph for everything already computed stays intact.
        """
        def mk(name):
            def hook(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                self._buf[name] = t if not self.detach else t.detach().float()
                if self.stop_after is not None and name == self.stop_after:
                    raise StopForward(name)
            return hook

        for name in taps:
            self.resolve_tap(name).register_forward_hook(mk(name))

    def forward_taps(self, x: torch.Tensor, timestep) -> dict[str, torch.Tensor]:
        """Run the model for its taps only, honouring `stop_after`.

        Deliberately does NOT bind the model's return value: with the full
        forward, `up_blocks[2..3]`'s saved tensors are only reachable through the
        returned `UNet3DOutput.sample`, and holding it pins memory we never use.
        """
        self._buf.clear()
        try:
            self.model(x, timestep=timestep)
        except StopForward:
            pass
        missing = [t for t in self.taps if t not in self._buf]
        if missing:
            raise RuntimeError(
                f"taps never fired: {missing}. If `stop_after` is set it must be the "
                f"DEEPEST tap in execution order, else shallower taps are skipped.")
        return dict(self._buf)

    @torch.no_grad()
    def feature_volumes(
        self,
        vol_norm: np.ndarray,
        timestep: int = 10,
        noise_seed: int | None = 0,
        noise_level: float = 0.0,
        align_patches: bool = True,
    ) -> dict[str, FeatureVolume]:
        """Run the denoiser over a tiled map and stitch each tap's activations.

        vol_norm must already be preprocessed (1.5 A/voxel, normalized).

        `timestep` picks where on the flow trajectory to evaluate. `noise_level`
        interpolates toward Gaussian noise: 0.0 feeds the clean map (the usual
        diffusion-features choice); >0 mixes in noise so that noise-stability
        can be measured.
        """
        # Pad each axis up to a MULTIPLE of PATCH (this subsumes the old "at least
        # one patch" pad, since the next multiple of 64 for n<64 is 64).
        #
        # Why a multiple and not just >= PATCH: GridPatches3D appends a final patch
        # per axis at `n - 64` whenever that is off the step grid (patchify.py:389),
        # and `n - 64` is a multiple of the tap stride only if `n` is. The stitching
        # loop below places features at `a // s`, so an off-grid patch is written
        # at a TRUNCATED destination -- misregistered by `a mod s` feature cells,
        # i.e. up to 7 voxels = 10.5 A at mid_block. Measured over random grids:
        # 56.7% of patches are misregistered at stride 8, 51.1% at 4, 35.7% at 2.
        # Worse, it is pose-DEPENDENT: a cube rotation permutes the axis lengths,
        # so a residue lands in a misregistered patch in some poses and not others,
        # which corrupts exactly the pose-stability measurement Phase 0a reports.
        # Padding to a multiple of 64 makes every patch start a multiple of 64 and
        # therefore of every tap stride. It costs nothing: the patch count per axis
        # is ceil(n/64) either way.
        #
        # Pad VALUE is the per-map median, i.e. the background level. Zero is wrong
        # here -- normalization maps raw density 0 to (0-0.04)/0.09 = -0.44, so a
        # zero pad injects a slab of above-background density around the border.
        # `align_patches=False` restores the old (buggy) behaviour -- pad only up
        # to one patch -- so the effect can be A/B'd in one process. Keep it: the
        # fix is obviously correct in principle, but on the Phase 0a protocol
        # (25 maps, 8 rotations, 380k residues) it moved pose consistency by
        # <0.01, so its practical size is an empirical question, not a given.
        pad = [(0, (-s) % PATCH if align_patches else max(0, PATCH - s))
               for s in vol_norm.shape]
        if any(p[1] for p in pad):
            vol_norm = np.pad(vol_norm, pad, mode="constant",
                              constant_values=float(np.median(vol_norm)))
        grid_shape = vol_norm.shape

        locs = GridPatches3D._get_patches_locations(
            np.array(grid_shape), np.array([PATCH] * 3), (0, 0, 0)
        )

        acc: dict[str, np.ndarray] = {}
        cnt: dict[str, np.ndarray] = {}
        for name in self.taps:
            s = TAP_STRIDES[name]
            fshape = tuple(d // s for d in grid_shape)
            acc[name] = None  # allocated once channel count is known
            cnt[name] = np.zeros(fshape, dtype=np.float32)

        gen = torch.Generator(device="cpu")
        if noise_seed is not None:
            gen.manual_seed(noise_seed)

        for i0 in range(0, len(locs), self.batch_size):
            chunk = locs[i0 : i0 + self.batch_size]
            patches = np.stack([
                vol_norm[a:d, b:e, c:f] for a, b, c, d, e, f in chunk
            ])
            x = torch.from_numpy(patches).float().unsqueeze(1)  # [B,1,64,64,64]
            if noise_level > 0:
                eps = torch.randn(x.shape, generator=gen)
                x = (1.0 - noise_level) * x + noise_level * eps
            # in_channels=2: ch0 = x_t, ch1 = conditioning (zeros, unconditional).
            x = torch.cat([x, torch.zeros_like(x)], dim=1).to(self.device)
            t = torch.full((x.shape[0],), timestep, device=self.device, dtype=torch.long)

            self._buf.clear()
            # Honour `stop_after` here as `forward_taps` already does. Without
            # this, constructing the tap with `stop_after=` and then calling
            # `feature_volumes` raised StopForward straight out of the method --
            # the two entry points disagreed, and the whole-map path was simply
            # unusable with the truncation that makes it affordable. Catch ONLY
            # StopForward: a bare `except Exception` would turn a real CUDA or
            # shape error into a silently short feature dict.
            try:
                self.model(x, timestep=t)
            except StopForward:
                pass
            missing = [n for n in self.taps if n not in self._buf]
            assert not missing, (
                f"taps {missing} were never populated -- `stop_after="
                f"{self.stop_after}` truncates the forward before them")

            for name in self.taps:
                s = TAP_STRIDES[name]
                feat = self._buf[name].cpu().numpy()  # [B, C, p, p, p]
                if acc[name] is None:
                    C = feat.shape[1]
                    acc[name] = np.zeros((C,) + tuple(d // s for d in grid_shape), dtype=np.float32)
                for j, (a, b, c, _d, _e, _f) in enumerate(chunk):
                    # Guard the misregistration bug above: an unaligned start would
                    # be silently truncated by `// s` rather than erroring. Only
                    # meaningful in the aligned arm -- `align_patches=False`
                    # reproduces the bug deliberately, so asserting there would
                    # make the A/B unrunnable.
                    assert not align_patches or (a % s == 0 and b % s == 0 and c % s == 0), (
                        f"patch start ({a},{b},{c}) is not a multiple of stride {s} "
                        f"for tap {name}; grid {grid_shape} must be padded to a "
                        "multiple of PATCH")
                    za, ya, xa = a // s, b // s, c // s
                    p = feat.shape[-1]
                    acc[name][:, za:za + p, ya:ya + p, xa:xa + p] += feat[j]
                    cnt[name][za:za + p, ya:ya + p, xa:xa + p] += 1.0

        out = {}
        for name in self.taps:
            out[name] = FeatureVolume(
                data=acc[name] / np.maximum(cnt[name], 1.0)[None],
                stride=TAP_STRIDES[name],
                grid_shape=grid_shape,
            )
        return out


    @torch.no_grad()
    def sample_frame_averaged(
        self,
        vol_norm: np.ndarray,
        coords_voxel: np.ndarray,
        n_frames: int = 24,
        **kw,
    ) -> dict[str, np.ndarray]:
        """Features at `coords_voxel`, averaged over the octahedral group.

        Returns ``{tap: [N, C]}``.

        CryoFM2 is not equivariant, so a single forward pass gives a feature that
        depends on how the molecule happens to sit on the voxel grid. Averaging
        over a group G makes the result exactly G-invariant by construction (Puny
        et al., ICLR 2022) -- and for the 24 cube rotations the group action is
        lossless (transpose + flip, no interpolation), so nothing is blurred to
        buy the invariance. It needs no atomic model and no canonical frame, which
        is what every other route here has foundered on.

        Averaging is done on SAMPLED VECTORS, not on stitched feature volumes, on
        purpose: each rotation pads to its own multiple of 64, so the feature
        grids have different shapes and un-rotating them would need a crop-and-
        align step that is easy to get subtly wrong. Sampling first sidesteps it.
        Channels are not permuted by the rotation -- the network is not
        equivariant, so there is no channel correspondence to track; we are
        averaging the values a fixed channel takes across poses.

        Measured by an independent review, under GENERIC SO(3) (i.e. rotations
        outside the averaged group, the honest test):

            mid_block     0.377 -> 0.634      top-1 0.029 -> 0.085
            up_blocks[0]  0.298 -> 0.634      top-1 0.030 -> 0.130
            up_blocks[1]  0.451 -> 0.767      top-1 0.084 -> 0.276

        Cost is `n_frames` forward passes. The K-sweep showed errors are largely
        independent across frames, so this is not near saturation at 24.
        """
        rots = cube_rotations()[:n_frames]
        shape = tuple(vol_norm.shape)
        acc: dict[str, np.ndarray] = {}
        for R in rots:
            fvs = self.feature_volumes(rotate_volume(vol_norm, R), **kw)
            # Unpadded rotated shape -- padding is applied on the +side only, so
            # it never shifts an index and must not enter the coordinate map.
            rc = rotate_coords(np.asarray(coords_voxel, dtype=np.float64), R, shape)
            for name, fv in fvs.items():
                v = sample_at(fv, rc)
                acc[name] = v if name not in acc else acc[name] + v
        return {k: v / len(rots) for k, v in acc.items()}


def sample_at(fv: FeatureVolume, coords_voxel: np.ndarray) -> np.ndarray:
    """Trilinearly sample a FeatureVolume at 1.5 A-grid voxel coordinates [N, 3].

    Returns [N, C]. Points outside the volume are zero-filled by grid_sample's
    default padding; callers should mask them out rather than trust them.
    """
    C = fv.data.shape[0]
    fshape = np.array(fv.data.shape[1:], dtype=np.float64)

    # Voxel index -> feature-cell index (cell centres sit at stride/2 offsets).
    cell = (np.asarray(coords_voxel, dtype=np.float64) - (fv.stride - 1) / 2.0) / fv.stride
    # -> normalized [-1, 1] for grid_sample, which expects (x, y, z) order.
    norm = 2.0 * cell / (fshape - 1.0) - 1.0
    grid = torch.from_numpy(norm[:, ::-1].copy()).float().view(1, -1, 1, 1, 3)

    vals = F.grid_sample(
        torch.from_numpy(fv.data).unsqueeze(0),
        grid,
        mode="bilinear",
        align_corners=True,
    )
    return vals.view(C, -1).T.numpy()


if __name__ == "__main__":
    # Self-test: the rotation convention must be internally consistent, i.e.
    # rotating the volume and rotating the coordinates must agree. If this is
    # wrong the pose-stability measurement is meaningless.
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(12, 14, 16)).astype(np.float32)
    pts = rng.integers(0, [12, 14, 16], size=(200, 3))

    bad = 0
    for R in cube_rotations():
        rvol = rotate_volume(vol, R)
        rpts = np.rint(rotate_coords(pts.astype(float), R, vol.shape)).astype(int)
        a = vol[pts[:, 0], pts[:, 1], pts[:, 2]]
        b = rvol[rpts[:, 0], rpts[:, 1], rpts[:, 2]]
        if not np.allclose(a, b):
            bad += 1
    print(f"cube rotation self-test: {24 - bad}/24 consistent")
    assert bad == 0, "rotation convention is inconsistent"
    print("OK")
