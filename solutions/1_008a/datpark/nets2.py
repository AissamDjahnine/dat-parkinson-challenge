"""Second-generation nets and a corrected affine augmentation.

**Nothing in `datpark/nets.py` is modified.** That module and the 15 shipped
checkpoints stay byte-identical so submission 003 remains reproducible; this file
adds new pieces alongside it and imports what it reuses.

Two things live here:

1. `random_affine_grid_anat` — an affine-grid builder whose axis semantics are
   actually anatomical. `nets.random_affine_grid` documents "flip_lr mirrors the
   Right-Left axis" and "rotation in the axial plane", but does neither, because
   `torch.nn.functional.affine_grid` on a 5D input orders grid coordinates
   ``(x, y, z)`` with **x indexing the LAST spatial dim**. Our cubes are RAS with
   no reorientation (see `datpark.preprocess.load_volume`), so:

       tensor dim 2 = array axis 0 = R-L   <- grid coord z, theta row 2
       tensor dim 3 = array axis 1 = A-P   <- grid coord y, theta row 1
       tensor dim 4 = array axis 2 = S-I   <- grid coord x, theta row 0

   The original writes the flip into row 0 and the rotation into rows 0/1, so it
   mirrors **S-I** (an anatomically impossible upside-down brain) and rotates in
   the **sagittal** plane. Verified empirically with marker volumes, not inferred
   from docs -- see `scripts/verify_aug_axes.py`.

   Note the *mirror TTA* is unaffected: `torch.flip(x, dims=[2])` does hit R-L, so
   test-time mirroring was always correct, and inference never calls this function.

2. `SiameseDatNet` — a structurally different architecture for the diversity
   experiment. Three resolutions of `DatNet` correlate at
   0.99+, so more of the same shape is a dead end. This one encodes the
   bilateral comparison a DaT reader actually performs: split the cube at the
   midline, encode both hemispheres with **shared** weights, and fuse them through
   ``[sum, |difference|]``. Striatal asymmetry stops being something the network
   must discover and becomes part of its structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from datpark.nets import ConvBlock  # reused unchanged

# Which theta row drives which anatomical axis, given affine_grid's (x,y,z) order.
ROW_SI, ROW_AP, ROW_RL = 0, 1, 2


# ------------------------------------------------------- corrected augmentation
def random_affine_grid_anat(
    n: int, device, rot_deg: float = 10.0, trans: float = 0.06,
    scale: float = 0.05, flip_lr: bool = False, generator=None,
    rot_plane: str = "axial",
):
    """Affine grids with correct anatomical axis semantics.

    Drop-in replacement for `nets.random_affine_grid` with identical signature and
    identical distributions -- only the axis assignment differs.

    ``flip_lr`` mirrors **R-L** (theta row 2). ``rot_plane`` selects the rotation
    plane: ``"axial"`` rotates about the S-I axis (yaw -- patient rotation in the
    scanner, the intended behaviour), ``"sagittal"`` reproduces the original's
    rotation about R-L (pitch) for a controlled comparison.
    """
    def u(lo, hi, shape=(n,)):
        return torch.rand(shape, device=device, generator=generator) * (hi - lo) + lo

    ang = u(-rot_deg, rot_deg) * torch.pi / 180.0
    cos, sin = torch.cos(ang), torch.sin(ang)
    s = 1.0 + u(-scale, scale)

    theta = torch.zeros(n, 3, 4, device=device)
    # isotropic scale on the diagonal first, then overwrite the rotating pair
    theta[:, ROW_SI, ROW_SI] = s
    theta[:, ROW_AP, ROW_AP] = s
    theta[:, ROW_RL, ROW_RL] = s

    # rotate within the plane spanned by the two axes that are NOT the rotation axis
    a, b = (ROW_RL, ROW_AP) if rot_plane == "axial" else (ROW_AP, ROW_SI)
    theta[:, a, a] = cos * s
    theta[:, a, b] = -sin * s
    theta[:, b, a] = sin * s
    theta[:, b, b] = cos * s

    theta[:, 0, 3] = u(-trans, trans)
    theta[:, 1, 3] = u(-trans, trans)
    theta[:, 2, 3] = u(-trans, trans)

    if flip_lr:
        mirror = (torch.rand(n, device=device, generator=generator) < 0.5).float() * (-2) + 1
        theta[:, ROW_RL, :] *= mirror[:, None]
    return theta


# ------------------------------------------------------- siamese hemisphere net
class SiameseDatNet(nn.Module):
    """Shared-weight hemisphere encoder fused by ``[sum, |difference|]``.

    The cube's R-L axis is spatial dim 0 (tensor dim 2). The two halves are split
    there and the right half is mirrored so both enter the encoder in the same
    canonical orientation; the encoder therefore only ever sees "a hemisphere",
    and its weights are shared across both.

    Fusion by sum and absolute difference makes the network **exactly invariant to
    an L<->R swap by construction** rather than approximately invariant through
    augmentation. Two consequences worth knowing:

    - Mirror TTA on the R-L axis is a mathematical no-op here. Harmless, but it
      buys nothing, so do not read a TTA gain into this arm.
    - R-L flip augmentation is likewise a no-op. This arm's effective augmentation
      is therefore weaker than `DatNet`'s by one transform; that is inherent to the
      architecture, not a configuration difference.

    Widths match `DatNet` so capacity stays comparable (~1.66 M vs ~1.64 M) and
    architecture is the variable under test, not parameter count.
    """

    def __init__(self, widths=(24, 48, 96, 160), dropout: float = 0.3, in_ch: int = 1,
                 stem_stride: int = 2, hidden: int = 96):
        super().__init__()
        w0 = widths[0]
        k = 2 * stem_stride + 1
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, w0, k, stride=stem_stride, padding=k // 2, bias=False),
            nn.BatchNorm3d(w0), nn.SiLU(inplace=True),
        )
        blocks, cin = [], w0
        for i, w in enumerate(widths):
            blocks.append(ConvBlock(cin, w, stride=1 if i == 0 else 2))
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * cin, hidden), nn.SiLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def encode(self, half: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(half)))

    def forward(self, x):
        half = x.shape[2] // 2
        left = x[:, :, :half]
        right = torch.flip(x[:, :, x.shape[2] - half:], dims=[2])  # canonicalise
        fl, fr = self.encode(left), self.encode(right)
        fused = torch.cat([fl + fr, (fl - fr).abs()], dim=1)
        return self.head(fused).squeeze(-1)


ARCHS = {"datnet": None, "siamese": SiameseDatNet}  # datnet resolved lazily from nets.py


def build(arch: str, **kw) -> nn.Module:
    """Construct an architecture by name. ``datnet`` is the unmodified original."""
    if arch == "datnet":
        from datpark.nets import DatNet
        return DatNet(**kw)
    if arch == "siamese":
        return SiameseDatNet(**kw)
    raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
