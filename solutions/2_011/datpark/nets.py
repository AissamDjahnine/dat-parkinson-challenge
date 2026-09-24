"""Compact 3D CNN for DaT cubes.

Sized for the data, not for a leaderboard photo: n=1362 with ~20% of labels
genuinely ambiguous. A large network would memorise the training set long before
it learned anything transferable, so this is deliberately small (~1M params) with
aggressive pooling and dropout.

Input is a 64^3 cube in multiples of the non-specific reference level (see
datpark.preprocess), so no dataset-level normalisation statistic is needed --
which also keeps the model deterministic with respect to the test set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.act = nn.SiLU(inplace=True)
        self.skip = (
            nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False),
                          nn.BatchNorm3d(cout))
            if (stride != 1 or cin != cout) else nn.Identity()
        )

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


class DatNet(nn.Module):
    """3D CNN over a DaT cube.

    ``stem_stride`` exists for the resolution sweep. Everything after the stem
    is fixed, so raising the input resolution without raising the stride would
    shrink the deepest layer's receptive field *in millimetres* — a finer-input
    run and a coarser-input run would then differ in two ways at once. Setting
    stride = voxels-per-4mm (2 at 2.0 mm, 3 at 1.33 mm, 4 at 1.0 mm) puts the
    post-stem feature map at the same physical scale in every run, so the only
    variable left is how much detail the first layer sees. The kernel grows with
    the stride (k = 2s+1) to avoid aliasing a stride-4 subsample through a 5^3
    window.

    Default stride 2 / kernel 5 is the shipped configuration and must stay
    unchanged — the 15 checkpoints in assets/cnn load against it.
    """

    def __init__(self, widths=(24, 48, 96, 160), dropout: float = 0.3, in_ch: int = 1,
                 stem_stride: int = 2):
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
        self.blocks = nn.Sequential(*blocks)          # 32 -> 16 -> 8 -> 4
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(cin, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------- augmentation
def random_affine_grid(
    n: int, device, rot_deg: float = 10.0, trans: float = 0.06,
    scale: float = 0.05, flip_lr: bool = False, generator=None,
):
    """Batch of random affine grids for grid_sample (done on GPU: it is free).

    flip_lr mirrors the Right-Left axis. Whether that is label-preserving is an
    empirical question -- asymmetric loss occurs on either side, so mirroring
    plausibly yields a valid scan, but it must be validated, not assumed.
    """
    def u(lo, hi, shape=(n,)):
        return torch.rand(shape, device=device, generator=generator) * (hi - lo) + lo

    ang = u(-rot_deg, rot_deg) * torch.pi / 180.0
    cos, sin = torch.cos(ang), torch.sin(ang)
    s = 1.0 + u(-scale, scale)

    theta = torch.zeros(n, 3, 4, device=device)
    # rotation in the axial plane (patient tilt in the scanner is mostly this)
    theta[:, 0, 0] = cos * s
    theta[:, 0, 1] = -sin * s
    theta[:, 1, 0] = sin * s
    theta[:, 1, 1] = cos * s
    theta[:, 2, 2] = s
    theta[:, 0, 3] = u(-trans, trans)
    theta[:, 1, 3] = u(-trans, trans)
    theta[:, 2, 3] = u(-trans, trans)

    if flip_lr:
        mirror = (torch.rand(n, device=device, generator=generator) < 0.5).float() * (-2) + 1
        theta[:, 0, :] *= mirror[:, None]
    return theta


def apply_affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, align_corners=False,
                                           padding_mode="zeros")
