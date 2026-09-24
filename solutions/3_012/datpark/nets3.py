"""Third architecture: a 2.5D slice encoder with attention pooling.

**Nothing in `datpark/nets.py` or `nets2.py` is modified.** Submissions 003, 004 and
005 stay byte-for-byte reproducible; this adds a third family alongside them.

Why 2.5D and not another 3D variant. The correlation table
says the same thing three times:

    resolution variants   same representation, finer grid   corr 0.99
    siamese net           different ARCHITECTURE            corr 0.93
    SBR + LightGBM        different REPRESENTATION          corr 0.78

Changing how the network is wired bought 0.93. Changing how the scan is *presented*
bought 0.78. Since the blend's scarce resource is decorrelation and not accuracy, the third member should change the representation.

So: slice the cube into axial planes, encode each with a **shared 2D CNN**, and
aggregate across slices with **learned attention**. No 3D kernels anywhere. The
striatum spans only ~15-19 slices of 96 at 1.333 mm, so the attention head has real
work to do -- selecting which slices carry signal is itself a different inductive
bias, not a rewiring of the same one.

Two further reasons this specific design:

* It is the **bridge to the blocked lever**. If the pretrained-weight licence question
  comes back favourably, this is the same architecture with a pretrained 2D backbone
  dropped into `SliceEncoder`'s place. Building it now is not wasted either way.
* The literature favours 2D-style encoders over purpose-built 3D nets around
  n ~ 1.4k, which is where we are (n = 1362).

Expected outcome, stated up front so it can be checked: this will probably score
**worse standalone** than `sm0` (0.2694) -- the siamese net does, at 0.2740, and still
earns a blend seat. What matters is whether correlation lands **below ~0.90**. At 0.96
it is another null.

★ A discovery from the tests, which changed the design. Attention pooling is a
softmax-weighted **sum** over slices, and a sum is order-invariant -- so a naive
SliceNet is **exactly invariant to an S-I flip** (verified: the slice embeddings under
the flip are the same set, merely reversed). Two consequences, both bad:

  1. The S-I flip is our single best augmentation (+0.0255) and this
     architecture would be **immune to it** -- the whole augmentation wasted.
  2. It could not see S-I ordering at all, discarding exactly the information the
     `extent_z` SBR feature captures.

Hence `pos_embed` (default **on**): sinusoidal positional encoding is added to the
slice embeddings before attention, which restores order-sensitivity. Parameter-free and
works at any slice count. Set `pos_embed=False` to train the invariant variant
deliberately -- that is a real experiment (invariance costs accuracy but may buy
decorrelation, cf. the siamese net in), not a fallback.

The embeddings are **LayerNormed before the positional encoding is added**. Without
that they measure ~0.002 at initialisation (global average pooling over a BN+SiLU map
averages most of the variance away) against a sinusoid of ~0.64 -- a 300x mismatch that
would let position swamp content for the whole early phase of training. LayerNorm is a
per-slice statistic, so it introduces no cross-sample dependence.

Axis convention (RAS, no reorientation -- see `datpark.preprocess.load_volume`):
tensor dims are (N, C, D=R-L, H=A-P, W=S-I). Axial planes are R-L x A-P, i.e. slices
taken along the **last** dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SLICE_AXIS = 4  # tensor dim for S-I; slicing here yields axial (R-L x A-P) planes


class Conv2dBlock(nn.Module):
    """Residual 2D block, deliberately mirroring nets.ConvBlock's shape.

    Keeping the block structure familiar means the comparison against DatNet isolates
    *dimensionality and pooling*, not incidental design differences.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)
        self.skip = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(cout))
            if (stride != 1 or cin != cout) else nn.Identity()
        )

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


class SliceNet(nn.Module):
    """2.5D: shared 2D encoder over axial slices, aggregated by attention.

    Every slice of every sample passes through the *same* encoder, so the batch fed to
    the 2D layers is (N*S, 1, D, H). BatchNorm therefore normalises over slices as well
    as samples during training -- which is fine, and at eval it uses running statistics
    like every other family, so a prediction still cannot depend on its batch-mates
    (asserted in `tests/test_nets3.py`).

    Attention pooling: a small MLP scores each slice, softmax over slices, weighted
    mean of the embeddings. This is what lets the model ignore the ~80 % of slices that
    contain no striatum, and it is the part with no analogue in `DatNet`.

    ``widths`` is narrower than DatNet's because 2D convs at full in-plane resolution
    are far cheaper per channel; the default lands at ~1.6 M params so capacity stays
    comparable and architecture is the variable under test.
    """

    def __init__(self, widths=(44, 88, 176, 264), dropout: float = 0.3, in_ch: int = 1,
                 attn_dim: int = 64, stem_stride: int = 2, pos_embed: bool = True):
        super().__init__()
        self.pos_embed = pos_embed
        w0 = widths[0]
        k = 2 * stem_stride + 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w0, k, stride=stem_stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(w0), nn.SiLU(inplace=True),
        )
        blocks, cin = [], w0
        for i, w in enumerate(widths):
            blocks.append(Conv2dBlock(cin, w, stride=1 if i == 0 else 2))
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        # per-slice attention score
        self.attn = nn.Sequential(
            nn.Linear(cin, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, 1))
        # per-slice, per-sample: no cross-sample statistic, so rule-6 safe
        self.norm = nn.LayerNorm(cin)
        self.embed_dim = cin

    def _sin_pos(self, s: int, dim: int, device, dtype) -> torch.Tensor:
        """Sinusoidal encoding, (S, dim). Parameter-free, any S."""
        pos = torch.arange(s, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim, device=device, dtype=torch.float32)
        freq = torch.exp(-torch.log(torch.tensor(10000.0, device=device))
                         * (2 * (i // 2)) / dim)
        ang = pos * freq
        out = torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))
        return out.to(dtype)

    def encode_slices(self, x: torch.Tensor, with_pos: bool | None = None) -> torch.Tensor:
        """(N, 1, D, H, W) -> (N, W, C) one embedding per axial slice."""
        n, c, d, h, w = x.shape
        # move the slice axis next to batch, then fold it in
        s = x.permute(0, 4, 1, 2, 3).reshape(n * w, c, d, h)
        e = self.norm(self.pool(self.blocks(self.stem(s))).view(n, w, self.embed_dim))
        use = self.pos_embed if with_pos is None else with_pos
        if use:
            # breaks the order-invariance that pure attention pooling would impose
            e = e + self._sin_pos(w, self.embed_dim, e.device, e.dtype).unsqueeze(0)
        return e

    def forward(self, x):
        e = self.encode_slices(x)                       # (N, S, C)
        a = self.attn(e).squeeze(-1)                    # (N, S)
        a = torch.softmax(a, dim=1).unsqueeze(-1)       # (N, S, 1)
        pooled = (e * a).sum(dim=1)                     # (N, C)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def slice_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights per slice — for checking it looks at the striatum."""
        return torch.softmax(self.attn(self.encode_slices(x)).squeeze(-1), dim=1)


def build(arch: str, **kw) -> nn.Module:
    """Extends nets2.build with the 2.5D family."""
    if arch == "slicenet":
        return SliceNet(**kw)
    from datpark.nets2 import build as build2
    return build2(arch, **kw)
