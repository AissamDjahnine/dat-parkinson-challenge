"""3D ResNet encoders with MedicalNet (Med3D) pretrained weights, in pure PyTorch.

    from datpark.med3d import build_med3d
    model = build_med3d(depth=10, weights="artifacts/pretrained/resnet_10_23dataset.pth")

WHY THIS EXISTS. Every lever on the training recipe is closed, and the only remaining route to a better blend is a
better *single* model. A pretrained initialisation is the one untried source of information
that is not more data: MedicalNet's encoders were trained on 23 public 3D medical datasets, so
their early filters have seen far more 3D anatomy than 1362 SPECT scans can teach.

⚠ THE PRIOR IS WEAK, AND HONESTY DEMANDS SAYING SO. Everything measured on the
initialisation/parameterisation side of this project came back at exactly 0.000 -- weight
decay, dropout, learning rate, capacity, extra seeds -- while data-side changes carried
everything. And the *frozen* ImageNet probe scored 0.4080 standalone for +0.0006 blended. This is worth one clean experiment, not a campaign.

── WHY NOT MONAI ─────────────────────────────────────────────────────────────────────────────
The submission container has **no network access and no MONAI**, and adding a dependency needs
a request to the organizers with lead time (SUBMISSION.md). So the architecture is
reimplemented here in plain torch: 72 tensors of a standard 3D ResNet, nothing exotic. That
also means what ships is a state_dict we already hold, with no import-time downloads.

── LICENCE, because prize eligibility depends on it ─────────────────────────────────────────
MedicalNet is **MIT** (Tencent / THL A29 Limited, 2019), which permits commercial use of the
resulting model -- the bar CONTEXT.md sets for prize eligibility. Note GitHub's licence API
reports NOASSERTION for the repo because its LICENSE file opens with a Tencent preamble before
the MIT text; the text itself is unambiguous MIT, and the HuggingFace mirror declares `mit`.

★ ANSWERED BY THE ORGANIZERS, 2026-07-31 (email, `licence_question_email.md`). Both of the
warnings that used to be here were WRONG and are corrected:

  1. NO PRE-SUBMISSION DISCLOSURE IS NEEDED. "Once the competition closes, we will reach out to
     the winners to collect information, including information on any external data or models
     used. Until then, there is no need to disclose this."
  2. The original `resnet_*.pth` does NOT need to ship. Only the FINE-TUNED state_dict is
     required, and it is self-contained. (This file previously claimed otherwise.)

They also settled the question that looked most likely to sink this: "Licensing of the data
used to train external models would not matter unless the resulting work has a different
license (e.g., CC BY-SA)." So the 23 unenumerated upstream datasets are irrelevant given the
MIT licence on the distributed weights.

Residual risk to keep in view, not hidden: the weights were *trained on* 23 datasets with
their own licences. Tencent distributes the resulting model under MIT, which is the licence we
rely on, but this is the same class of question as the pending ImageNet one. Ask the forum
before submitting, not after.

── ARCHITECTURE, verified against the checkpoint rather than the paper ──────────────────────
`resnet_10_23dataset.pth` holds 72 tensors / 14,361,292 params under a `module.` prefix
(DataParallel), and contains **no classifier** -- MedicalNet published segmentation encoders,
so the head is ours to add and nothing is discarded on load. Layout read off the keys:

    conv1   7x7x7, 1 -> 64            layer1  BasicBlock x1   64
    bn1                               layer2  BasicBlock x1  128, downsample 1x1x1 conv+BN
    maxpool 3x3x3                     layer3  BasicBlock x1  256, downsample
                                      layer4  BasicBlock x1  512, downsample

`downsample.0.weight` being 1x1x1 fixes shortcut type **B**. Strides and dilations are NOT
encoded in weight shapes, so they are free parameters here: MedicalNet used stride 1 +
dilation in layer3/4 for dense segmentation output, while this classifier halves at every
stage (96 -> 48 -> 24 -> 24 -> 12 -> 6 -> 3, then global average pool).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

DEPTHS = {10: (1, 1, 1, 1), 18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}
WIDTHS = (64, 128, 256, 512)

# ⚠ THE SHORTCUT TYPE IS NOT THE SAME FOR EVERY CHECKPOINT, and getting it wrong makes the
# weights unloadable. MedicalNet's own model card states the settings each file was trained
# with, and they differ by depth:
#
#     resnet_10_23dataset.pth  --resnet_shortcut B      projection: 1x1x1 conv + BN
#     resnet_18_23dataset.pth  --resnet_shortcut A      parameter-free: pool + zero-pad
#     resnet_34_23dataset.pth  --resnet_shortcut A
#
# Verified against the files rather than trusted: r10 has 72 tensors including
# `layer2.0.downsample.0.weight`; r18 has 102 tensors and **no downsample keys at all**.
DEPTH_SHORTCUT = {10: "B", 18: "A", 34: "A"}


class BasicBlock3d(nn.Module):
    """Two 3x3x3 convs plus a shortcut, in MedicalNet's two flavours.

    B -- a learned 1x1x1 projection with BatchNorm (extra tensors in the checkpoint).
    A -- the original ResNet trick: subsample spatially, then pad the new channels with
         zeros. Carries no parameters, which is why an r18 checkpoint has none.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1, shortcut: str = "B"):
        super().__init__()
        if shortcut not in ("A", "B"):
            raise ValueError(f"shortcut must be 'A' or 'B', got {shortcut!r}")
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut
        self.stride = stride
        self.cout = cout
        self.changes_shape = stride != 1 or cin != cout
        # Present only for type B, which is exactly when the checkpoint has the tensors.
        self.downsample = None
        if self.changes_shape and shortcut == "B":
            self.downsample = nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm3d(cout))

    def _identity(self, x):
        if not self.changes_shape:
            return x
        if self.downsample is not None:
            return self.downsample(x)
        out = nn.functional.avg_pool3d(x, kernel_size=1, stride=self.stride)
        pad = torch.zeros(out.shape[0], self.cout - out.shape[1], *out.shape[2:],
                          device=out.device, dtype=out.dtype)
        return torch.cat([out, pad], dim=1)

    def forward(self, x):
        idt = self._identity(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class MedResNet3d(nn.Module):
    """MedicalNet encoder + a fresh scalar classification head.

    Returns logits of shape (N,), matching `datpark.nets.DatNet` so the training loop, the
    loss functions and the TTA code are all reused unchanged.
    """

    def __init__(self, depth: int = 10, dropout: float = 0.3, in_ch: int = 1,
                 maxpool: bool = True, shortcut: str | None = None):
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError(f"depth must be one of {sorted(DEPTHS)}, got {depth}")
        blocks = DEPTHS[depth]
        self.shortcut = shortcut or DEPTH_SHORTCUT[depth]
        self.conv1 = nn.Conv3d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1) if maxpool else nn.Identity()
        cin = 64
        for i, (n, cout) in enumerate(zip(blocks, WIDTHS), start=1):
            stride = 1 if i == 1 else 2
            layer = nn.Sequential(*[
                BasicBlock3d(cin if j == 0 else cout, cout, stride if j == 0 else 1,
                             shortcut=self.shortcut)
                for j in range(n)])
            setattr(self, f"layer{i}", layer)
            cin = cout
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(cin, 1))

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
        return x

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def set_train_mode(model: MedResNet3d) -> None:
    """`model.train()`, but frozen submodules stay in eval — call this instead of `.train()`.

    ⚠ THE BUG THIS FIXES, which was live until it was found. `p.requires_grad_(False)` stops
    gradients and nothing else. A frozen BatchNorm in train() mode still overwrites its running
    mean and variance from each batch, so a "frozen" encoder keeps mutating every epoch and the
    pretrained statistics are gradually replaced by SPECT statistics. The linear-probe arm would
    then not have been a linear probe, and the freeze curve would have measured nothing clean.

    Freezing means two things and both are needed: no gradient AND no running-stat update.
    """
    model.train()
    for mod in getattr(model, "frozen_modules", []):
        mod.eval()


def standardise_per_scan(X: torch.Tensor, eps: float = 1e-5, chunk: int = 64) -> torch.Tensor:
    """Z-score every scan by its OWN mean and std — the input form MedicalNet was trained on.

    ★ WHY THIS EXISTS, and why omitting it would have produced a fake null. Our cubes are
    divided by each scan's non-specific-binding reference level, so values live in roughly
    [0, 12] with a mean near 1. MedicalNet's own pipeline z-scores each volume to zero mean and
    unit variance. Handing a pretrained first conv an input whose scale is ~10x what it was
    trained on can destroy transfer for a reason that has nothing to do with whether medical
    pretraining is useful — and the experiment would have reported "pretraining does not
    transfer" while never having tested it fairly.

    Statistics are PER SCAN, computed from that scan alone, so rule 6 (test-sample
    independence) holds exactly: no scan's normalisation can shift because of its neighbours.
    That is also why a dataset-wide mean/std is deliberately NOT used here.

    ⚠ CHUNKED FOR A REASON, not for tidiness. The naive one-liner (`X.float()` on the whole
    tensor, then subtract) peaks at ~11 GB for the real (1362, 1, 96, 96, 96) cache: 2.2 GB of
    float16 plus two 4.5 GB float32 temporaries. This box has 31 GB with ~16 GB free while
    another project trains, so the one-shot version was a plausible 3 a.m. OOM kill that would
    have taken out one arm of an overnight run. At chunk=64 the peak temporary is ~230 MB.
    """
    out = torch.empty_like(X)
    for i in range(0, len(X), chunk):
        blk = X[i : i + chunk].float()
        flat = blk.reshape(blk.shape[0], -1)
        shape = (-1, *([1] * (X.ndim - 1)))
        mean = flat.mean(dim=1).reshape(shape)
        std = flat.std(dim=1).reshape(shape)
        out[i : i + chunk] = ((blk - mean) / (std + eps)).to(X.dtype)
    return out


def load_medicalnet(model: MedResNet3d, weights: str | Path) -> dict:
    """Load MedicalNet weights into `model`, accounting for EVERY tensor.

    ⚠ THE FAILURE THIS GUARDS AGAINST. `load_state_dict(..., strict=False)` silently
    tolerates a total key mismatch, so a renamed layer would leave the "pretrained" model
    randomly initialised and the experiment would report a null that means nothing. Any
    encoder tensor that fails to load raises here instead. Only the head may be missing --
    the checkpoint is a segmentation encoder and has no classifier.
    """
    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    own = model.state_dict()
    loadable, skipped = {}, {}
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped[k] = tuple(v.shape)
    if skipped:
        raise RuntimeError(f"checkpoint tensors that do not fit the model: {skipped}")

    missing = [k for k in own if k not in loadable]
    unexpected = [k for k in missing if not k.startswith("head.")]
    if unexpected:
        raise RuntimeError(f"encoder tensors NOT found in the checkpoint: {unexpected}")

    model.load_state_dict(loadable, strict=False)
    enc = sum(v.numel() for v in loadable.values())
    return {"loaded_tensors": len(loadable), "loaded_params": enc,
            "fresh_tensors": sorted(missing), "checkpoint": str(weights)}


def build_med3d(depth: int = 10, weights: str | Path | None = None, dropout: float = 0.3,
                maxpool: bool = True, freeze_until: str = "none", verbose: bool = True,
                shortcut: str | None = None):
    """Build the model; `weights=None` gives the random-init CONTROL for the A/B.

    The control is not optional. Without it a win could just mean "a 14M-param ResNet beats
    our 1.6M DatNet" and a loss could just mean "this architecture is wrong for SPECT" --
    neither of which is a statement about pretraining. Same architecture, same schedule, only
    the initialisation differs.
    """
    model = MedResNet3d(depth=depth, dropout=dropout, maxpool=maxpool, shortcut=shortcut)
    info = {"pretrained": False, "depth": depth, "shortcut": model.shortcut,
            "params": sum(p.numel() for p in model.parameters())}
    if weights is not None:
        info.update(load_medicalnet(model, weights))
        info["pretrained"] = True

    frozen = 0
    model.frozen_modules = []
    if freeze_until != "none":
        order = ["stem", "layer1", "layer2", "layer3", "layer4"]
        if freeze_until not in order:
            raise ValueError(f"freeze_until must be 'none' or one of {order}")
        stop = order.index(freeze_until)
        groups = {"stem": [model.conv1, model.bn1],
                  **{f"layer{i}": [getattr(model, f'layer{i}')] for i in range(1, 5)}}
        for name in order[: stop + 1]:
            for mod in groups[name]:
                for p in mod.parameters():
                    p.requires_grad_(False)
                    frozen += p.numel()
                model.frozen_modules.append(mod)
    info["frozen_params"] = frozen
    info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  med3d r{depth}  params {info['params']:,}  "
              f"pretrained={info['pretrained']}  frozen={frozen:,}  "
              f"trainable={info['trainable_params']:,}", flush=True)
    return model, info
