"""Acquisition-physics augmentation: strong Gaussian blur and correlated additive noise.

    from datpark.augment3 import blur_noise
    xb = blur_noise(xb, M=2.5, P=0.9, voxel_mm=1.3333, generator=gen)

── PROVENANCE ──────────────────────────────────────────────────────────────────────────────
Buddenkotte & Buchert, "Unrealistic Data Augmentation Improves the Robustness of Deep
Learning-Based Classification of Dopamine Transporter SPECT Against Variability Between Sites and
Between Cameras", J Nucl Med 2024;65(9):1463. DOI 10.2967/jnumed.124.267570.

Their setup is unusually close to ours: binary normal/abnormal DaT SPECT, n=1100 training, a small
ResNet (16/32/64 filters), a 5-model ensemble, and reference-ratio intensity scaling. Results on
two independent out-of-distribution test sets, both McNemar p < 0.001:

    test set                        no aug   nnU-Net "realistic" aug   THIS (M=2.5, P=0.9)
    PPMI, lower resolution           0.960            0.972                  0.989
    MPH,  higher resolution          0.953            0.950                  0.975
    in-distribution held-out            —                —                unchanged

The middle column is the finding. **Realistic-magnitude intensity augmentation was WORSE than
nothing on one set.** The gain comes only from magnitudes far past physical plausibility — blur to
15 mm FWHM across a striatum that is itself only ~20 mm across. Their stated mechanism is
"corruption of irrelevant image features that prevents the CNN from learning these (e.g.
extracranial signal that might vary between datasets)".

── WHY THIS IS NOT THE AUGMENTATION LEVER WE ALREADY CLOSED ──────────────────
009 swept elastic deformation, mixup and random erasing: geometric, mixing and occlusion. This is
the ACQUISITION-PHYSICS family — spatial resolution and count statistics — which is the axis our
10 hospitals actually differ on, and which we have never touched. Confirmed: no blur and no
additive noise appear anywhere else in this codebase.

── ⚠ EXPECT LESS THAN THEIR HEADLINE, AND HERE IS WHY ──────────────────────────────────────
Their +2-3 points came from testing on genuinely different scanners. We checked the competition's
demonstration test set: **all 20 scans fall into acquisition groups that already exist in our
training data** (9 distinct groups, every one seen). So the competition test split is a random draw
over the SAME 10 centres, not a held-out-centre split. Our grouped CV, which holds out 2-3 whole
centres per fold, is therefore HARDER than the real test on this axis — so if anything grouped CV
over-rewards this method relative to what it will buy on the private split. The standard bar
applies; no protocol departure is warranted.

── THE RECIPE, VERBATIM IN STRUCTURE ───────────────────────────────────────────────────────
    out = a_blur * Gauss(M * FWHM_blur)[img]  +  (1 - a_blur) * img
        + a_noise * rescale_to_SD(M * SD)[ Gauss(FWHM_noise)[white noise] ]

    FWHM_blur  ~ U(1, 6) mm,  scaled by M
    FWHM_noise ~ U(5, 15) mm, NOT scaled by M
    SD         ~ U(0.10, 0.25) x (SD of the volume), scaled by M
    a_blur, a_noise ~ Bernoulli(P), drawn independently

`a_blur` is a binary switch, not a blend weight: a sample is either fully blurred or untouched.
Their grid was M in {0.5 … 3.0} x P in {0.25 … 1.0}; the optimum was **M=2.5, P=0.9**, sitting
right at the edge before in-distribution performance starts to degrade.

── IMPLEMENTATION ──────────────────────────────────────────────────────────────────────────
sigma_voxels = FWHM / 2.355 / voxel_mm. At M=2.5 that is 0.80 to 4.78 voxels on our 96^3 @1.333mm
grid — a large blur, as intended.

The sigma differs PER SAMPLE, which naively means a Python loop over the batch. Instead each 1-D
pass is done as a single grouped convolution: the batch is folded into the channel dimension and
each sample gets its own kernel row via `groups=N`. That keeps it to 3 kernel launches per batch
instead of 3N, so it does not bottleneck the dataloader.

Zero padding is deliberate. Our volumes are a brain in a mostly-empty box, so blur bleeding into
empty space is what a real scanner's point-spread function does at the field-of-view edge.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493      # 1 / (2 * sqrt(2 * ln 2))


def _kernels(sigma_vox: torch.Tensor, radius: int) -> torch.Tensor:
    """(N, 1, 2*radius+1) normalised 1-D Gaussian kernels, one per sample."""
    t = torch.arange(-radius, radius + 1, device=sigma_vox.device, dtype=torch.float32)
    k = torch.exp(-(t[None, :] ** 2) / (2.0 * sigma_vox[:, None].clamp_min(1e-6) ** 2))
    k = k / k.sum(dim=1, keepdim=True)
    return k.unsqueeze(1)


def blur3d(x: torch.Tensor, sigma_vox: torch.Tensor,
           axes: tuple[int, ...] = (0, 1, 2)) -> torch.Tensor:
    """Separable Gaussian blur with a DIFFERENT sigma per sample.

    x is (N, 1, D, H, W); sigma_vox is (N,) in voxels. Each axis is one grouped conv1d-style
    conv3d over the batch-as-channels view, so the whole batch costs len(axes) kernel launches.

    ★ `axes` selects WHICH spatial axes are blurred. The default (0, 1, 2) is isotropic 3-D and is
    the behaviour every 011 arm used — do not change it, existing results depend on it.

    `axes=(0, 1)` blurs IN-PLANE ONLY, i.e. within the axial (R-L x A-P) plane, leaving the
    superior-inferior axis untouched. That is what Buddenkotte & Buchert actually do (JNM 2024): their
    augmentation acts on a 2-D 12 mm slab, so it cannot blur across slices at all. An earlier experiment records
    why this matters — at equal nominal FWHM, isotropic 3-D blur destroys strictly more than their
    2-D blur, because it also averages across the S-I direction where the striatum is only ~10-12
    voxels deep. Our measurement that blur COSTS 0.0014 in the blend may therefore be a fact about
    our implementation rather than about blur, and this argument is the reason 011's blur arms cannot
    be read as testing their method.

    Axis mapping, from `datpark/nets3.py` (SLICE_AXIS = 4 documents S-I) and `nets2.py`
    (flip dims=[2] hits R-L): spatial axis 0 = R-L, 1 = A-P, 2 = S-I.
    """
    n = x.shape[0]
    radius = max(1, int(math.ceil(3.0 * float(sigma_vox.max().item()))))
    k = _kernels(sigma_vox.to(torch.float32), radius).to(x.dtype)     # (N, 1, K)
    out = x.reshape(1, n, *x.shape[2:])                               # fold batch into channels
    for axis in axes:
        shape = [n, 1, 1, 1, 1]
        shape[2 + axis] = k.shape[-1]
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = radius        # F.pad is W,H,D order
        out = F.conv3d(F.pad(out, pad), k.reshape(shape), groups=n)
    return out.reshape(x.shape)


def blur_noise(x: torch.Tensor, M: float, P: float, voxel_mm: float = 1.3333,
               blur: bool = True, noise: bool = True,
               generator: torch.Generator | None = None,
               blur_axes: tuple[int, ...] = (0, 1, 2)) -> torch.Tensor:
    """Buddenkotte & Buchert acquisition-physics augmentation. M=0 or P=0 is a no-op.

    `blur` / `noise` allow either component to be disabled, so the 011 sweep can attribute a
    result to one of them rather than to the pair.

    ★ `blur_axes` defaults to isotropic 3-D, which is what every 011 arm used. `(0, 1)` restricts the
    blur to the axial plane, matching what Buddenkotte & Buchert actually do on their 2-D slabs — see
    `blur3d`. The NOISE field is deliberately left 3-D in both cases: their noise is
    2-D only because their whole image is, and a spatially-correlated noise field has no reason to be
    flat along S-I in a volume.
    """
    if M <= 0.0 or P <= 0.0 or not (blur or noise):
        return x
    n, dev = x.shape[0], x.device
    rnd = lambda: torch.rand(n, device=dev, generator=generator)      # noqa: E731

    out = x
    if blur:
        fwhm = (1.0 + rnd() * 5.0) * M                               # U(1,6) mm, scaled by M
        sigma = fwhm * FWHM_TO_SIGMA / voxel_mm
        applied = (rnd() < P).to(x.dtype).view(n, 1, 1, 1, 1)         # Bernoulli(P) switch
        out = applied * blur3d(out, sigma, axes=blur_axes) + (1.0 - applied) * out

    if noise:
        fwhm_n = 5.0 + rnd() * 10.0                                  # U(5,15) mm, NOT scaled by M
        sigma_n = fwhm_n * FWHM_TO_SIGMA / voxel_mm
        target = (0.10 + rnd() * 0.15) * M                           # U(0.10,0.25) x M, of vol SD
        white = torch.randn(x.shape, device=dev, generator=generator, dtype=x.dtype)
        field = blur3d(white, sigma_n)
        # rescale each sample's smoothed field to the requested SD, expressed as a fraction of
        # that sample's own intensity SD -- so the dose is relative, not absolute
        fsd = field.flatten(1).std(dim=1).clamp_min(1e-8).view(n, 1, 1, 1, 1)
        xsd = x.flatten(1).std(dim=1).view(n, 1, 1, 1, 1)
        applied = (rnd() < P).to(x.dtype).view(n, 1, 1, 1, 1)
        out = out + applied * field * (target.view(n, 1, 1, 1, 1) * xsd / fsd)

    return out
