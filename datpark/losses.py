"""Noise-robust objectives for binary classification with ~20 % ambiguous labels.

Motivation, and a correction to . That section swept **label smoothing** and
found it monotonically harmful: it improved the hard 10 % of scans by 29 % but cost the
easy 90 % about three times more, because a uniform target softening taxes every sample
equally. An earlier analysis then wrongly concluded the label-noise lever was closed, on the grounds that
targeting the hard scans "would need identifying them at inference, which test-sample
independence forbids".

That reasoning was wrong. These losses identify hard samples **during training, from
training labels only**. Inference is completely untouched — the shipped model is an
ordinary sigmoid classifier, every test scan is processed independently, and nothing here
appears at prediction time. What an earlier analysis actually established is narrower: *uniform* softening
fails. A **per-sample adaptive** objective is the intervention it said was needed.

All four losses reduce to plain BCE at their neutral parameter setting, which makes the
sweep a clean interpolation from the current behaviour rather than a jump.

Sign convention: all take raw logits and a float target in [0,1], and return a scalar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def bce(logit: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
    """Plain binary cross-entropy. The incumbent, for reference."""
    return F.binary_cross_entropy_with_logits(logit, target)


def gce(logit: torch.Tensor, target: torch.Tensor, q: float = 0.7, **_) -> torch.Tensor:
    """Generalised cross-entropy: (1 - p_t^q) / q.

    Zhang & Sabuncu 2018. Interpolates between BCE (q -> 0) and unhinged MAE (q = 1).
    MAE is provably robust to symmetric label noise but trains slowly; q trades the two.

    Why it should behave differently from label smoothing: the gradient magnitude here
    **decreases** as p_t falls, so a confidently-mislabelled scan contributes *less*
    rather than dominating. Smoothing instead caps confidence on every sample equally.
    """
    p = torch.sigmoid(logit)
    p_t = (target * p + (1 - target) * (1 - p)).clamp(EPS, 1.0)
    return ((1.0 - p_t.pow(q)) / q).mean()


def sce(logit: torch.Tensor, target: torch.Tensor, alpha: float = 1.0,
        beta: float = 0.5, **_) -> torch.Tensor:
    """Symmetric cross-entropy: alpha * CE(y, p) + beta * CE(p, y).

    Wang et al. 2019. The reverse term RCE is bounded and noise-tolerant, while the
    forward CE term keeps the fast convergence. beta = 0 recovers plain BCE.
    """
    p = torch.sigmoid(logit).clamp(EPS, 1 - EPS)
    t = target.clamp(EPS, 1 - EPS)
    ce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    rce = -(p * torch.log(t) + (1 - p) * torch.log(1 - t))
    return (alpha * ce + beta * rce).mean()


def bootstrap_soft(logit: torch.Tensor, target: torch.Tensor, beta: float = 0.9,
                   **_) -> torch.Tensor:
    """Soft bootstrapping: regress toward beta*label + (1-beta)*own prediction.

    Reed et al. 2015. The model's own (detached) belief is blended into the target, so a
    scan the model is confident about in the *other* direction gets its target pulled --
    a per-sample, self-adaptive softening. beta = 1 recovers plain BCE.

    This is the closest thing to "label smoothing, but only where it is warranted", which
    is precisely what the earlier per-centre breakdown implied was needed.
    """
    with torch.no_grad():
        p = torch.sigmoid(logit)
    t = beta * target + (1.0 - beta) * p
    return F.binary_cross_entropy_with_logits(logit, t)


def trimmed_bce(logit: torch.Tensor, target: torch.Tensor, drop_frac: float = 0.05,
                **_) -> torch.Tensor:
    """BCE with the worst `drop_frac` of the batch removed.

    The bluntest possible version of "ignore the scans that look mislabelled". Included
    because it is the most direct test of whether the hard core is *noise* (dropping it
    should help) or *signal* (dropping it should hurt). drop_frac = 0 recovers BCE.

    NB this is a within-batch order statistic, so it is a cross-sample computation --
    that is fine because it happens only in the TRAINING loss. It never runs at inference.
    """
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if drop_frac <= 0:
        return per.mean()
    k = max(1, int(round(len(per) * (1.0 - drop_frac))))
    keep, _ = torch.sort(per)
    return keep[:k].mean()


def focal(logit: torch.Tensor, target: torch.Tensor, gamma: float = 1.0,
          focal_normalise: bool = True, **_) -> torch.Tensor:
    """BCE with hard examples weighted UP by (1 - p_t)^gamma. gamma = 0 recovers BCE exactly.

    ── WHY THIS IS THE ONE LOSS WORTH TRYING ──────────────────────────
    Every robust loss on this project weights hard examples DOWN: gce shrinks the gradient when
    the model disagrees with the label, sce bounds it, bootstrap_soft replaces the target with
    the model's own belief, trimmed_bce deletes the worst 5 % outright. All five lost.

    And we know why. Dropping the worst 5 % of each batch costs **0.0587** — one of the largest
    effects ever measured here, on a project where most levers return 0.000. The hard core is
    SIGNAL, not label noise.

    Focal is the mirror image of that family, and the only direction on the axis never tested.

    ⚠ It does NOT follow that up-weighting must help. "Do not delete them" and "weight them more"
    are different claims, and with roughly one scan in five genuinely ambiguous (the organizers'
    own figure) focal could instead overfit the label noise inside the hard core. That is exactly
    why it is worth measuring rather than reasoning about — an earlier analysis reasoned mixup away and was right,
    but only established it once it was run.

    ── THE CONFOUND, AND WHY `focal_normalise` DEFAULTS TO TRUE ───────────────────────────
    The modulating factor (1 - p_t)^gamma is < 1 almost everywhere, so raw focal shrinks the total
    loss and therefore the gradient scale. On a fixed OneCycle schedule that is indistinguishable
    from lowering the learning rate — so a naive focal arm tests "reweighting AND a smaller LR"
    and a null result could not be attributed. Dividing by the batch-mean weight restores the
    average gradient magnitude to BCE's, leaving only the RESHAPING across samples, which is the
    hypothesis. The divisor is detached so it rescales without contributing gradient of its own.
    """
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if gamma <= 0:
        return per.mean()                      # exactly bce, so gamma=0 is a free control
    p = torch.sigmoid(logit)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    w = (1.0 - p_t).clamp_min(0.0) ** gamma
    if focal_normalise:
        w = w / w.mean().detach().clamp_min(1e-8)
    return (w * per).mean()


LOSSES = {"bce": bce, "gce": gce, "sce": sce, "bootstrap": bootstrap_soft, "focal": focal,
          "trimmed": trimmed_bce}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}")
    return LOSSES[name]
