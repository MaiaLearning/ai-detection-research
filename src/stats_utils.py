"""AUC estimation with bootstrap confidence intervals.

CLAUDE.md requires confidence intervals rather than point estimates for
every reported result, and fixed, recorded random seeds.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))


def bootstrap_auc_ci(
    y_true,
    scores,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    point = compute_auc(y, s)

    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_sample = y[idx]
        if len(np.unique(y_sample)) < 2:
            continue  # degenerate resample, can't score an AUC
        boot_aucs.append(compute_auc(y_sample, s[idx]))

    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def threshold_at_fpr(negative_scores, target_fpr: float) -> float:
    """Score threshold such that roughly target_fpr of negative_scores fall
    at or above it (i.e. the operating point for a target false positive
    rate on the negative/human class)."""
    return float(np.quantile(np.asarray(negative_scores, dtype=float), 1 - target_fpr))


def rate_at_threshold(scores, threshold: float) -> float:
    """Fraction of scores at or above threshold (TPR when scores are the
    positive/AI class, FPR when scores are the negative/human class)."""
    return float(np.mean(np.asarray(scores, dtype=float) >= threshold))


def bootstrap_paired_delta_ci(
    y_true,
    pred_a,
    pred_b,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a),
    resampling rows (not each side independently) so correlation between the
    two models' predictions on the same rows is preserved. Use this instead
    of subtracting two separately-computed CIs whenever pred_a and pred_b
    come from the same underlying rows (e.g. nested model comparison)."""
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    point = float(metric_fn(y, b) - metric_fn(y, a))

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(y[idx], b[idx]) - metric_fn(y[idx], a[idx])
        if np.isfinite(val):
            deltas.append(val)

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def partial_spearman(x, y, z) -> float:
    """Spearman correlation between x and y with z partialled out, via the
    standard partial-correlation formula applied to Spearman rhos."""
    rho_xy = spearmanr(x, y).statistic
    rho_xz = spearmanr(x, z).statistic
    rho_yz = spearmanr(y, z).statistic
    denom = np.sqrt((1 - rho_xz ** 2) * (1 - rho_yz ** 2))
    return float((rho_xy - rho_xz * rho_yz) / denom)


def bootstrap_stat_ci(
    arrays: list,
    stat_fn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for any statistic stat_fn(*arrays) that takes any number
    of same-length, row-corresponding arrays (e.g. a partial correlation's
    x, y, and control variable). Every array is resampled with the SAME
    draw of indices each iteration, so row correspondence between them is
    preserved — resampling one array but not another silently decouples
    the bootstrap distribution from the reported point estimate."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    point = float(stat_fn(*arrays))

    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = stat_fn(*(a[idx] for a in arrays))
        if np.isfinite(val):
            boot_vals.append(val)

    lo, hi = np.percentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def bootstrap_corr_ci(
    x,
    y,
    corr_fn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for a two-array correlation-like statistic corr_fn(x, y)."""
    return bootstrap_stat_ci([x, y], corr_fn, n_boot=n_boot, seed=seed, alpha=alpha)
