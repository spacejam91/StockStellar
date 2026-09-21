"""Numeric helpers with no SciPy dependency.

SciPy is ~100MB installed and this runs on an 8GB machine, so the one function
we actually need from it — the inverse standard normal CDF — is implemented
here instead. Accuracy is ~1.15e-9 absolute, which is many orders of magnitude
tighter than anything in the scoring path cares about.
"""

from __future__ import annotations

import numpy as np

# Acklam's rational approximation coefficients.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)

_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_ppf(p: np.ndarray | float) -> np.ndarray:
    """Inverse standard normal CDF, vectorised. Equivalent to scipy.stats.norm.ppf.

    Inputs outside (0, 1) return NaN rather than +/-inf — callers in this package
    always pass (rank - 0.5)/n, which is open on both ends by construction, so a
    NaN here means a bug upstream and should be loud rather than silently infinite.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)

    lo = (p > 0) & (p < _P_LOW)
    mid = (p >= _P_LOW) & (p <= _P_HIGH)
    hi = (p > _P_HIGH) & (p < 1)

    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
                   / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
                    / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1))
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
                    / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))

    return out if out.ndim else float(out)


def rank_to_z(values: np.ndarray, ascending: bool = True) -> np.ndarray:
    """Cross-sectional rank -> z, via percentile = (rank - 0.5)/n.

    The -0.5 is what keeps the percentile off 0 and 1, so norm_ppf never sees an
    endpoint. NaNs stay NaN and are excluded from n, so a name missing an input
    does not shift everyone else's rank.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = ~np.isnan(v)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.0
        return out

    x = v[ok] if ascending else -v[ok]
    # average ranks for ties, 1-based
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # resolve ties to their mean rank so equal inputs get equal scores
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(uniq) != n:
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]

    out[ok] = norm_ppf((ranks - 0.5) / n)
    return out


def max_attainable_z(n: int) -> float:
    """Ceiling on composite_z for a universe of n names.

    A selection threshold above this returns nothing forever and looks exactly
    like a quiet market, which is the failure mode worth catching loudly.
    """
    if n < 2:
        return float("nan")
    return float(norm_ppf(np.array([(n - 0.5) / n]))[0])
