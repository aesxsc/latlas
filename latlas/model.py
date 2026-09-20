"""The latency/distance measurement model.

Physical structure
------------------
A round trip between two points separated by great-circle distance ``d`` covers
a path of length ``L >= 2d`` -- each leg is at least the great-circle separation,
by the triangle inequality on the sphere. Signals travel at most at ``c``, so

    RTT >= 2d / c          i.e.  d <= (c/2) * RTT = 149.9 km per ms.

That inequality is the *hard* constraint and it is what :mod:`latlas.estimate`
turns into a certificate. It holds for fibre, free-space optics and satellite
alerts alike and cannot be falsified by routing.

Terrestrial paths are fibre, whose group velocity is ``v_f ~ 204300 km/s`` =
204.3 km/ms, and fibre is never laid along a great circle, so the path is
*stretched* by a detour factor. Writing the stretch on the round-trip path as
``alpha >= 1``:

    r  =  alpha * d / k  +  beta,     k = v_f / 2 = 102.15 km/ms,  beta >= 0
    log(alpha) ~ Normal(mu, sigma^2) truncated to alpha >= 1

``beta`` is the distance-independent overhead: serialisation, forwarding
lookups, host network stack. ``k`` is half the fibre speed because ``r`` is a
round trip.

Why the truncation is not cosmetic
----------------------------------
An earlier revision of this file modelled ``r = exp(nu) * (d/k + beta)`` with
unbounded ``nu``. With the fitted spread, that placed roughly a third of the
probability mass on ``alpha < 1`` -- paths *faster than fibre*, and for small
``d`` faster than light, contradicting the certificate the same code was
deriving. The likelihood must live on the same side of the light cone as the
certificate, so the detour distribution is truncated at ``alpha = 1``: a
simulation built from this model now provably satisfies the cone constraints
that the estimator applies.

Contamination
-------------
Real target lists contain anycast hosts that answer from a nearby replica and
entries whose published coordinates are simply wrong; both produce RTTs far too
short for the advertised position. A two-component mixture absorbs them: an
informative component that depends on ``d``, and a broad component that does
not. Measurements the geometry cannot explain are attributed to the second
component, so the anchor is down-weighted instead of corrupting the estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import log_ndtr

from .geo import C_KM_PER_MS, FIBER_RT_KM_PER_MS

LOG_2PI = math.log(2.0 * math.pi)

#: Fibre speed expressed as km of ROUND-TRIP distance per ms of RTT (102.15).
#: The neighbouring constant MAX_RT_KM_PER_MS = 149.90 is the vacuum bound that
#: the certificate uses; this one is tighter and therefore model-only.
K_FIBER = FIBER_RT_KM_PER_MS

#: RTT below which a target is treated as co-located, to keep the log of the
#: distance term bounded when an anchor sits inside the same city.
MIN_DISTANCE_KM = 0.5


@dataclass(frozen=True)
class DelayParams:
    """Parameters of the two-component latency/distance mixture."""

    #: Fibre speed as round-trip km per ms.
    k: float = K_FIBER
    #: Distance-independent minimum RTT in ms.
    beta: float = 0.35
    #: Mean of log(detour factor) for the truncated normal on alpha >= 1.
    mu: float = 0.26
    #: Standard deviation of log(detour factor).
    sigma: float = 0.30
    #: Prior probability a measurement is uninformative about distance.
    pi_out: float = 0.04
    #: Uninformative component in log-RTT space.
    mu_out: float = 3.6
    sigma_out: float = 1.0

    # ---- support ------------------------------------------------------

    def alpha_lower_bound(self) -> float:
        return 1.0

    def min_rtt_ms(self, d_km: np.ndarray) -> np.ndarray:
        """Smallest RTT the model allows at distance ``d`` (``alpha = 1``)."""
        d = np.asarray(d_km, dtype=np.float64)
        return np.maximum(d, MIN_DISTANCE_KM) / self.k + self.beta

    def implied_distance_km(self, r_ms: np.ndarray) -> np.ndarray:
        """Median distance consistent with an observed RTT."""
        r = np.asarray(r_ms, dtype=np.float64)
        excess = r - self.beta
        out = np.where(excess > 0, np.maximum(excess, 0.0) * self.k * math.exp(-self.mu),
                       np.nan)
        return out

    # ---- densities ----------------------------------------------------

    def inlier_logpdf(self, d_km: np.ndarray, r_ms: np.ndarray) -> np.ndarray:
        """Log density of ``r_ms`` under the informative component.

        Zero probability mass below ``d/k + beta``, which is exactly the fibre
        light-cone floor; measurements below it can only be explained by the
        uninformative component.
        """
        d = np.maximum(np.asarray(d_km, dtype=np.float64), MIN_DISTANCE_KM)
        r = np.asarray(r_ms, dtype=np.float64)
        excess = r - self.beta
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.log(np.where(excess > 0, excess, np.nan)) - np.log(d / self.k)
        z = (t - self.mu) / self.sigma
        logphi = -0.5 * z * z - 0.5 * LOG_2PI
        # Truncation normaliser: 1 - Phi((0 - mu)/sigma) = Phi(mu/sigma).
        log_norm = -math.log(self.sigma) - float(log_ndtr(self.mu / self.sigma))
        out = logphi + log_norm - np.log(np.where(excess > 0, excess, np.nan))
        out = np.where(np.isfinite(t) & (t >= 0.0), out, -np.inf)
        return out

    def outlier_logpdf(self, r_ms: np.ndarray) -> np.ndarray:
        r = np.asarray(r_ms, dtype=np.float64)
        y = np.log(np.maximum(r, 1e-3))
        z = (y - self.mu_out) / self.sigma_out
        return -0.5 * z * z - math.log(self.sigma_out) - 0.5 * LOG_2PI

    def logpdf(self, d_km: np.ndarray, r_ms: np.ndarray) -> np.ndarray:
        """Log density of the two-component mixture; broadcasting applies."""
        li = math.log(max(1e-12, 1.0 - self.pi_out)) + self.inlier_logpdf(d_km, r_ms)
        lo = math.log(max(1e-12, self.pi_out)) + self.outlier_logpdf(r_ms)
        return np.logaddexp(li, lo)

    def responsibility(self, d_km: np.ndarray, r_ms: np.ndarray) -> np.ndarray:
        """P(measurement is informative | r, d).

        Near-zero values flag anchors whose RTT is incompatible with their
        claimed position: the signature of an anycast replica or a bad
        coordinate. Reported rather than silently dropped.
        """
        li = math.log(max(1e-12, 1.0 - self.pi_out)) + self.inlier_logpdf(d_km, r_ms)
        lo = math.log(max(1e-12, self.pi_out)) + self.outlier_logpdf(r_ms)
        m = np.maximum(li, lo)
        pi = np.exp(np.where(np.isfinite(li), li - m, -np.inf))
        po = np.exp(lo - m)
        denom = pi + po
        return np.where(denom > 0, pi / np.maximum(denom, 1e-300), 0.0)

    def as_dict(self) -> dict:
        return {
            "k_km_per_ms": self.k,
            "beta_ms": round(self.beta, 4),
            "mu_log_detour": round(self.mu, 4),
            "sigma_log_detour": round(self.sigma, 4),
            "implied_median_detour": round(math.exp(self.mu), 4),
            "pi_outlier": round(self.pi_out, 4),
            "mu_out_log_rtt": round(self.mu_out, 4),
            "sigma_out_log_rtt": round(self.sigma_out, 4),
            "fiber_cone_km_per_ms": round(self.k, 3),
            "vacuum_cone_km_per_ms": round(C_KM_PER_MS / 2.0, 3),
        }


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_outlier_component(r_ms: np.ndarray, params: DelayParams,
                          quantile: float = 0.9) -> DelayParams:
    """Anchor the uninformative component in the upper tail of observed RTTs.

    The informative component already explains most measurements; the broad
    component exists to absorb the unexplained remainder, so it is centred on the
    slow tail rather than on the bulk, which keeps the two components from
    competing for the same observations.
    """
    y = np.log(np.maximum(np.asarray(r_ms, dtype=np.float64), 1e-3))
    if y.size == 0:
        return params
    hi = y[y >= np.quantile(y, quantile)]
    if hi.size < 3:
        hi = y
    return replace(params, mu_out=float(hi.mean()),
                   sigma_out=float(max(0.6, hi.std(ddof=0))))


def fit_params(d_km: np.ndarray, r_ms: np.ndarray, *,
               beta_grid: np.ndarray | None = None,
               mu_grid: np.ndarray | None = None,
               sigma_grid: np.ndarray | None = None,
               base: DelayParams | None = None) -> DelayParams:
    """Maximum-likelihood fit of ``(beta, mu, sigma)`` for known distances.

    A coarse grid search rather than a gradient method: the truncated normal has
    a discontinuity in its support, the objective is cheap to vectorise over a
    few thousand anchors, and a grid cannot fail to converge or land in a local
    optimum of a nearly flat ridge (``mu`` and ``sigma`` trade off strongly).
    Coarse-to-fine refinement gives about four significant digits for the cost
    of a few hundred vectorised evaluations.
    """
    d = np.asarray(d_km, dtype=np.float64)
    r = np.asarray(r_ms, dtype=np.float64)
    base = base or DelayParams()
    if d.size < 4 or np.all(r <= 0):
        return base

    beta_grid = beta_grid if beta_grid is not None else np.concatenate(
        [[0.0], np.linspace(0.02, 1.2, 14)])
    mu_grid = mu_grid if mu_grid is not None else np.linspace(0.0, 0.9, 19)
    sigma_grid = sigma_grid if sigma_grid is not None else np.linspace(0.06, 0.9, 18)

    obs_weight = 1.0 - base.pi_out
    lo_const = math.log(max(1e-12, base.pi_out))
    best = None
    for round_idx in range(2):
        for beta in beta_grid:
            p = replace(base, beta=float(beta))
            excess = r - p.beta
            if not np.any(excess > 0):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (np.log(np.where(excess > 0, excess, np.nan))
                     - np.log(np.maximum(d, MIN_DISTANCE_KM) / p.k))
            valid = np.isfinite(t) & (t >= 0.0)
            lo = lo_const + p.outlier_logpdf(r)
            if not valid.any():
                continue
            tv = t[valid]
            rv = r[valid]
            for mu in mu_grid:
                for sigma in sigma_grid:
                    if sigma <= 0:
                        continue
                    z = (tv - mu) / sigma
                    li = (math.log(obs_weight) - 0.5 * z * z - math.log(sigma)
                          - 0.5 * LOG_2PI - float(log_ndtr(mu / sigma))
                          - np.log(rv - beta))
                    total = float(np.logaddexp(li, lo[valid]).sum())
                    # Out-of-support observations must still be paid for.
                    total += float(lo[~valid].sum())
                    if best is None or total > best[0]:
                        best = (total, float(beta), float(mu), float(sigma))
        if best is None:
            break
        # Refine around the winner.
        _, b0, m0, s0 = best
        span_b = max(1e-3, (beta_grid[-1] - beta_grid[0]) / 8.0)
        beta_grid = np.linspace(max(0.0, b0 - span_b), b0 + span_b, 7)
        mu_grid = np.linspace(max(0.0, m0 - 0.12), m0 + 0.12, 9)
        sigma_grid = np.linspace(max(0.02, s0 - 0.10), s0 + 0.10, 9)

    if best is None:
        return base
    _, beta, mu, sigma = best
    return replace(base, beta=beta, mu=mu, sigma=sigma)


def loss_probability(d_km: np.ndarray, lambda0: float = 0.05,
                     d_scale_km: float = 6000.0, max_extra: float = 0.35):
    """Probability an echo to a host at distance ``d`` is dropped.

    A floor (policy and topology drops that ignore distance) plus a saturating
    distance term. Weak evidence, used only as a mild likelihood factor.
    """
    d = np.asarray(d_km, dtype=np.float64)
    frac = 1.0 - np.exp(-np.maximum(d, 0.0) / d_scale_km)
    return lambda0 + (1.0 - lambda0) * max_extra * frac
