"""Latency-only position estimation.

Two independent sources of information are combined.

**Certificate (model-free).** Each anchor ``i`` at known position ``a_i`` that
answered with a minimum round-trip time ``r_i`` implies, by the constancy of the
speed of light in any medium,

    d(x, a_i) <= (c/2) * r_i        for the true client position x.

The intersection of those spherical caps provably contains ``x`` with no
statistical assumptions at all. This is the honest uncertainty statement, and it
is also what makes the search space finite: the single smallest observed RTT
already confines the client to one cap.

**Likelihood (model-based).** Inside the certificate region, the delay model of
:mod:`latlas.model` sharpens the estimate. The two components are reported
separately so a reader can see how much of the precision comes from physics and
how much from the fitted network model.

Anchors are allowed to be wrong. Some fraction of any large public target list
answers from an anycast replica or carries coordinates that simply do not match
the box serving the traffic; those anchors generate constraints that exclude the
true location. The certificate is therefore computed as an intersection with
tolerance -- the region where *all but a small fraction* of constraints hold --
which is the standard robust form of a feasibility problem and stays valid as
long as fewer than that fraction of anchors are bad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .geo import (EARTH_RADIUS_KM, MAX_RT_KM_PER_MS, cap_points,
                  great_circle_km, latlon_to_unit, spherical_cap_area_km2,
                  unit_to_latlon)
from .measure import AnchorMeasurement
from .model import (DelayParams, fit_outlier_component, fit_params,
                    loss_probability)
from .calibrate import load_params as load_calibrated_params

#: Maximum ground distance covered by one millisecond of round-trip time.
#: Uses vacuum light speed, so it holds for fibre, free-space optics and
#: satellite relays alike. The fibre figure (~102 km/ms) would be ~36% tighter
#: and is deliberately NOT used for the certificate.
MAX_KM_PER_MS = MAX_RT_KM_PER_MS

#: Slack added to each measured RTT before forming its constraint, in ms. The
#: Windows ICMP backend reports whole milliseconds; if it rounds rather than
#: truncates this is slack, if it truncates this is exactly the correction.
QUANTUM_MS = 1.0


# --------------------------------------------------------------------------
# Problem setup
# --------------------------------------------------------------------------


@dataclass
class AnchorObservation:
    """One anchor reduced to what the estimator consumes."""

    key: str
    lat: float
    lon: float
    rtt_ms: float
    loss: float
    source: str
    city: str = ""
    country: str = ""
    n_samples: int = 0

    @property
    def unit(self) -> np.ndarray:
        return latlon_to_unit(self.lat, self.lon)

    @property
    def radius_km(self) -> float:
        """Rigorous upper bound on the client's distance from this anchor."""
        return MAX_KM_PER_MS * (self.rtt_ms + QUANTUM_MS)


def observations_from_measurements(
        measurements: Iterable[AnchorMeasurement],
        floor_q: float = 0.15) -> list[AnchorObservation]:
    """Reduce each measurement series to a floor RTT, dropping silent anchors."""
    out: list[AnchorObservation] = []
    for m in measurements:
        r = m.floor_ms(floor_q)
        if r is None:
            continue
        out.append(AnchorObservation(
            key=m.key, lat=m.lat, lon=m.lon, rtt_ms=float(r), loss=float(m.loss),
            source=m.source, city=m.city, country=m.country, n_samples=m.sent,
        ))
    return out


@dataclass
class Constraints:
    """Vectorised view of the anchor set for evaluation."""

    keys: list[str]
    units: np.ndarray          # (N, 3)
    rtt_ms: np.ndarray         # (N,)
    radius_km: np.ndarray      # (N,)
    loss: np.ndarray           # (N,)
    lats: np.ndarray
    lons: np.ndarray
    sources: list[str]

    def __len__(self) -> int:
        return len(self.keys)


def build_constraints(obs: Sequence[AnchorObservation]) -> Constraints:
    return Constraints(
        keys=[o.key for o in obs],
        units=np.stack([o.unit for o in obs], axis=0),
        rtt_ms=np.array([o.rtt_ms for o in obs], dtype=np.float64),
        radius_km=np.array([o.radius_km for o in obs], dtype=np.float64),
        loss=np.array([o.loss for o in obs], dtype=np.float64),
        lats=np.array([o.lat for o in obs], dtype=np.float64),
        lons=np.array([o.lon for o in obs], dtype=np.float64),
        sources=[o.source for o in obs],
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass
class Evaluation:
    """Per-candidate evaluation results."""

    violation_frac: np.ndarray   # (P,) fraction of constraints violated
    violation_count: np.ndarray  # (P,) number of constraints violated
    violation_excess_km: np.ndarray  # (P,) total km by which constraints are exceeded
    loglik: np.ndarray           # (P,) mixture log-likelihood
    mean_responsibility: np.ndarray  # (P,) mean inlier responsibility


def evaluate(constraints: Constraints, points: np.ndarray, params: DelayParams,
             chunk: int = 4096, relevant: np.ndarray | None = None,
             use_loss: bool = False) -> Evaluation:
    """Evaluate cone violations and mixture likelihood at every candidate point.

    Two model-free violation measures are accumulated. The integer count is the
    primary robust criterion -- how many anchors does this location contradict --
    and the summed excess in kilometres breaks ties continuously, so the search
    has a real gradient even when the count is flat. Both are pure geometry and
    involve no fitted parameters, which is what makes them trustworthy when the
    statistical model is not yet calibrated.

    Distances are computed in chunks so the (points x anchors) matrix never has
    to exist in full; at a few hundred thousand candidates by a couple of
    thousand anchors that matrix would be several gigabytes.
    """
    n_pts = len(points)
    viol = np.zeros(n_pts, dtype=np.float64)
    count = np.zeros(n_pts, dtype=np.float64)
    excess = np.zeros(n_pts, dtype=np.float64)
    ll = np.zeros(n_pts, dtype=np.float64)
    resp = np.zeros(n_pts, dtype=np.float64)

    sub_units = constraints.units if relevant is None else constraints.units[relevant]
    sub_radius = constraints.radius_km if relevant is None else constraints.radius_km[relevant]
    sub_rtt = constraints.rtt_ms if relevant is None else constraints.rtt_ms[relevant]
    n_anch = len(sub_units)
    if n_anch == 0:
        return Evaluation(viol, count, excess, ll, resp)

    for start in range(0, n_pts, chunk):
        stop = min(n_pts, start + chunk)
        pts = points[start:stop]
        d = great_circle_km(pts[:, None, :], sub_units[None, :, :])
        over = d - sub_radius[None, :]
        violated = over > 0.0
        count[start:stop] = np.count_nonzero(violated, axis=1)
        excess[start:stop] = np.where(violated, over, 0.0).sum(axis=1)
        viol[start:stop] = count[start:stop] / n_anch
        lp = params.logpdf(d, sub_rtt[None, :])
        if use_loss:
            # Non-response carries weak distance information; weight it lightly so
            # that ICMP policy drops cannot dominate the geometric evidence.
            d_loss = np.log(np.maximum(1.0 - loss_probability(d), 1e-6))
            lp = lp + 0.35 * d_loss
        ll[start:stop] = lp.sum(axis=1)
        resp[start:stop] = params.responsibility(d, sub_rtt[None, :]).mean(axis=1)
    return Evaluation(viol, count, excess, ll, resp)


def relevant_anchors(constraints: Constraints, centre: np.ndarray,
                     region_radius_km: float) -> np.ndarray:
    """Anchors whose cap can actually cut the search region.

    An anchor whose reach ``radius_km`` falls short of the closest point of the
    region cannot exclude anything inside it, so its constraint is vacuous and it
    can be skipped entirely in the certificate pass.
    """
    d = great_circle_km(np.asarray(centre)[None, :], constraints.units)
    return d <= (region_radius_km + constraints.radius_km)


# --------------------------------------------------------------------------
# Region arithmetic
# --------------------------------------------------------------------------


def approximate_enclosing_cap(points: np.ndarray, sample: int = 400,
                              iters: int = 6) -> tuple[np.ndarray, float]:
    """Approximate minimal cap covering ``points``.

    Uses the classic farthest-pair construction: the minimal enclosing cap has
    radius at least half the largest pairwise separation, and the cap centred on
    that pair's midpoint is a constant-factor approximation. A few refinement
    rounds then shrink it. ``sample`` bounds the cost, which is what keeps this
    usable on a hundred thousand posterior samples.
    """
    if len(points) == 0:
        return np.array([1.0, 0.0, 0.0]), 0.0
    if len(points) == 1:
        return points[0], 0.0
    step = max(1, len(points) // sample)
    sub = points[::step]
    d = great_circle_km(sub[:, None, :], sub[None, :, :])
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    centre = sub[i] + sub[j]
    n = np.linalg.norm(centre)
    centre = sub[i] if n < 1e-12 else centre / n
    # great_circle_km(one, many) broadcasts to shape (len(many),).
    dist = great_circle_km(centre[None, :], sub)
    radius = float(dist.max())
    for _ in range(iters):
        far = int(np.argmax(dist))
        if dist[far] <= radius * 1e-9:
            break
        centre = centre + (sub[far] - centre) * 0.5
        centre = centre / np.linalg.norm(centre)
        dist = great_circle_km(centre[None, :], sub)
        radius = float(dist.max())
    return centre, radius


def _greedy_clusters(points: np.ndarray, weights: np.ndarray,
                     cluster_radius_km: float, max_seeds: int = 256,
                     mass_cover: float = 0.9995) -> list[np.ndarray]:
    """Greedy mode finding: repeatedly take the heaviest unassigned point as a
    seed and absorb everything within ``cluster_radius_km`` of it.

    Only the heaviest ``max_seeds`` points are tried as seeds, and the scan stops
    once ``mass_cover`` of the total weight has been captured. Both bounds matter:
    seeding from every weighted point makes this quadratic in the candidate count,
    which at 60,000 points is billions of distance evaluations and made mesh
    validation take many minutes per case. Because a mode is defined by its peak,
    the heaviest seeds are exactly the ones that can start a new cluster, so upper
    modes are unaffected.
    """
    order = np.argsort(-weights)
    order = order[weights[order] > 0][:max_seeds]
    assigned = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []
    total = float(weights.sum())
    captured = 0.0
    for idx in order:
        if weights[idx] <= 0 or assigned[idx]:
            continue
        d = great_circle_km(points[idx][None, :], points)
        members = (d <= cluster_radius_km) & (~assigned)
        if not members.any():
            continue
        assigned |= members
        clusters.append(np.flatnonzero(members))
        captured += float(weights[members].sum())
        if total > 0 and captured >= mass_cover * total:
            break
    return clusters


def _weighted_direction(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalised weighted mean direction on the sphere."""
    acc = (points * weights[:, None]).sum(axis=0)
    n = np.linalg.norm(acc)
    if n < 1e-12:
        return points[int(np.argmax(weights))]
    return acc / n


# --------------------------------------------------------------------------
# Main estimator
# --------------------------------------------------------------------------


@dataclass
class Mode:
    lat: float
    lon: float
    mass: float
    radius_km: float
    area_km2: float

    def to_dict(self) -> dict:
        return {"lat": round(self.lat, 4), "lon": round(self.lon, 4),
                "posterior_mass": round(self.mass, 4),
                "radius_km": round(self.radius_km, 1),
                "area_km2": round(self.area_km2, 1)}


@dataclass
class Estimate:
    """Full result of one localisation attempt."""

    lat: float
    lon: float
    map_lat: float
    map_lon: float
    modes: list[Mode]
    credible: dict
    certificate: dict
    params: DelayParams
    diagnostics: dict
    #: Headline point estimate: the centroid of the feasible region. Measured
    #: more stable than the region's enclosing-cap centre, which sits on the
    #: boundary and jumps whenever the tightest constraint moves.
    cert_lat: float = 0.0
    cert_lon: float = 0.0
    cert_radius_km: float = 0.0
    #: Centre of the enclosing cap that the certificate reports as its region.
    cap_lat: float = 0.0
    cap_lon: float = 0.0
    points: np.ndarray = field(default=None, repr=False)
    weights: np.ndarray = field(default=None, repr=False)

    @property
    def unit(self) -> np.ndarray:
        return latlon_to_unit(self.lat, self.lon)

    def separation_km(self) -> float:
        """Distance between the model-free and model-refined point estimates."""
        from .geo import haversine_km
        return float(haversine_km(self.lat, self.lon, self.cert_lat, self.cert_lon))

    def to_dict(self, include_samples: bool = False) -> dict:
        d = {
            "estimate": {"lat": round(self.cert_lat, 5), "lon": round(self.cert_lon, 5)},
            "map": {"lat": round(self.map_lat, 5), "lon": round(self.map_lon, 5)},
            "certificate_centroid": {"lat": round(self.cert_lat, 5),
                                     "lon": round(self.cert_lon, 5),
                                     "radius_km": round(self.cert_radius_km, 1)},
            "certificate_cap_centre": {"lat": round(self.cap_lat, 5),
                                       "lon": round(self.cap_lon, 5)},
            "posterior_mean": {"lat": round(self.lat, 5), "lon": round(self.lon, 5)},
            "estimate_separation_km": round(self.separation_km(), 1),
            "modes": [m.to_dict() for m in self.modes],
            "credible_regions": self.credible,
            "certificate": self.certificate,
            "delay_model": self.params.as_dict(),
            "diagnostics": self.diagnostics,
        }
        return d


def estimate_location(
        measurements: Sequence[AnchorMeasurement],
        *,
        exploration: int = 40000,
        final_points: int = 220_000,
        levels: int = 2,
        floor_q: float = 0.15,
        use_loss: bool = False,
        fit_iterations: int = 3,
        fit_locally: bool = True,
        params_prior: DelayParams | None = None,
        progress=lambda s: None,
        rng: np.random.Generator | None = None,
) -> Estimate:
    """Estimate the client position from anchor round-trip times.

    Feasibility is not a tunable tolerance here; it is derived from the data as
    the fewest speed-of-light violations any location achieves (see
    :func:`_consensus_point`). The statistical model is fitted at that
    model-free consensus and only ever sharpens the answer inside it.

    ``params_prior`` defaults to the ground-truth calibration produced by
    :mod:`latlas.calibrate`. ``fit_locally`` allows the model to then re-fit at
    the consensus using this client's own data; the mesh validation measures
    whether that refinement helps or hurts, and the default follows the result.
    """
    obs = observations_from_measurements(measurements, floor_q=floor_q)
    if len(obs) < 4:
        raise ValueError(f"need at least 4 responding anchors, got {len(obs)}")
    con = build_constraints(obs)
    obs_sorted = sorted(obs, key=lambda o: o.rtt_ms)

    # ---- 1. certificate-driven search window -----------------------------
    # The nearest anchor alone confines the client to one spherical cap.
    nearest = obs_sorted[0]
    cap_radius = nearest.radius_km
    centre = nearest.unit

    # A single mislabelled or anycast anchor could shrink that window below the
    # truth, so widen modestly; the widening costs resolution, not correctness,
    # because real constraints are applied afterwards on every candidate.
    safety = 1.30
    search_radius = min(math.pi * EARTH_RADIUS_KM, cap_radius * safety + 120.0)
    progress(f"  nearest anchor {nearest.key} floor {nearest.rtt_ms:.2f} ms -> "
             f"certificate cap {cap_radius:,.0f} km (searching {search_radius:,.0f} km)")

    relevant = relevant_anchors(con, centre, search_radius)

    # ---- 2. exploration pass --------------------------------------------
    pts = cap_points(centre, search_radius, exploration)
    params = params_prior or load_calibrated_params() or DelayParams()
    params = fit_outlier_component(con.rtt_ms, params)
    ev = evaluate(con, pts, params, relevant=relevant, use_loss=use_loss)
    consensus = _consensus_point(pts, ev)
    v_star = float(ev.violation_count[_consensus_index(ev)])
    progress(f"  exploration: {len(pts):,} points, best consensus violates "
             f"{int(v_star)} of {len(con)} constraints")

    # ---- 3. alternate: fit the model, then re-find the consensus ---------
    # The delay model is fitted *at the consensus point*, which is established
    # without it, so a poorly specified model cannot drag the geometry off the
    # physical solution.
    for it in range(fit_iterations if fit_locally else 0):
        d = great_circle_km(consensus[None, :], con.units)
        inlier = d <= con.radius_km
        d_in = d[inlier]
        r_in = con.rtt_ms[inlier]
        if d_in.size >= 8:
            params = fit_params(d_in, r_in, base=params)
        params = fit_outlier_component(con.rtt_ms, params)
        ev = evaluate(con, pts, params, relevant=relevant, use_loss=use_loss)
        consensus = _consensus_point(pts, ev)
        v_star = float(ev.violation_count[_consensus_index(ev)])
        c_lat, c_lon = unit_to_latlon(consensus)
        progress(f"  iter {it+1}: consensus {float(c_lat):+.3f},{float(c_lon):+.3f} "
                 f"violations {int(v_star)}, median detour {math.exp(params.mu):.3f}, "
                 f"sigma {params.sigma:.3f}, beta {params.beta:.2f} ms")

    # ---- 4. locate candidate sub-regions and resample uniformly ----------
    # Feasibility is defined by the consensus itself, not by a fixed tolerance:
    # the region is "no worse than the best explanation found", which adapts to
    # how contaminated this particular anchor set turns out to be.
    mask = ev.violation_count <= v_star
    if not mask.any():
        raise ValueError("no candidate satisfies any cone constraints; "
                         "the anchor set may be too heavily contaminated")
    w = _normalised_weights(ev, mask)
    caps: list[tuple[np.ndarray, float, float]] = []   # centre, radius, mass
    keep_mass = 0.999
    clusters = _greedy_clusters(pts, w, cluster_radius_km=max(120.0, search_radius * 0.08))
    clusters = [c for c in clusters if w[c].sum() > 1e-6]
    clusters.sort(key=lambda c: -w[c].sum())
    acc = 0.0
    for c in clusters:
        mass = float(w[c].sum())
        cap_c, cap_r = approximate_enclosing_cap(pts[c])
        caps.append((cap_c, cap_r, mass))
        acc += mass
        if acc >= keep_mass or len(caps) >= 6:
            break
    progress(f"  posterior condensed into {len(caps)} mode(s) covering "
             f"{100*acc:.1f}% of mass")

    # ---- 5. final pass at survey resolution, area-equal weights ----------
    caps = [(c, max(r * 1.25 + 15.0, 40.0), m) for c, r, m in caps]
    areas = np.array([spherical_cap_area_km2(r) for _, r, _ in caps])
    share = areas / areas.sum()
    quota = np.maximum(4000, (share * final_points).astype(int))
    parts = [cap_points(c, r, int(n)) for (c, r, _), n in zip(caps, quota)]
    final_pts = np.concatenate(parts, axis=0)
    # Equal-area weights: each point stands for (cap area / n points in cap), and
    # the caps were sized proportionally, so a single global constant suffices.
    ev_f = evaluate(con, final_pts, params, relevant=relevant, use_loss=use_loss)
    mask_f = ev_f.violation_count <= v_star
    if not mask_f.any():
        mask_f = np.ones(len(final_pts), dtype=bool)
    w_f = _normalised_weights(ev_f, mask_f)

    # ---- 6. summarise ----------------------------------------------------
    modes = _summarise_modes(final_pts, w_f, caps)
    mean_dir = _weighted_direction(final_pts, _restrict_to_top_mode(final_pts, w_f, modes))
    mean_lat, mean_lon = unit_to_latlon(mean_dir)
    best_i = int(np.argmax(np.where(mask_f, ev_f.loglik, -np.inf)))
    map_lat, map_lon = unit_to_latlon(final_pts[best_i])

    credible = _credible_regions(final_pts, w_f, caps)
    # The certificate is computed from the *exploration* pass, which is uniform
    # over the whole search window. Computing it from the refined points instead
    # would shrink it to whatever the likelihood happened to concentrate on, and
    # it did: it reported a 405 km region when the true zero-violation set was
    # 750 km across. A certificate that under-reports its own feasible set is
    # simply wrong, so it must come from an unbiased sample of the window.
    certificate = _certificate_summary(pts, ev, ev.violation_count <= v_star,
                                       v_star, con, params)
    feas = ev.violation_count <= v_star
    if feas.any():
        # Headline position = area centroid of the region physics permits.
        # The enclosing cap describes how big that region is; its centre is a
        # boundary point and swings much more between scans, so it is recorded
        # separately rather than used as the answer.
        centroid = pts[feas].mean(axis=0)
        n = float(np.linalg.norm(centroid))
        centroid = (pts[feas][0] if n < 1e-12 else centroid / n)
        cert_lat, cert_lon = unit_to_latlon(centroid)
        cert_lat, cert_lon = float(cert_lat), float(cert_lon)
        cap_unit, cert_radius = approximate_enclosing_cap(pts[feas])
        cap_lat, cap_lon = unit_to_latlon(cap_unit)
        cap_lat, cap_lon = float(cap_lat), float(cap_lon)
    else:
        cert_lat, cert_lon = float(mean_lat), float(mean_lon)
        cap_lat, cap_lon, cert_radius = cert_lat, cert_lon, float("nan")

    diagnostics = _diagnostics(con, params, measurements, nearest,
                              latlon_to_unit(cert_lat, cert_lon))
    return Estimate(
        lat=float(mean_lat), lon=float(mean_lon),
        map_lat=float(map_lat), map_lon=float(map_lon),
        modes=modes, credible=credible, certificate=certificate,
        params=params, diagnostics=diagnostics,
        cert_lat=cert_lat, cert_lon=cert_lon, cert_radius_km=float(cert_radius),
        cap_lat=cap_lat, cap_lon=cap_lon,
        points=final_pts, weights=w_f,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _consensus_index(ev: Evaluation, mask: np.ndarray | None = None) -> int:
    """Index of the best explanation under a lexicographic, model-free order.

    Minimise the number of contradicted speed-of-light constraints, then the
    total contradicted distance, and only then prefer higher likelihood. The
    ordering is deliberate: an earlier revision let a mis-calibrated likelihood
    choose a location that contradicted the tightest physical constraints in the
    data, and it placed the estimate 1,000 km from a location that satisfied all
    of them.
    """
    idx = np.flatnonzero(mask) if mask is not None else np.arange(len(ev.loglik))
    if idx.size == 0:
        return 0
    # np.lexsort's final key is the primary one, and it returns a permutation,
    # so the best element is order[0] -- not an argmin over the permutation.
    order = np.lexsort((-ev.loglik[idx], ev.violation_excess_km[idx],
                        ev.violation_count[idx]))
    return int(idx[order[0]])


def _consensus_point(pts: np.ndarray, ev: Evaluation,
                     count_limit: float | None = None) -> np.ndarray:
    """Location contradicting the fewest, and least severely, speed-of-light bounds."""
    mask = None
    if count_limit is not None:
        m = ev.violation_count <= count_limit
        if m.any():
            mask = m
    return pts[_consensus_index(ev, mask)]


def _normalised_weights(ev: Evaluation, mask: np.ndarray) -> np.ndarray:
    """Posterior weights over an area-equal candidate set."""
    ll = np.where(mask, ev.loglik, -np.inf)
    m = float(np.max(ll))
    if not np.isfinite(m):
        return np.zeros(len(ev.loglik))
    w = np.exp(ll - m)
    w[~mask] = 0.0
    s = w.sum()
    return w / s if s > 0 else w


def _restrict_to_top_mode(points: np.ndarray, weights: np.ndarray,
                          modes: Sequence[Mode]) -> np.ndarray:
    if not modes:
        return weights
    d = great_circle_km(latlon_to_unit(modes[0].lat, modes[0].lon)[None, :], points)
    m = d <= max(modes[0].radius_km * 1.5, 50.0)
    out = np.zeros_like(weights)
    out[m] = weights[m]
    s = out.sum()
    return out / s if s > 0 else weights


def _summarise_modes(points: np.ndarray, weights: np.ndarray,
                     caps) -> list[Mode]:
    clusters = _greedy_clusters(points, weights, cluster_radius_km=100.0)
    modes: list[Mode] = []
    for c in clusters[:5]:
        mass = float(weights[c].sum())
        if mass < 1e-4:
            continue
        wc = weights[c] / max(weights[c].sum(), 1e-300)
        direction = _weighted_direction(points[c], wc)
        lat, lon = unit_to_latlon(direction)
        d = great_circle_km(direction[None, :], points[c])
        # Radius covering the given posterior mass of this mode.
        order = np.argsort(d)
        cum = np.cumsum(wc[order])
        k = int(np.searchsorted(cum, 0.9))
        radius = float(d[order[min(k, len(order) - 1)]])
        modes.append(Mode(lat=float(lat), lon=float(lon), mass=mass,
                          radius_km=radius,
                          area_km2=spherical_cap_area_km2(radius)))
    modes.sort(key=lambda m: -m.mass)
    return modes


def _credible_regions(points: np.ndarray, weights: np.ndarray, caps) -> dict:
    """Smallest-area regions holding 50% and 90% of the posterior mass."""
    order = np.argsort(-weights)
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    out = {}
    area_per_pt = _area_per_point(caps, points)
    for level in (0.5, 0.9):
        k = int(np.searchsorted(cum, level)) + 1
        sel = order[:k]
        centre, radius = approximate_enclosing_cap(points[sel])
        lat, lon = unit_to_latlon(centre)
        out[f"p{int(level*100)}"] = {
            "centre_lat": round(float(lat), 5),
            "centre_lon": round(float(lon), 5),
            "enclosing_cap_radius_km": round(float(radius), 1),
            "enclosing_cap_diameter_km": round(float(radius) * 2, 1),
            "area_km2": round(float(area_per_pt * k), 1),
        }
    return out


def _area_per_point(caps, points: np.ndarray) -> float:
    """Mean area represented by one candidate point (caps are area-weighted)."""
    total_area = sum(spherical_cap_area_km2(r) for _, r, _ in caps)
    return total_area / max(1, len(points))


def _certificate_summary(points: np.ndarray, ev: Evaluation, mask: np.ndarray,
                         v_star: float, con: Constraints,
                         params: DelayParams) -> dict:
    """Model-free uncertainty statement plus contamination diagnostics.

    ``v_star`` is the smallest number of speed-of-light constraints that any
    location on the globe was found to contradict. The reported region is the set
    of locations contradicting no more than that many, so its validity rests only
    on the constancy of the speed of light and on no more than ``v_star`` anchors
    being mislocated or anycast -- not on the statistical model at all.
    """
    strict = ev.violation_count == 0
    out: dict = {
        "method": "intersection of speed-of-light spherical caps",
        "max_km_per_ms": round(MAX_KM_PER_MS, 3),
        "quantum_slack_ms": QUANTUM_MS,
        "anchors_used": len(con),
        "min_violations_achievable": int(v_star),
        "tolerance_frac": round(v_star / max(1, len(con)), 5),
    }
    for name, m in (("strict", strict), ("consensus", mask)):
        if m.any():
            centre, radius = approximate_enclosing_cap(points[m])
            lat, lon = unit_to_latlon(centre)
            out[f"{name}_region"] = {
                "centre_lat": round(float(lat), 5),
                "centre_lon": round(float(lon), 5),
                "radius_km": round(float(radius), 1),
                "diameter_km": round(float(radius) * 2, 1),
                "n_points": int(m.sum()),
            }
        else:
            out[f"{name}_region"] = None
    if mask.any():
        i = int(np.argmax(np.where(mask, ev.loglik, -np.inf)))
        out["n_constraints_violated_at_estimate"] = int(ev.violation_count[i])
    return out


def _diagnostics(con: Constraints, params: DelayParams,
                 measurements: Sequence[AnchorMeasurement],
                 nearest: AnchorObservation, centre: np.ndarray) -> dict:
    """Contamination report, scored against the certificate centre.

    Two distinct questions are asked, because conflating them produced a wildly
    misleading number earlier. First, how many anchors does the solution actually
    *contradict* -- a hard, model-free count. Second, how many anchors does the
    statistical model find hard to explain, which is a soft and much noisier
    notion. An earlier revision reported only the second while describing it as
    the first, and claimed a third of a demonstrably clean anchor set was
    anycast.
    """
    d = great_circle_km(np.asarray(centre)[None, :], con.units)
    violated = np.flatnonzero(d > con.radius_km)
    resp = params.responsibility(d, con.rtt_ms)
    low = np.flatnonzero(resp < 0.05)

    # Distance the RTT implies under the fitted model, versus what was advertised.
    implied = params.implied_distance_km(con.rtt_ms)
    implied = np.where(np.isfinite(implied), implied, 0.0)
    order = low[np.argsort(-(d[low] - implied[low]))] if low.size else low
    offenders = []
    for i in order[:12]:
        offenders.append({
            "key": con.keys[int(i)], "source": con.sources[int(i)],
            "advertised_km": round(float(d[i]), 0),
            "rtt_implied_km": round(float(implied[i]), 0),
            "rtt_ms": round(float(con.rtt_ms[int(i)]), 2),
            "responsibility": round(float(resp[i]), 4),
        })

    return {
        "responding_anchors": len(con),
        "attempted_anchors": len(measurements),
        "response_rate": round(len(con) / max(1, len(measurements)), 4),
        "nearest_anchor": {"key": nearest.key, "rtt_ms": round(nearest.rtt_ms, 2),
                           "certificate_radius_km": round(nearest.radius_km, 0),
                           "source": nearest.source, "city": nearest.city,
                           "country": nearest.country},
        "implied_median_detour": round(math.exp(params.mu), 4),
        "min_observed_rtt_ms": round(float(np.min(con.rtt_ms)), 2),
        "median_observed_rtt_ms": round(float(np.median(con.rtt_ms)), 2),
        "max_observed_rtt_ms": round(float(np.max(con.rtt_ms)), 2),
        "constraints_violated_at_certificate": int(len(violated)),
        "violated_keys": [con.keys[int(i)] for i in violated[:20]],
        "model_inconsistent_anchors": int(len(low)),
        "model_inconsistent_frac": round(float(len(low)) / max(1, len(con)), 4),
        "suspected_mislocated_or_anycast": offenders,
        "n_suspected": int(len(violated)),
        "suspected_frac": round(float(len(violated)) / max(1, len(con)), 4),
    }
