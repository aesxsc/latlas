"""Validation: measuring accuracy and, more importantly, calibration.

Three independent checks are run, and they test different things.

**Mesh validation (real, thousands of cases).** RIPE Atlas anchor-mesh
measurements supply real RTT vectors from ~1000 real vantage points with
published coordinates. The exact estimator is re-run from each one and scored
against the known answer. This is the only check with statistical power, and it
is the one that can expose a wrong delay model, because it exercises the full
pipeline on networks the author never touched.

**Self validation (real, one case).** The machine that produced the measurements
is itself a vantage point with a known answer. It is the deployment case, so it
is reported even though n=1; it is also the only check that exercises the ICMP
probe code end to end.

**Robustness (real, derived from the same data).** Jackknife over the nearest
anchors, bootstrap over the sample series, and synthetic noise injection probe
how the estimate degrades and whether the reported uncertainty tracks the real
error. These are the checks that would catch an over-confident uncertainty
statement, which is the failure mode that matters most for a system whose whole
purpose is to be honest about what it knows.

The headline number is *containment*: the fraction of cases in which the true
location actually fell inside the region the system reported. An estimator that
is accurate on average but whose error bars never cover the truth is worse than
useless here, and only containment reveals that.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .estimate import Estimate, estimate_location
from .geo import great_circle_km, haversine_km, latlon_to_unit
from .measure import AnchorMeasurement
from .mesh_truth import VantageCase


def measurement_from_observation(key: str, lat: float, lon: float, rtt_ms: float,
                                 loss: float = 0.0, source: str = "mesh") -> AnchorMeasurement:
    """Wrap a single anchor RTT in the estimator's measurement type."""
    return AnchorMeasurement(
        key=key, ip="", lat=lat, lon=lon, source=source,
        samples_ms=[float(rtt_ms)], errors=["success"],
    )


def case_to_measurements(case: VantageCase) -> list[AnchorMeasurement]:
    return [measurement_from_observation(k, la, lo, r)
            for (k, la, lo, r) in case.observations]


def _cap_contains(centre_lat: float, centre_lon: float, radius_km: float,
                  lat: float, lon: float) -> bool:
    d = haversine_km(centre_lat, centre_lon, lat, lon)
    return float(d) <= radius_km


def containment(est: Estimate, lat: float, lon: float) -> dict:
    """Did the reported regions actually contain the truth?"""
    out = {"error_km": float(haversine_km(est.lat, est.lon, lat, lon)),
           "feasible_centroid_error_km": float(haversine_km(est.cert_lat, est.cert_lon,
                                                            lat, lon)),
           "cap_centre_error_km": float(haversine_km(est.cap_lat, est.cap_lon,
                                                     lat, lon))}
    cert = est.certificate.get("consensus_region")
    if cert:
        out["in_certificate"] = _cap_contains(cert["centre_lat"], cert["centre_lon"],
                                             cert["radius_km"], lat, lon)
        out["certificate_radius_km"] = cert["radius_km"]
    else:
        out["in_certificate"] = False
        out["certificate_radius_km"] = float("nan")
    for key in ("p50", "p90"):
        r = est.credible.get(key)
        if r:
            out[f"in_{key}"] = _cap_contains(r["centre_lat"], r["centre_lon"],
                                            r["enclosing_cap_radius_km"], lat, lon)
            out[f"{key}_radius_km"] = r["enclosing_cap_radius_km"]
        else:
            out[f"in_{key}"] = False
            out[f"{key}_radius_km"] = float("nan")
    # Distance from the truth to the point estimate, measured against the
    # reported p90 radius: the ratio is the sharpest single measure of whether
    # the uncertainty is calibrated (ideal: errors roughly equal radii).
    out["p90_ratio"] = (out["error_km"] / out["p90_radius_km"]
                        if out.get("p90_radius_km") else float("nan"))
    return out


@dataclass
class MeshValidation:
    n_cases: int = 0
    n_ok: int = 0
    n_failed: int = 0
    errors_km: list[float] = field(default_factory=list)
    feasible_centroid_errors_km: list[float] = field(default_factory=list)
    cap_centre_errors_km: list[float] = field(default_factory=list)
    cert_radii_km: list[float] = field(default_factory=list)
    p50_radii_km: list[float] = field(default_factory=list)
    p90_radii_km: list[float] = field(default_factory=list)
    in_cert: int = 0
    in_p50: int = 0
    in_p90: int = 0
    rows: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        e = sorted(x for x in self.errors_km if math.isfinite(x))
        if not e:
            return {"n_cases": self.n_cases, "n_ok": self.n_ok,
                    "n_failed": self.n_failed, "error_km": None}

        def pct(q, series=None):
            s = sorted(x for x in (series if series is not None else e)
                       if math.isfinite(x))
            if not s:
                return None
            return round(s[min(len(s) - 1, int(q * len(s)))], 1)

        def med(v):
            v = [x for x in v if math.isfinite(x)]
            return round(statistics.median(v), 1) if v else None

        return {
            "n_cases": self.n_cases,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "posterior_mean_error_km": {
                "median": pct(0.5), "mean": round(sum(e) / len(e), 1),
                "p25": pct(0.25), "p75": pct(0.75), "p90": pct(0.90),
                "p95": pct(0.95), "max": round(e[-1], 1)},
            "feasible_centroid_error_km": {
                "median": pct(0.5, self.feasible_centroid_errors_km),
                "p75": pct(0.75, self.feasible_centroid_errors_km),
                "p90": pct(0.90, self.feasible_centroid_errors_km)},
            "cap_centre_error_km": {
                "median": pct(0.5, self.cap_centre_errors_km),
                "p75": pct(0.75, self.cap_centre_errors_km),
                "p90": pct(0.90, self.cap_centre_errors_km)},
            "certificate_radius_km": {"median": med(self.cert_radii_km)},
            "p50_region_radius_km": {"median": med(self.p50_radii_km)},
            "p90_region_radius_km": {"median": med(self.p90_radii_km)},
            "containment": {
                "certificate": round(self.in_cert / self.n_ok, 4) if self.n_ok else None,
                "p50_region": round(self.in_p50 / self.n_ok, 4) if self.n_ok else None,
                "p90_region": round(self.in_p90 / self.n_ok, 4) if self.n_ok else None,
            },
        }


def validate_mesh(cases: Sequence[VantageCase], *, limit: int | None = None,
                  progress: Callable[[str], None] = print,
                  **estimator_kwargs) -> MeshValidation:
    """Run the estimator from every mesh vantage point and score against truth."""
    cases = list(cases)[:limit] if limit else list(cases)
    res = MeshValidation()
    for i, case in enumerate(cases):
        res.n_cases += 1
        ms = case_to_measurements(case)
        if len(ms) < 6:
            res.n_failed += 1
            continue
        try:
            est = estimate_location(ms, progress=lambda s: None, **estimator_kwargs)
        except Exception as e:  # noqa: BLE001
            res.n_failed += 1
            res.failures.append({"probe_id": case.probe_id, "error": str(e),
                                 "n_anchors": len(ms)})
            continue
        c = containment(est, case.lat, case.lon)
        res.n_ok += 1
        res.errors_km.append(c["error_km"])
        res.feasible_centroid_errors_km.append(
            c.get("feasible_centroid_error_km", float("nan")))
        res.cap_centre_errors_km.append(c.get("cap_centre_error_km", float("nan")))
        res.cert_radii_km.append(c["certificate_radius_km"])
        res.p50_radii_km.append(c["p50_radius_km"])
        res.p90_radii_km.append(c["p90_radius_km"])
        res.in_cert += int(bool(c.get("in_certificate")))
        res.in_p50 += int(bool(c.get("in_p50")))
        res.in_p90 += int(bool(c.get("in_p90")))
        res.rows.append({
            "probe_id": case.probe_id, "country": case.country,
            "n_anchors": len(case.observations),
            "true_lat": round(case.lat, 4), "true_lon": round(case.lon, 4),
            "est_lat": round(est.lat, 4), "est_lon": round(est.lon, 4),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()},
        })
        if (i + 1) % 25 == 0:
            progress(f"    validated {i+1}/{len(cases)} vantage points "
                     f"(median error {statistics.median(res.errors_km):.0f} km)")
    return res


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def nearest_anchor_jackknife(measurements: Sequence[AnchorMeasurement],
                             true_lat: float, true_lon: float, *,
                             drops: Sequence[int] = (0, 1, 2, 4, 8, 16, 32),
                             progress=print, **estimator_kwargs) -> list[dict]:
    """Re-estimate after discarding the K nearest anchors.

    The nearest anchors carry almost all of the localising power, so this is the
    sharpest available stress test: it shows what happens when a client happens
    to sit in a region with poor coverage. It is also a direct probe of the
    certificate, which is driven entirely by the smallest observed RTT.
    """
    unit = latlon_to_unit(true_lat, true_lon)
    dist = [float(haversine_km(true_lat, true_lon, m.lat, m.lon))
            for m in measurements]
    order = sorted(range(len(measurements)), key=lambda i: dist[i])
    out = []
    for k in drops:
        keep = set(order[k:])
        ms = [m for i, m in enumerate(measurements) if i in keep]
        row = {"dropped_nearest": k, "remaining_anchors": len(ms)}
        if len(ms) < 6:
            row["error"] = None
            out.append(row)
            continue
        try:
            est = estimate_location(ms, progress=lambda s: None, **estimator_kwargs)
            c = containment(est, true_lat, true_lon)
            row.update({
                "error_km": round(c["error_km"], 1),
                "certificate_radius_km": round(c["certificate_radius_km"], 1),
                "in_certificate": c.get("in_certificate"),
                "p90_radius_km": round(c["p90_radius_km"], 1),
                "in_p90": c.get("in_p90"),
                "est_lat": round(est.lat, 4), "est_lon": round(est.lon, 4),
            })
            progress(f"    drop {k:>2} nearest: error {row['error_km']:>8} km, "
                     f"certificate r={row['certificate_radius_km']} km")
        except Exception as e:  # noqa: BLE001
            row["error"] = None
            row["failure"] = str(e)
        out.append(row)
    return out


def sample_bootstrap(measurements: Sequence[AnchorMeasurement], true_lat: float,
                     true_lon: float, *, trials: int = 8, seed: int = 7,
                     progress=print, **estimator_kwargs) -> dict:
    """Re-estimate using a random half of each anchor's echo series.

    Quantifies how much of the reported uncertainty is measurement noise rather
    than geometry: if the estimate swings wildly between resamples, the answer is
    not trustworthy no matter how tight the reported region looks.
    """
    rng = np.random.default_rng(seed)
    errs, lats, lons = [], [], []
    for t in range(trials):
        ms = []
        for m in measurements:
            r = m.rtts
            if not r:
                continue
            pick = [r[i] for i in rng.integers(0, len(r), size=max(1, len(r) // 2))]
            ms.append(AnchorMeasurement(
                key=m.key, ip=m.ip, lat=m.lat, lon=m.lon, source=m.source,
                samples_ms=[float(x) for x in pick], errors=["success"],
            ))
        try:
            est = estimate_location(ms, progress=lambda s: None,
                                    rng=np.random.default_rng(seed + t), **estimator_kwargs)
        except Exception:
            continue
        errs.append(float(haversine_km(est.lat, est.lon, true_lat, true_lon)))
        lats.append(est.lat); lons.append(est.lon)
    if not errs:
        return {"trials": 0}
    # Spread of the resampled estimates, measured about their spherical mean.
    # Averaging latitude and longitude numerically would be wrong across the
    # antimeridian, so the mean is taken on unit vectors.
    u = latlon_to_unit(np.asarray(lats), np.asarray(lons))
    centre = u.mean(axis=0)
    n = float(np.linalg.norm(centre))
    if n < 1e-12:
        spread = float(great_circle_km(u[0][None, :], u).max())
    else:
        spread = float(great_circle_km((centre / n)[None, :], u).max())
    return {
        "trials": len(errs),
        "error_km": {"median": round(statistics.median(errs), 1),
                     "min": round(min(errs), 1), "max": round(max(errs), 1)},
        "estimate_spread_km": round(spread, 1),
        "lat_std_deg": round(float(np.std(lats)), 4),
        "lon_std_deg": round(float(np.std(lons)), 4),
    }
