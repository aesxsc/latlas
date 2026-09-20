"""Regression tests for the estimator core.

These cover the failure modes that actually occurred during development and that
end-to-end output would not have revealed:

* a latitude/longitude transposition and a spherical-geometry error, caught by
  axis checks on the unit-vector conversion (distance-from-origin is symmetric in
  lat and lon, so it cannot detect a swap);
* the delay model assigning probability to paths faster than light, caught by
  asserting the informative component is exactly zero below the cone floor;
* selection choosing a location that contradicts the tightest constraints,
  caught by asserting the model-free feasible region contains a known truth.

Run with ``python -m pytest tests`` or directly as ``python tests/test_latlas.py``.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latlas.estimate import MAX_KM_PER_MS, QUANTUM_MS, estimate_location
from latlas.geo import (C_KM_PER_MS, FIBER_RT_KM_PER_MS, great_circle_km,
                        haversine_km, latlon_to_unit, unit_to_latlon)
from latlas.measure import AnchorMeasurement
from latlas.model import K_FIBER, DelayParams


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_unit_vector_axes():
    """lat is latitude and lon is longitude; a swap must be detectable."""
    assert np.allclose(latlon_to_unit(90, 0), [0, 0, 1], atol=1e-9)
    assert np.allclose(latlon_to_unit(-90, 0), [0, 0, -1], atol=1e-9)
    assert np.allclose(latlon_to_unit(0, 0), [1, 0, 0], atol=1e-9)
    assert np.allclose(latlon_to_unit(0, 90), [0, 1, 0], atol=1e-9)
    assert np.allclose(latlon_to_unit(0, -90), [0, -1, 0], atol=1e-9)
    assert not np.allclose(latlon_to_unit(0, 90), latlon_to_unit(90, 0), atol=1e-6)


def test_round_trip_latlon():
    for lat, lon in [(0, 0), (48.85, 2.35), (-33.87, 151.21), (64.0, -21.0),
                     (52.52, 13.40), (1.0, 179.9), (-1.0, -179.9)]:
        la, lo = unit_to_latlon(latlon_to_unit(lat, lon))
        assert abs(float(la) - lat) < 1e-9, (lat, lon)
        assert abs(float(lo) - lon) < 1e-9, (lat, lon)


def test_distance_consistency_and_scale():
    assert abs(float(haversine_km(10, 20, 10, 20))) < 1e-9
    assert abs(float(haversine_km(0, 0, 0, 180)) - math.pi * 6371.0088) < 1.0
    assert abs(float(haversine_km(0, 0, 1, 0)) - 111.19) < 0.5
    # Vector path must agree with the degree-based path.
    a = float(haversine_km(60, 10, -20, 140))
    b = float(great_circle_km(latlon_to_unit(60, 10), latlon_to_unit(-20, 140)))
    assert abs(a - b) < 1e-6
    # Broadcasting: one point against many yields one distance per point.
    d = great_circle_km(latlon_to_unit(0, 0)[None, :],
                        latlon_to_unit([0, 0, 0], [10, 20, 30]))
    assert d.shape == (3,)


def test_speed_constants_are_consistent():
    """The model's floor must never beat the certificate's ceiling."""
    assert MAX_KM_PER_MS == C_KM_PER_MS / 2.0
    assert FIBER_RT_KM_PER_MS < MAX_KM_PER_MS, "fibre must be slower than vacuum"
    assert abs(K_FIBER - 102.15) < 0.01


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def test_model_has_no_superluminal_support():
    """The informative component must be exactly zero below the fibre floor.

    An earlier revision used an unbounded multiplicative detour, which put
    roughly a third of its mass on paths faster than fibre -- and for short
    distances faster than light -- contradicting the very certificate the same
    code derived from those measurements.
    """
    p = DelayParams()
    for d in (0.0, 5.0, 100.0, 1000.0, 12000.0):
        floor_ms = d / K_FIBER + p.beta
        # Strictly below the floor: no density at all.
        assert p.inlier_logpdf(np.array([d]), np.array([floor_ms * 0.99]))[0] == -np.inf
        # At and above the floor: finite density.
        assert np.isfinite(p.inlier_logpdf(np.array([d]), np.array([floor_ms * 1.05]))[0])


def test_model_floor_respects_vacuum_cone():
    """Distance implied by the model must satisfy the hard certificate bound."""
    p = DelayParams()
    for r in (0.5, 1.0, 5.0, 50.0, 300.0):
        # Worst case the model allows: alpha = 1, beta = 0.
        d_at_floor = (r - p.beta) * K_FIBER
        if d_at_floor > 0:
            assert d_at_floor <= MAX_KM_PER_MS * (r + QUANTUM_MS) + 1e-6


def test_model_monotone_in_distance():
    """A fixed RTT must imply a larger distance when the anchor is farther.

    The RTT used must clear the fibre floor at the *nearer* distance too, since
    below that floor the density is legitimately zero.
    """
    p = DelayParams()
    r = np.array([60.0])
    assert 60.0 > 4000.0 / K_FIBER + p.beta, "pick an r above the floor at 4000 km"
    near = p.inlier_logpdf(np.array([1000.0]), r)[0]
    far = p.inlier_logpdf(np.array([4000.0]), r)[0]
    assert near > -np.inf and far > -np.inf
    assert far > near


# --------------------------------------------------------------------------
# Estimator, known answer
# --------------------------------------------------------------------------


def _synth(rng, true_lat, true_lon, n_anchors=200, mu=0.26, sigma=0.30,
           beta=0.35, n_samples=6):
    """RTTs drawn from the model at a known location, with an integer quantum.

    The detour is drawn from the *truncated* family so the synthetic data obeys
    the same light-cone inequality the estimator enforces.
    """
    from latlas.geo import cap_points
    pts = []
    local = cap_points(latlon_to_unit(true_lat, true_lon), 3000.0, n_anchors // 3)
    lat_l, lon_l = unit_to_latlon(local)
    pts.extend(zip(lat_l.tolist(), lon_l.tolist()))
    for i in range(n_anchors - len(pts)):
        z = 1.0 - 2.0 * (i + 0.5) / (n_anchors - len(pts))
        r_ = math.sqrt(max(0.0, 1 - z * z))
        th = 2 * math.pi * i / ((1 + 5 ** 0.5) / 2)
        la, lo = unit_to_latlon(np.array([r_ * math.cos(th), r_ * math.sin(th), z]))
        pts.append((float(la), float(lo)))
    ms = []
    for idx, (la, lo) in enumerate(pts):
        d = float(haversine_km(true_lat, true_lon, la, lo))
        samples = []
        for _ in range(n_samples):
            alpha = 0.0
            while alpha < 1.0:
                alpha = math.exp(rng.normal(mu, sigma))
            samples.append(float(round(alpha * d / K_FIBER + beta)))
        ms.append(AnchorMeasurement(key=f"a{idx}", ip="", lat=la, lon=lo,
                                    source="synthetic", samples_ms=samples,
                                    errors=["success"] * n_samples))
    return ms


def test_estimator_recovers_known_locations():
    rng = np.random.default_rng(20240920)
    errors = []
    for tlat, tlon in [(48.85, 2.35), (-20.0, -160.0), (64.0, -21.0),
                       (-45.0, 60.0), (1.0, 179.0)]:
        ms = _synth(rng, tlat, tlon)
        est = estimate_location(ms, exploration=12000, final_points=45000,
                                fit_iterations=2, progress=lambda s: None)
        err = float(haversine_km(est.lat, est.lon, tlat, tlon))
        errors.append(err)
        # The truth must satisfy its own certificate; if not, the geometry is wrong.
        cert = est.certificate.get("consensus_region")
        assert cert is not None, "no feasible region found"
        d_true = float(haversine_km(cert["centre_lat"], cert["centre_lon"], tlat, tlon))
        assert d_true <= cert["radius_km"], (
            f"truth {tlat},{tlon} lies {d_true:.0f} km outside the reported "
            f"{cert['radius_km']:.0f} km certificate")
        assert err < 2000.0, f"error {err:.0f} km at {tlat},{tlon}"
    assert sorted(errors)[len(errors) // 2] < 400.0, errors


def test_estimator_prefers_consensus_over_likelihood():
    """A point violating the tightest constraints must never beat one satisfying all.

    The estimator must order feasibility ahead of likelihood. When it did not, it
    produced a 1,045 km error on a case where a zero-violation location existed.
    """
    rng = np.random.default_rng(7)
    ms = _synth(rng, 52.52, 13.40)
    est = estimate_location(ms, exploration=15000, final_points=50000,
                            fit_iterations=2, progress=lambda s: None)
    assert est.certificate["min_violations_achievable"] == 0
    assert est.certificate["n_constraints_violated_at_estimate"] == 0


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [PASS] {name}")
            except AssertionError as e:
                print(f"  [FAIL] {name}: {e}")
                failures.append(name)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
                failures.append(name)
    print("\n" + ("ALL TESTS PASSED" if not failures else f"FAILED: {failures}"))
    raise SystemExit(1 if failures else 0)
