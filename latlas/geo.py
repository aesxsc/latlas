"""Spherical geometry and equal-area point generation on the globe.

Everything here works in 3-D unit vectors rather than (lat, lon) pairs: the
great-circle distance is then a single dot product, which vectorises into one
matrix multiply, and the formula stays numerically stable at both very small and
very large angular separations. A (lat, lon) -> arccos formulation loses relative
precision for nearby points, which is exactly the regime that matters when
localising to a few kilometres.

The globe is sampled with a Fibonacci lattice: it is low-discrepancy, has no
seam or pole clustering (unlike a lat/lon grid), needs no projection
bookkeeping, and is trivially parallel to refine.
"""

from __future__ import annotations

import math
import numpy as np

#: Volumetric mean radius of the Earth (IUGG), kilometres.
EARTH_RADIUS_KM = 6371.0088

#: Speed of light in vacuum, one-way kilometres per millisecond.
C_KM_PER_MS = 299.792458

#: Nominal one-way signal speed in single-mode optical fibre, km/ms. Corning
#: SMF-28 has a group index of ~1.467 at 1550 nm: 299792.458/1.467 ~= 204300 km/s.
FIBER_SPEED_KM_PER_MS = 204.3

# The two quantities below are ROUND-TRIP conversions: kilometres of path that
# one millisecond of measured RTT can account for. A one-way speed v implies a
# round-trip conversion of v/2, because the signal covers the path twice.
# Confusing the two is a silent factor-of-two error that makes modelled paths
# appear to travel at twice the speed of light.

#: Rigorous upper bound on ground distance per ms of RTT. Uses vacuum light
#: speed, so it holds for fibre, free-space optics and satellite relays alike.
#: This is the constant the localisation certificate must use.
MAX_RT_KM_PER_MS = C_KM_PER_MS / 2.0

#: Realistic round-trip conversion for terrestrial fibre routes. Tighter than
#: the vacuum bound, and therefore only usable inside the statistical model,
#: where an outlier component absorbs free-space and satellite paths.
FIBER_RT_KM_PER_MS = FIBER_SPEED_KM_PER_MS / 2.0

#: Backwards-compatible alias used by earlier revisions; keeps a single source
#: of truth for the fibre round-trip figure.
FIBER_KM_PER_MS = FIBER_RT_KM_PER_MS

#: Radians of angular radius corresponding to one kilometre of ground distance.
_KM_PER_RAD = EARTH_RADIUS_KM


def latlon_to_unit(lat_deg, lon_deg) -> np.ndarray:
    """Convert degrees to unit vectors. Accepts scalars or arrays; returns (..., 3)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def unit_to_latlon(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`latlon_to_unit`."""
    u = np.asarray(u, dtype=np.float64)
    lat = np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
    return lat, lon


def great_circle_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distances between unit vectors.

    ``a`` is (..., 3) and ``b`` is (..., 3); broadcasting rules apply, so
    ``a`` (N,1,3) against ``b`` (1,M,3) yields an (N,M) distance matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.einsum("...i,...i->...", a, b)
    # chord = |a - b| = 2 sin(theta/2); distance = R * theta.
    chord = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(dot, -1.0, 1.0)))
    return 2.0 * _KM_PER_RAD * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance for degree inputs; stable at short range.

    Scalars or arrays on either side. Converting with ``math.radians`` would make
    ``lat1`` scalar-only while its partners broadcast, which is exactly the kind
    of asymmetry that produces a ``TypeError`` deep inside a caller; everything
    goes through numpy here so all four arguments behave the same way.
    """
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    l1 = np.radians(np.asarray(lon1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    l2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dp = p2 - p1
    dl = l2 - l1
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * _KM_PER_RAD * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform points on the unit sphere as an (n, 3) array.

    Zero-clustering and seam-free, unlike a latitude/longitude grid, which makes
    it the natural way to sample the globe without introducing a geographic bias
    of the kind the estimator is required not to have.
    """
    i = np.arange(n, dtype=np.float64)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * i / golden
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)


def _orthonormal_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``center`` into a right-handed frame."""
    c = np.asarray(center, dtype=np.float64)
    ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, c)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return e1, e2


def cap_points(center: np.ndarray, radius_km: float, n: int) -> np.ndarray:
    """``n`` near-uniform points inside the spherical cap of given radius.

    Uniform over the cap means the local polar angle has density proportional to
    sin(t), so cos(t) is drawn uniformly on [cos(radius), 1]. Using the golden
    angle for the azimuth makes the set low-discrepancy rather than random,
    which removes clustering artefacts at any sample count.
    """
    c = np.asarray(center, dtype=np.float64)
    c = c / np.linalg.norm(c)
    theta_max = min(math.pi, radius_km / _KM_PER_RAD)
    if theta_max <= 0:
        return c.reshape(1, 3).copy()
    i = np.arange(n, dtype=np.float64)
    u = (i + 0.5) / n
    cos_t = 1.0 - u * (1.0 - math.cos(theta_max))
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = i * golden_angle
    local = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)
    e1, e2 = _orthonormal_basis(c)
    return local[:, 0:1] * e1 + local[:, 1:2] * e2 + local[:, 2:3] * c


def spherical_cap_area_km2(radius_km: float) -> float:
    """Area of a spherical cap on the Earth."""
    t = min(math.pi, radius_km / _KM_PER_RAD)
    return 2.0 * math.pi * EARTH_RADIUS_KM ** 2 * (1.0 - math.cos(t))


def diameter_km(points: np.ndarray) -> float:
    """Largest pairwise great-circle separation within a point set."""
    if len(points) < 2:
        return 0.0
    # Exact for small sets; for large sets a farthest-pair-from-centroid bound is
    # unnecessary because feasible sets here are small by construction.
    best = 0.0
    step = max(1, len(points) // 400)
    sub = points[::step]
    d = great_circle_km(sub[:, None, :], sub[None, :, :])
    return float(d.max()) if d.size else 0.0


def unit_from_bearing(center: np.ndarray, bearing_deg: float, dist_km: float) -> np.ndarray:
    """Point at ``dist_km`` from ``center`` along ``bearing_deg`` (clockwise from north)."""
    c = np.asarray(center, dtype=np.float64)
    c = c / np.linalg.norm(c)
    e1, e2 = _orthonormal_basis(c)
    # e1 is east-ish, e2 = c x e1 is north-ish; verify orientation.
    north = np.cross(e1, c)
    east = np.cross(c, north)
    b = math.radians(bearing_deg)
    t = dist_km / _KM_PER_RAD
    direction = math.cos(b) * north + math.sin(b) * east
    return c * math.cos(t) + direction * math.sin(t)
