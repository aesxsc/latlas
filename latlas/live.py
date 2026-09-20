"""Live, incremental position estimation.

The full estimator in :mod:`latlas.estimate` is thorough but takes tens of
seconds: it refits the delay law, alternates, and evaluates hundreds of thousands
of candidate points. That is the right thing for the final answer and the wrong
thing for a map that should move while the scan runs.

This module provides a cheap approximation that can be recomputed every few dozen
anchors. It uses only the model-free half of the method -- the speed-of-light
cones -- which is also the half that carries the guarantee:

    distance <= (c/2) * RTT   for every anchor that answered

Sampling a few thousand points inside the tightest cone, keeping those that
violate the fewest constraints, and taking their centroid reproduces the shape of
the final answer closely enough to watch it converge, at roughly a hundredth of
the cost.

The running estimate is explicitly *not* the reported result. It is a preview,
and the UI labels it as such.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .estimate import MAX_KM_PER_MS, QUANTUM_MS
from .geo import (EARTH_RADIUS_KM, cap_points, great_circle_km, latlon_to_unit,
                  spherical_cap_area_km2, unit_to_latlon)


@dataclass
class LiveFix:
    """One incremental position preview."""

    lat: float
    lon: float
    cap_radius_km: float        # rigorous bound from the closest anchor so far
    region_radius_km: float     # extent of the feasible set found
    anchors_used: int
    min_rtt_ms: float
    elapsed_s: float
    sequence: int

    def to_dict(self) -> dict:
        return {
            "lat": round(self.lat, 5), "lon": round(self.lon, 5),
            "cap_radius_km": round(self.cap_radius_km, 1),
            "region_radius_km": round(self.region_radius_km, 1),
            "anchors_used": self.anchors_used,
            "min_rtt_ms": round(self.min_rtt_ms, 2),
            "elapsed_s": round(self.elapsed_s, 1),
            "sequence": self.sequence,
        }


class QuickLocator:
    """Accumulates anchor results and re-derives a cheap position on demand.

    Thread-safe: the measurement threads call :meth:`add` while the estimate is
    read from another thread.
    """

    def __init__(self, candidates: int = 2600, safety: float = 1.25,
                 min_cap_km: float = 60.0) -> None:
        self.candidates = candidates
        self.safety = safety
        self.min_cap_km = min_cap_km
        self._lock = threading.Lock()
        self._units: list[np.ndarray] = []
        self._rtt: list[float] = []
        self._started = 0.0
        self._sequence = 0

    def reset(self, started: float) -> None:
        with self._lock:
            self._units.clear()
            self._rtt.clear()
            self._started = started
            self._sequence = 0

    def add(self, lat: float, lon: float, rtt_ms: float | None) -> None:
        if rtt_ms is None:
            return
        u = latlon_to_unit(lat, lon)
        with self._lock:
            self._units.append(u)
            self._rtt.append(float(rtt_ms))

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._rtt)

    def estimate(self) -> LiveFix | None:
        """Current best preview, or None until enough anchors have answered."""
        with self._lock:
            if len(self._rtt) < 4:
                return None
            units = np.stack(self._units, axis=0)
            rtt = np.asarray(self._rtt, dtype=np.float64)
            started = self._started
            self._sequence += 1
            seq = self._sequence

        # The closest anchor alone bounds the machine to one spherical cap.
        nearest = int(np.argmin(rtt))
        cap = MAX_KM_PER_MS * (float(rtt[nearest]) + QUANTUM_MS)
        radius = np.maximum(cap, self.min_cap_km) * self.safety
        radius = min(radius, np.pi * EARTH_RADIUS_KM)

        # Anchors whose reach cannot cut this cap are vacuous and are dropped, so
        # the cost does not grow linearly with the anchor count.
        centre = units[nearest]
        reach = MAX_KM_PER_MS * (rtt + QUANTUM_MS)
        d_centre = great_circle_km(centre[None, :], units)
        binding = d_centre <= (radius + reach)
        units_b, reach_b = units[binding], reach[binding]

        pts = cap_points(centre, radius, self.candidates)
        d = great_circle_km(pts[:, None, :], units_b[None, :, :])
        counts = np.count_nonzero(d > reach_b[None, :], axis=1)
        best = int(counts.min())
        feasible = counts <= best

        centroid = pts[feasible].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-12:
            centroid = pts[feasible][0]
        else:
            centroid = centroid / norm
        lat, lon = unit_to_latlon(centroid)

        extent = float(great_circle_km(centroid[None, :], pts[feasible]).max())
        return LiveFix(
            lat=float(lat), lon=float(lon), cap_radius_km=float(cap),
            region_radius_km=extent, anchors_used=len(rtt),
            min_rtt_ms=float(rtt[nearest]),
            elapsed_s=(time.time() - started) if started else 0.0,
            sequence=seq,
        )
