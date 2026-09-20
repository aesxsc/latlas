"""High-level API: measure, estimate, and describe where this machine is.

This is the single entry point the CLI and the web UI both call, so the two can
never drift apart. One call does everything: probe the bundled anchor set, infer
a position, work out the uncertainty, and name the places around it.

The headline answer is the **certificate centroid** -- the centre of the region
that the speed-of-light constraints permit -- not the model-refined point. That
ordering is empirical, not aesthetic: the certificate centre was the more
accurate of the two both on the bundled anchor set (259 km vs 790 km against a
known location) and across 400 independent real vantage points (median 100 km vs
113 km). The model-refined estimate is still computed and reported, because on a
well-behaved network it is sharper, but it is not what the tool leads with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .anchors import Anchor
from .estimate import Estimate, estimate_location
from .geo_data import (Place, anchors_hash, baked_anchors, describe_point,
                       nearest_places)
from .measure import Campaign, campaign_summary, run_campaign

#: Echoes per anchor for a normal run. Enough for a stable floor estimate
#: without making a scan feel slow.
DEFAULT_SAMPLES = 10
DEFAULT_WORKERS = 64
DEFAULT_TIMEOUT_MS = 1200


def _noop(_msg: str) -> None:
    pass


@dataclass
class SenseResult:
    """Everything one scan produced, in a form both front ends can render."""

    lat: float                      # headline position (certificate centroid)
    lon: float
    place: str                      # human description of the headline position
    nearest: list[Place]
    certificate_radius_km: float
    credible_radius_km: float | None
    model_lat: float
    model_lon: float
    model_place: str
    separation_km: float
    estimate: Estimate
    campaign: Campaign
    summary: dict
    anchors_used: int
    anchors_total: int
    response_rate: float
    min_rtt_ms: float
    elapsed_s: float
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "position": {"lat": round(self.lat, 5), "lon": round(self.lon, 5)},
            "place": self.place,
            "nearest": [
                {"name": p.name, "country": p.country, "lat": round(p.lat, 5),
                 "lon": round(p.lon, 5), "population": p.population,
                 "distance_km": round(p.distance_km, 1),
                 "bearing_deg": round(p.bearing_deg, 1), "compass": p.compass}
                for p in self.nearest
            ],
            "uncertainty": {
                "certificate_radius_km": round(self.certificate_radius_km, 1),
                "is_hard_bound": bool(
                    self.estimate.certificate.get("is_hard_bound", True)),
                "contradictions": self.estimate.certificate.get("contradictions", 0),
                "credible_radius_km": (None if self.credible_radius_km is None
                                       else round(self.credible_radius_km, 1)),
                # Centre of the enclosing cap that the radius describes; distinct
                # from the reported position, which is the region's centroid.
                "region_centre": {
                    "lat": round(self.estimate.cap_lat, 5),
                    "lon": round(self.estimate.cap_lon, 5)},
                "method": "intersection of speed-of-light spherical caps",
            },
            "model_refined": {
                "lat": round(self.model_lat, 5), "lon": round(self.model_lon, 5),
                "place": self.model_place,
                "separation_km": round(self.separation_km, 1),
            },
            "measurement": {
                "anchors_total": self.anchors_total,
                "anchors_used": self.anchors_used,
                "response_rate": round(self.response_rate, 4),
                "min_rtt_ms": round(self.min_rtt_ms, 2),
                "elapsed_s": round(self.elapsed_s, 1),
                "backend": self.campaign.backend,
                "samples_per_anchor": self.campaign.samples_per_anchor,
            },
            "delay_model": self.params,
            "diagnostics": self.estimate.diagnostics,
        }


def sense(*,
          anchors: list[Anchor] | None = None,
          samples: int = DEFAULT_SAMPLES,
          workers: int = DEFAULT_WORKERS,
          timeout_ms: int = DEFAULT_TIMEOUT_MS,
          quick: bool = False,
          places: int = 5,
          progress: Callable[[str], None] = _noop,
          on_progress: Callable[[dict], None] | None = None,
          inter_sample_s: float = 0.02) -> SenseResult:
    """Measure the bundled anchors and return where the machine appears to be.

    ``on_progress`` receives a dict per batch of probed anchors, which is what the
    web UI streams to the browser so the map can move while the scan runs.
    """
    t0 = time.time()
    anchor_list = list(anchors) if anchors is not None else list(baked_anchors())
    progress(f"probing {len(anchor_list)} anchors "
             f"({samples} echoes each, {workers} in parallel)")

    def _progress_probe(done: int, total: int) -> None:
        if on_progress:
            on_progress({"type": "progress", "probed": done, "total": total,
                         "elapsed_s": round(time.time() - t0, 1)})

    campaign = run_campaign(
        anchor_list, samples=samples, timeout_ms=timeout_ms,
        max_workers=workers, inter_sample_s=inter_sample_s,
        anchors_hash=anchors_hash(), progress=_noop,
        on_batch=_progress_probe)

    live = [m for m in campaign.measurements if m.received > 0]
    progress(f"  {len(live)}/{len(anchor_list)} anchors replied in "
             f"{time.time()-t0:.0f}s; inferring position")
    if on_progress:
        on_progress({"type": "stage", "stage": "estimating"})

    est = estimate_location(
        campaign.measurements, progress=lambda s: progress(s),
        **({"exploration": 15000, "final_points": 60000, "fit_iterations": 2}
           if quick else {}))

    head_lat, head_lon = est.cert_lat, est.cert_lon
    near = nearest_places(head_lat, head_lon, k=places)
    model_near = nearest_places(est.lat, est.lon, k=1)
    elapsed = time.time() - t0
    progress(f"  done in {elapsed:.0f}s")

    summ = campaign_summary(campaign)
    result = SenseResult(
        lat=head_lat, lon=head_lon,
        place=describe_point(head_lat, head_lon),
        nearest=near,
        certificate_radius_km=est.cert_radius_km,
        credible_radius_km=(est.credible.get("p90") or {}).get("enclosing_cap_radius_km"),
        model_lat=est.lat, model_lon=est.lon,
        model_place=(model_near[0].name if model_near else ""),
        separation_km=est.separation_km(),
        estimate=est, campaign=campaign, summary=summ,
        anchors_used=summ["responded"], anchors_total=summ["anchors"],
        response_rate=summ["response_rate"], min_rtt_ms=summ["min_floor_ms"],
        elapsed_s=elapsed, params=est.params.as_dict(),
    )
    if on_progress:
        on_progress({"type": "result", "data": result.to_dict()})
    return result
