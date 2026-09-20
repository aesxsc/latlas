"""Independent ground truth, for SCORING ONLY.

Isolation is the whole point of this module. Latency-only geolocation is defined
by what it refuses to use, so the true location must never reach the estimator
through any path. This module therefore:

* is never imported by :mod:`latlas.estimate`, :mod:`latlas.model` or any other
  estimator module (enforced by :func:`check_isolation`);
* is only ever invoked by the validation/reporting layer, and only when the user
  explicitly asks for a scoring run.

Public IP geolocation is itself imprecise in exactly the way that matters here:
consumer IP blocks are commonly registered at a regional hub rather than the
subscriber's city. Multiple independent providers are therefore queried and the
spread is reported alongside the consensus, because a validation result can
never be sharper than the truth it is scored against.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import urllib.request
from dataclasses import dataclass, field

_UA = {"User-Agent": "latlas/1.0 (validation only)"}

_PROVIDERS = (
    ("ip-api", "http://ip-api.com/json/",
     lambda d: (d.get("lat"), d.get("lon"), d.get("city"), d.get("countryCode"))),
    ("ipinfo", "https://ipinfo.io/json",
     lambda d: ((d.get("loc") or ",").split(",")[0] or None,
                (d.get("loc") or ",").split(",")[1] if "," in (d.get("loc") or "") else None,
                d.get("city"), d.get("country"))),
    ("ipwho", "https://ipwho.is/",
     lambda d: (d.get("latitude"), d.get("longitude"), d.get("city"), d.get("country_code"))),
    ("freeipapi", "https://freeipapi.com/api/json",
     lambda d: (d.get("latitude"), d.get("longitude"), d.get("cityName"), d.get("countryCode"))),
)


@dataclass
class GroundTruth:
    """Consensus location with an explicit, measured disagreement."""

    lat: float
    lon: float
    samples: list[dict] = field(default_factory=list)
    spread_km: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {"lat": round(self.lat, 5), "lon": round(self.lon, 5),
                "spread_km": round(self.spread_km, 1),
                "n_sources": len(self.samples),
                "sources": self.samples, "note": self.note}


def _haversine(a_lat, a_lon, b_lat, b_lon) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(h)))


def _fetch(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                    timeout=15) as f:
            return json.loads(f.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - provider availability is not our problem
        return None


def lookup(progress=print) -> GroundTruth:
    """Query several independent providers and return a consensus location."""
    pts: list[tuple[float, float, str, str]] = []
    samples: list[dict] = []
    for name, url, extract in _PROVIDERS:
        d = _fetch(url)
        if not isinstance(d, dict):
            samples.append({"provider": name, "ok": False})
            continue
        try:
            lat, lon, city, cc = extract(d)
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            samples.append({"provider": name, "ok": False})
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            samples.append({"provider": name, "ok": False})
            continue
        pts.append((lat, lon, city or "", cc or ""))
        samples.append({"provider": name, "ok": True, "lat": round(lat, 4),
                        "lon": round(lon, 4), "city": city, "country": cc})
        progress(f"    {name:10s} -> {lat:+.4f}, {lon:+.4f}  {city}, {cc}")

    if not pts:
        raise RuntimeError("no geolocation provider answered")

    lat = statistics.median(p[0] for p in pts)
    lon = statistics.median(p[1] for p in pts)
    spread = max((_haversine(lat, lon, p[0], p[1]) for p in pts), default=0.0)
    note = ""
    if len(pts) == 1:
        note = "single provider; accuracy unverifiable"
    elif spread > 250:
        note = ("providers disagree by more than 250 km, which is common when an "
                "ISP registers subscriber blocks at a regional hub; the true "
                "location may be anywhere in this spread")
    return GroundTruth(lat=lat, lon=lon, samples=samples,
                       spread_km=float(spread), note=note)


# --------------------------------------------------------------------------
# Isolation guard
# --------------------------------------------------------------------------

_FORBIDDEN_IMPORTERS = ("latlas.estimate", "latlas.model", "latlas.geo",
                        "latlas.measure", "latlas.icmp", "latlas.anchors",
                        "latlas.build_anchors")


def check_isolation(package_dir: str | None = None) -> list[str]:
    """Static check that no estimator module imports this one.

    Returns the list of offending files; empty means the separation holds.
    """
    import re
    pkg = package_dir or os.path.dirname(os.path.abspath(__file__))
    bad: list[str] = []
    pat = re.compile(r"^\s*(?:from\s+\.\s*groundtruth|import\s+.*groundtruth)",
                     re.MULTILINE)
    for fn in sorted(os.listdir(pkg)):
        if not fn.endswith(".py") or fn == "groundtruth.py":
            continue
        stem = f"latlas.{fn[:-3]}"
        if stem not in _FORBIDDEN_IMPORTERS:
            continue
        with open(os.path.join(pkg, fn), "r", encoding="utf-8") as f:
            if pat.search(f.read()):
                bad.append(fn)
    return bad
