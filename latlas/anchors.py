"""Anchor discovery: assembling a global set of location-known probe targets.

Two sources are used, both of which publish coordinates and are *designed* to be
repeatedly probed by third parties, so the measurement load is sanctioned:

``ripe-atlas``
    RIPE Atlas software anchors. ~1070 are live with an IPv4 address. The
    controlling body publishes the host city and a coordinate pair, and the
    hardware exists specifically to answer network measurements.

``speedtest``
    Ookla Speedtest server list. Each entry carries the operator-declared city
    and coordinate. This adds consumer/ISP network diversity that the
    datacenter-heavy Atlas fleet lacks.

Both sources have the same two failure modes and both are handled explicitly:

*Aliasing* — a hostname that is a CDN or hosting alias resolves to a front-end
that may not be where the coordinates say. Such entries are dropped by
hostname pattern where detectable, and the residue is absorbed by the
estimator's robustness term rather than being trusted blindly.

*Anycast* — a target advertising one coordinate while actually answering from a
different site on every continent violates the speed-of-light cone for its
claimed position. Nodes whose measurements are physically inconsistent are
demoted by the estimator's robust weighting; a host that is faster than light
from the claimed position is positive evidence of anycast.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Sequence

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def cache_dir() -> str:
    """Writable directory for measurement artifacts, created on demand.

    When running from a checkout, artifacts belong beside the source so they are
    easy to inspect. An installed wheel lives in ``site-packages``, which is
    typically read-only and is the wrong place for per-machine data anyway, so
    there the cache moves to the platform's user cache location. ``LATLAS_DATA``
    overrides both.
    """
    env = os.environ.get("LATLAS_DATA")
    if env:
        os.makedirs(env, exist_ok=True)
        return env
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo, "data")
    if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
        return candidate
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache")
    out = os.path.join(base, "latlas")
    os.makedirs(out, exist_ok=True)
    return out

_UA = {
    "User-Agent": "latlas/1.0 (latency-only geolocation research; contact: local user)",
    "Accept": "application/json",
}
_UA_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.speedtest.net/",
    "Origin": "https://www.speedtest.net",
}

# Hostname fragments that indicate a hosting/CDN front-end rather than the
# operator's own site-bearing equipment.
_ALIAS_MARKERS = (
    "prod.hosts.ooklaserver.net",
    ".cdn.",
    "cloudflare",
    "fastly",
    "akamai",
    "edgekey",
    "edgesuite",
    "llnwd",
    "cachefly",
    "incapdns",
)


@dataclass
class Anchor:
    """A probe target whose physical location is known a priori."""

    id: str
    lat: float
    lon: float
    ip: str
    source: str
    country: str = ""
    city: str = ""
    operator: str = ""
    #: 'high' for purpose-built measurement hosts, 'medium' for operator-declared
    #: third-party hosts whose front-end may not match the published coordinate.
    location_confidence: str = "medium"
    hostname: str = ""
    asn: int | None = None

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Anchor":
        return Anchor(**{k: v for k, v in d.items() if k in Anchor.__dataclass_fields__})


def _get_json(url: str, headers: dict, tries: int = 4, pause: float = 1.5):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=35) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001 - network layer, retried
            last = e
            if attempt < tries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {last}")


# --------------------------------------------------------------------------
# RIPE Atlas
# --------------------------------------------------------------------------


def fetch_ripe_atlas(progress=print) -> list[Anchor]:
    """All live, IPv4-addressed RIPE Atlas software anchors."""
    out: list[Anchor] = []
    url = "https://atlas.ripe.net/api/v2/anchors/?page_size=100"
    page = 0
    while url:
        d = _get_json(url, _UA)
        for a in d["results"]:
            if a.get("is_disabled") or a.get("date_decommissioned"):
                continue
            ip = a.get("ip_v4")
            geo = a.get("geometry") or {}
            coords = geo.get("coordinates") or []
            if not ip or len(coords) != 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            out.append(Anchor(
                id=str(a["id"]), lat=lat, lon=lon, ip=ip, source="ripe-atlas",
                country=a.get("country") or "", city=a.get("city") or "",
                operator=a.get("company") or "", location_confidence="high",
                hostname=a.get("fqdn") or a.get("hostname") or "",
                asn=a.get("as_v4"),
            ))
        url = d.get("next")
        page += 1
        progress(f"  ripe-atlas page {page}: {len(out)} live anchors")
    return out


# --------------------------------------------------------------------------
# Speedtest
# --------------------------------------------------------------------------

_GRID_STEP_LAT = 10
_GRID_STEP_LON = 12


def _speedtest_url(lat: float, lon: float, limit: int = 100,
                   distance: int = 2000) -> str:
    # The ``distance`` cap lifts the response ceiling from ~27 to the full
    # ``limit``; without it most grid samples return a near-duplicate
    # neighbourhood and the sweep yields a third of the available servers.
    return ("https://www.speedtest.net/api/js/servers?engine=js"
            f"&limit={limit}&lat={lat}&lon={lon}&distance={distance}")


def fetch_speedtest(cache_path: str | None = None, progress=print,
                    max_requests: int = 600) -> list[Anchor]:
    """Harvest the global speedtest server list by sampling a lat/lon grid.

    The endpoint returns servers ranked by distance from the supplied point, so
    a single call only ever reveals a local neighbourhood; sweeping a global
    grid and unioning by server id is what yields worldwide coverage.
    """
    cache = os.path.join(DATA_DIR, "speedtest_raw.json") if cache_path is None else cache_path
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            raw = json.load(f)
        progress(f"  speedtest: {len(raw)} entries from cache")
    else:
        seen: dict[str, dict] = {}
        grid = [(la, lo)
                for la in range(-54, 75, _GRID_STEP_LAT)
                for lo in range(-180, 180, _GRID_STEP_LON)]
        n_req = 0
        for la, lo in grid:
            if n_req >= max_requests:
                break
            try:
                d = _get_json(_speedtest_url(la, lo), _UA_BROWSER, tries=2, pause=1.0)
                n_req += 1
            except Exception:
                continue
            if isinstance(d, list):
                for s in d:
                    if "id" in s:
                        seen[s["id"]] = s
            time.sleep(0.25)
        raw = list(seen.values())
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        progress(f"  speedtest: {len(raw)} unique servers from {n_req} requests")

    out: list[Anchor] = []
    for s in raw:
        host = (s.get("host") or "").strip()
        if not host:
            continue
        low = host.lower()
        if any(m in low for m in _ALIAS_MARKERS):
            continue
        hostname = host.rsplit(":", 1)[0] if ":" in host else host
        try:
            lat = float(s["lat"]); lon = float(s["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        out.append(Anchor(
            id=str(s["id"]), lat=lat, lon=lon, ip=hostname, source="speedtest",
            country=s.get("cc") or "", city=s.get("name") or "",
            operator=s.get("sponsor") or "", location_confidence="medium",
            hostname=hostname,
        ))
    return out


# --------------------------------------------------------------------------
# Spatial diversity selection
# --------------------------------------------------------------------------


def equal_area_cell(lat: float, lon: float, n_bands: int = 30) -> tuple[int, int]:
    """Map a coordinate to an approximately equal-area cell index.

    Latitude bands are uniform in degrees; the number of longitude divisions
    inside each band is proportional to the band's true width, so cells near the
    poles are not wildly over-represented the way they are in a plain lat/lon
    grid.
    """
    lat = max(-89.999, min(89.999, lat))
    band = int((lat + 90.0) / 180.0 * n_bands)
    band = min(band, n_bands - 1)
    band_center = -90.0 + (band + 0.5) * (180.0 / n_bands)
    n_lon = max(1, int(round(n_bands * 2 * math.cos(math.radians(band_center)))))
    lon_cell = int((lon + 180.0) / 360.0 * n_lon) % n_lon
    return band, lon_cell


def select_diverse(anchors: Sequence[Anchor], target: int, *,
                   max_per_operator: int = 8,
                   max_per_cell: int = 24,
                   n_bands: int = 30,
                   prefer_confidence: bool = True) -> list[Anchor]:
    """Choose ``target`` anchors spread across space and across networks.

    Greedy round-robin over equal-area cells, so a region that happens to be
    dense cannot crowd out a region that is sparse: every occupied cell
    contributes its first anchor before any cell contributes its second. The
    per-cell cap therefore does not hurt spatial spread, it only stops one
    metropolis from consuming the entire budget, and the per-operator cap keeps
    a single ISP from dominating a neighbourhood. Cells are ~670 km across at
    30 bands, which is the natural scale of the budget here: with ~200 occupied
    land cells, the default caps leave room for the full 1000-2000 quota while
    still guaranteeing that no cell contributes more than its share.
    """
    high = [a for a in anchors if a.location_confidence == "high"]
    med = [a for a in anchors if a.location_confidence != "high"]
    ordered = (high, med) if prefer_confidence else ([*high, *med],)

    cells: dict[tuple[int, int], list[Anchor]] = {}
    for group in ordered:
        for a in group:
            cells.setdefault(equal_area_cell(a.lat, a.lon, n_bands), []).append(a)

    # Rotate starting cell per pass for fairness across the globe.
    keys = sorted(cells.keys())
    # Purpose-built measurement hosts carry better coordinates than
    # operator-declared third-party entries, so within a cell they must come
    # first. The confidence rank is part of the sort key precisely so that the
    # later operator grouping cannot scramble it.
    for k in keys:
        cells[k].sort(key=lambda a: (0 if a.location_confidence == "high" else 1,
                                     a.operator, a.id))

    chosen: list[Anchor] = []
    op_count: dict[str, int] = {}
    taken: dict[tuple[int, int], int] = {}
    idx = {k: 0 for k in keys}
    while len(chosen) < target:
        progressed = False
        for k in keys:
            if len(chosen) >= target:
                break
            if taken.get(k, 0) >= max_per_cell:
                continue
            lst = cells[k]
            i = idx[k]
            while i < len(lst):
                a = lst[i]
                i += 1
                op = a.operator or a.key
                if op_count.get(op, 0) >= max_per_operator:
                    continue
                chosen.append(a)
                op_count[op] = op_count.get(op, 0) + 1
                taken[k] = taken.get(k, 0) + 1
                progressed = True
                break
            idx[k] = i
        if not progressed:
            break
    return chosen


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_anchors(anchors: Sequence[Anchor], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(anchors),
        "anchors": [a.to_dict() for a in anchors],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    h = hashlib.sha256(json.dumps([a.to_dict() for a in anchors],
                                  sort_keys=True).encode()).hexdigest()[:16]
    return h


def load_anchors(path: str) -> list[Anchor]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [Anchor.from_dict(x) for x in d["anchors"]]
