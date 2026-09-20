"""Independent ground truth for validation, from the RIPE Atlas anchor mesh.

This module exists to answer one question honestly: *how accurate is the
estimator, measured against known truth on real networks?*

The estimator's own measurements are taken from a single vantage point, which
supports exactly one ground-truth comparison. The anchor mesh supplies thousands.
RIPE runs a set of ongoing "Anchoring Mesh" ping measurements, each of which has
every connected probe ping one location-known anchor; harvesting all of them and
joining on probe id yields a probe x anchor round-trip-time matrix in which every
row is a real vantage point whose coordinates are published, and every column is
a real target whose coordinates are published. Each row is therefore a genuine
latency-only geolocation problem with a known answer, measured on the public
internet rather than simulated.

Two limitations are carried into the report rather than hidden:

* Probe coordinates are the registrant's declared location, typically city
  level. Ground-truth error therefore bounds the accuracy any method can
  demonstrate, and a few kilometres of apparent error may be label error.
* The mesh targets are Atlas anchors, so the validation anchor set is smaller
  and more datacentre-biased than the frozen production set. Accuracy on the
  production set is expected to be better than what this harness measures.

The estimator must never import this module; the dependency runs one way only.
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

BASE = "https://atlas.ripe.net/api/v2"
from .anchors import cache_dir

DATA_DIR = cache_dir()
_UA = {"User-Agent": "latlas/1.0 (latency-only geolocation research)"}


def _get(url: str, tries: int = 4, timeout: int = 60):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                        timeout=timeout) as f:
                return json.loads(f.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(2.0 * (i + 1))
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(1.5 * (i + 1))
                continue
    raise RuntimeError(f"GET {url} failed: {last}")


@dataclass
class VantageCase:
    """One real vantage point with known location and measured RTTs."""

    probe_id: int
    lat: float
    lon: float
    country: str = ""
    asn: int | None = None
    #: (anchor_key, lat, lon, min_rtt_ms)
    observations: list[tuple[str, float, float, float]] = field(default_factory=list)
    n_sent: int = 0
    n_rcvd: int = 0

    @property
    def loss(self) -> float:
        return 1.0 - self.n_rcvd / self.n_sent if self.n_sent else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observations"] = [list(o) for o in self.observations]
        return d

    @staticmethod
    def from_dict(d: dict) -> "VantageCase":
        return VantageCase(
            probe_id=d["probe_id"], lat=d["lat"], lon=d["lon"],
            country=d.get("country", ""), asn=d.get("asn"),
            observations=[tuple(o) for o in d["observations"]],
            n_sent=d.get("n_sent", 0), n_rcvd=d.get("n_rcvd", 0),
        )


def fetch_mesh_measurements(progress=print, max_measurements: int = 4000) -> list[dict]:
    """All 'Anchoring Mesh' IPv4 ping measurements, ongoing ones first."""
    out: list[dict] = []
    url = (f"{BASE}/measurements/?description__contains=Anchoring+Mesh"
           f"&type=ping&page_size=100")
    while url and len(out) < max_measurements:
        d = _get(url)
        out.extend(d["results"])
        url = d.get("next")
        progress(f"  listed {len(out)}/{d['count']} mesh measurements")
    return out


def fetch_measurement_latest(msm_id: int) -> list[dict]:
    d = _get(f"{BASE}/measurements/{msm_id}/latest/")
    return d if isinstance(d, list) else []


def fetch_all_probe_coords(progress=print, cache: str | None = None,
                           max_workers: int = 8) -> dict[int, dict]:
    """Coordinates and ASN for every Atlas probe, cached to disk."""
    cache = cache or os.path.join(DATA_DIR, "atlas_probes.json")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            raw = json.load(f)
        progress(f"  probes: {len(raw)} from cache")
        return {int(k): v for k, v in raw.items()}

    first = _get(f"{BASE}/probes/?page_size=500")
    pages = (first["count"] + 499) // 500
    urls = [f"{BASE}/probes/?page_size=500&page={p}" for p in range(1, pages + 1)]
    out: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, d in enumerate(ex.map(_get, urls)):
            for p in d["results"]:
                g = p.get("geometry") or {}
                c = g.get("coordinates") or []
                out[str(p["id"])] = {
                    "lat": c[1] if len(c) == 2 else None,
                    "lon": c[0] if len(c) == 2 else None,
                    "country": p.get("country_code") or "",
                    "asn": p.get("asn_v4"),
                    "status": (p.get("status") or {}).get("name"),
                }
            if i % 5 == 0:
                progress(f"  probes page {i+1}/{len(urls)} ({len(out)} probes)")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f)
    progress(f"  probes: {len(out)} fetched and cached")
    return {int(k): v for k, v in out.items()}


def build_mesh_matrix(*, progress=print, max_measurements: int = 400,
                      workers: int = 8, min_targets: int = 12,
                      out_path: str | None = None) -> list[VantageCase]:
    """Harvest the mesh and reduce it to per-vantage-point geolocation problems."""
    from .anchors import fetch_ripe_atlas

    out_path = out_path or os.path.join(DATA_DIR, "mesh_cases.json")

    anchors = fetch_ripe_atlas(progress=lambda s: None)
    by_ip = {a.ip: a for a in anchors}
    progress(f"  {len(anchors)} anchors indexed by IPv4")

    msms = [m for m in fetch_mesh_measurements(progress=progress)
            if (m.get("status") or {}).get("name") == "Ongoing" and m.get("af") == 4]
    progress(f"  {len(msms)} ongoing IPv4 mesh measurements")
    msms = msms[:max_measurements]

    # row: probe_id -> {anchor_key: rtt}
    rows: dict[int, dict[str, float]] = collections.defaultdict(dict)
    meta: dict[int, dict] = {}
    sent: collections.Counter = collections.Counter()
    rcvd: collections.Counter = collections.Counter()
    done = 0
    lock = threading.Lock()

    def pull(m: dict):
        nonlocal done
        try:
            latest = fetch_measurement_latest(m["id"])
        except Exception:
            latest = []
        with lock:
            done += 1
            if done % 25 == 0:
                progress(f"    pulled {done}/{len(msms)} measurements")
        return m, latest

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for m, latest in ex.map(pull, msms):
            for e in latest:
                dst = e.get("dst_addr") or ""
                a = by_ip.get(dst)
                if a is None:
                    continue
                pid = e.get("prb_id")
                if pid is None:
                    continue
                rtts = [r["rtt"] for r in (e.get("result") or [])
                        if isinstance(r, dict) and "rtt" in r]
                sent[pid] += int(e.get("sent") or 0)
                rcvd[pid] += int(e.get("rcvd") or 0)
                if rtts:
                    prev = rows[pid].get(a.key)
                    v = min(rtts)
                    if prev is None or v < prev:
                        rows[pid][a.key] = v

    progress(f"  {len(rows)} probes produced at least one RTT")

    coords = fetch_all_probe_coords(progress=progress)
    a_by_key = {a.key: a for a in anchors}

    cases: list[VantageCase] = []
    for pid, obs in rows.items():
        if len(obs) < min_targets:
            continue
        c = coords.get(pid)
        if not c or c.get("lat") is None or c.get("lon") is None:
            continue
        if c["lat"] == 0 and c["lon"] == 0:
            continue
        if c.get("status") != "Connected":
            continue
        observations = []
        for k, rtt in obs.items():
            a = a_by_key.get(k)
            if a is None:
                continue
            observations.append((k, a.lat, a.lon, float(rtt)))
        if len(observations) < min_targets:
            continue
        cases.append(VantageCase(
            probe_id=pid, lat=float(c["lat"]), lon=float(c["lon"]),
            country=c.get("country") or "", asn=c.get("asn"),
            observations=observations, n_sent=sent[pid], n_rcvd=rcvd[pid],
        ))

    cases.sort(key=lambda c: -len(c.observations))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(cases), "cases": [c.to_dict() for c in cases]}, f)
    progress(f"  {len(cases)} usable vantage cases -> {out_path}")
    return cases


def load_mesh_cases(path: str | None = None) -> list[VantageCase]:
    path = path or os.path.join(DATA_DIR, "mesh_cases.json")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [VantageCase.from_dict(x) for x in d["cases"]]


def mesh_summary(cases: list[VantageCase]) -> dict:
    import statistics
    cc = collections.Counter(c.country for c in cases)
    return {
        "vantage_points": len(cases),
        "countries": len(cc),
        "top_countries": cc.most_common(12),
        "median_targets_per_vantage": round(
            statistics.median([len(c.observations) for c in cases]), 1),
        "total_measurements": sum(len(c.observations) for c in cases),
    }
