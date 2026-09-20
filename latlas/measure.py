"""Concurrent latency measurement engine.

Design notes
------------
*Concurrency without self-interference.* Probing is I/O bound and dominated by
timeouts on unreachable hosts, so the work is parallelised with a thread pool.
The pool size is deliberately modest: the goal is to saturate the wide-area path
without queueing probes behind each other on the local NIC, which would corrupt
the very quantity being measured. Per-host samples are spaced by a small
inter-sample delay for the same reason.

*Repetition.* A single echo is a noisy estimate of the path's propagation floor.
Each anchor is probed ``samples`` times; the estimator consumes the whole series
so it can distinguish an unqueued floor from a congested path, and so that loss
rate is available as evidence.

*Two rounds.* An optional second campaign quantifies measurement stability and
gives the validation stage something to compare against.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .anchors import Anchor
from .icmp import ProbeSeries, Prober


@dataclass
class AnchorMeasurement:
    """Everything learned about one anchor in one campaign."""

    key: str
    ip: str
    lat: float
    lon: float
    source: str
    country: str = ""
    city: str = ""
    operator: str = ""
    location_confidence: str = "medium"
    samples_ms: list[float | None] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    resolved_ip: str = ""
    started: float = 0.0
    duration_s: float = 0.0

    # ---- derived views -------------------------------------------------

    @property
    def rtts(self) -> list[float]:
        return [s for s in self.samples_ms if s is not None]

    @property
    def sent(self) -> int:
        return len(self.samples_ms)

    @property
    def received(self) -> int:
        return len(self.rtts)

    @property
    def loss(self) -> float:
        return 1.0 - self.received / self.sent if self.sent else 1.0

    def floor_ms(self, q: float = 0.15) -> float | None:
        r = sorted(self.rtts)
        if not r:
            return None
        if len(r) <= 3:
            return r[0]
        k = max(1, min(len(r) - 1, int(round(q * len(r)))))
        return sum(r[:k]) / k

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.rtts) if self.rtts else None

    @property
    def jitter_ms(self) -> float | None:
        r = self.rtts
        if len(r) < 2:
            return None
        return max(r) - min(r)

    def to_dict(self) -> dict:
        d = {
            "key": self.key, "ip": self.ip, "lat": self.lat, "lon": self.lon,
            "source": self.source, "country": self.country, "city": self.city,
            "operator": self.operator,
            "location_confidence": self.location_confidence,
            "resolved_ip": self.resolved_ip,
            "samples_ms": self.samples_ms, "errors": self.errors,
            "duration_s": round(self.duration_s, 3),
        }
        d["floor_ms"] = self.floor_ms()
        d["median_ms"] = self.median_ms
        d["jitter_ms"] = self.jitter_ms
        d["loss"] = round(self.loss, 4)
        return d

    @staticmethod
    def from_dict(d: dict) -> "AnchorMeasurement":
        m = AnchorMeasurement(
            key=d["key"], ip=d["ip"], lat=d["lat"], lon=d["lon"],
            source=d.get("source", ""), country=d.get("country", ""),
            city=d.get("city", ""), operator=d.get("operator", ""),
            location_confidence=d.get("location_confidence", "medium"),
            samples_ms=list(d.get("samples_ms", [])),
            errors=list(d.get("errors", [])),
            resolved_ip=d.get("resolved_ip", ""),
            duration_s=d.get("duration_s", 0.0),
        )
        return m


@dataclass
class Campaign:
    """A complete measurement round over an anchor set."""

    started_iso: str
    finished_iso: str
    backend: str
    samples_per_anchor: int
    anchors_hash: str
    measurements: list[AnchorMeasurement]

    def to_dict(self) -> dict:
        return {
            "started": self.started_iso, "finished": self.finished_iso,
            "backend": self.backend, "samples_per_anchor": self.samples_per_anchor,
            "anchors_hash": self.anchors_hash,
            "measurements": [m.to_dict() for m in self.measurements],
        }

    @staticmethod
    def from_dict(d: dict) -> "Campaign":
        return Campaign(
            started_iso=d["started"], finished_iso=d["finished"],
            backend=d["backend"], samples_per_anchor=d["samples_per_anchor"],
            anchors_hash=d.get("anchors_hash", ""),
            measurements=[AnchorMeasurement.from_dict(m) for m in d["measurements"]],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        # Sample series make these files large; keep the writer honest about it.

    @staticmethod
    def load(path: str) -> "Campaign":
        with open(path, "r", encoding="utf-8") as f:
            return Campaign.from_dict(json.load(f))


def _iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def run_campaign(anchors: Sequence[Anchor], *,
                 samples: int = 12,
                 timeout_ms: int = 1200,
                 max_workers: int = 64,
                 inter_sample_s: float = 0.02,
                 backend_name: str | None = None,
                 anchors_hash: str = "",
                 progress: Callable[[str], None] = print,
                 progress_every: int = 150,
                 on_batch: Callable[[int, int], None] | None = None,
                 on_anchor: Callable[[AnchorMeasurement], None] | None = None) -> Campaign:
    """Probe every anchor ``samples`` times and return the campaign record.

    ``on_batch(done, total)`` fires at each progress milestone so a live front end
    can show a scan in motion without polling the campaign object.
    ``on_anchor`` fires once per completed anchor, which is what lets a live view
    plot each result as it lands rather than in batches.
    """
    t_start = time.time()
    total = len(anchors)
    done = 0
    lock = threading.Lock()
    results: list[AnchorMeasurement] = []

    prober = Prober(backend_name=backend_name)
    progress(f"  probe backend: {prober.name}  workers={max_workers} "
             f"samples/anchor={samples}")

    def work(a: Anchor) -> AnchorMeasurement:
        nonlocal done
        t0 = time.time()
        try:
            s = prober.series(a.ip, n=samples, interval=inter_sample_s,
                              timeout_ms=timeout_ms)
        except Exception as e:  # noqa: BLE001 - one bad host must not kill a run
            s = ProbeSeries(ip=a.ip, samples_ms=[None] * samples,
                            errors=[f"exception:{type(e).__name__}"])
        m = AnchorMeasurement(
            key=a.key, ip=a.ip, lat=a.lat, lon=a.lon, source=a.source,
            country=a.country, city=a.city, operator=a.operator,
            location_confidence=a.location_confidence,
            samples_ms=s.samples_ms, errors=s.errors, started=t0,
            duration_s=time.time() - t0,
        )
        with lock:
            done += 1
            if on_anchor is not None:
                on_anchor(m)
            if done % progress_every == 0 or done == total:
                progress(f"    {done}/{total} anchors probed "
                         f"({time.time() - t_start:.0f}s)")
                if on_batch is not None:
                    on_batch(done, total)
        return m

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(work, anchors))
    finally:
        prober.close()

    t_end = time.time()
    progress(f"  campaign finished in {t_end - t_start:.1f}s; "
             f"{sum(1 for m in results if m.received)}/{total} anchors replied")
    return Campaign(
        started_iso=_iso(t_start), finished_iso=_iso(t_end), backend=prober.name,
        samples_per_anchor=samples, anchors_hash=anchors_hash,
        measurements=results,
    )


def campaign_summary(c: Campaign) -> dict:
    """Aggregate health statistics for a campaign."""
    n = len(c.measurements)
    resp = [m for m in c.measurements if m.received > 0]
    floors = [m.floor_ms() for m in resp if m.floor_ms() is not None]
    losses = [m.loss for m in c.measurements]
    return {
        "anchors": n,
        "responded": len(resp),
        "response_rate": round(len(resp) / n, 4) if n else 0.0,
        "mean_loss": round(sum(losses) / n, 4) if n else 0.0,
        "min_floor_ms": min(floors) if floors else None,
        "median_floor_ms": statistics.median(floors) if floors else None,
        "max_floor_ms": max(floors) if floors else None,
        "by_source": _count_by(c.measurements, "source", lambda m: m.received > 0),
        "by_confidence": _count_by(c.measurements, "location_confidence",
                                   lambda m: m.received > 0),
    }


def _count_by(ms, attr, pred) -> dict:
    out: dict[str, list[int]] = {}
    for m in ms:
        k = getattr(m, attr) or "?"
        e = out.setdefault(k, [0, 0])
        e[0] += 1
        if pred(m):
            e[1] += 1
    return {k: {"total": v[0], "responded": v[1]} for k, v in sorted(out.items())}
