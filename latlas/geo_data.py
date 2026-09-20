"""Bundled reference data: the anchor set and a city gazetteer.

Both ship inside the package, so the tool has everything it needs to answer the
question without a discovery or download step. The anchor set is frozen and
versioned; the gazetteer turns raw coordinates into something a human can read
("76 km north of İzmir, Türkiye"), which is the only part of the output that
wants a place-name lookup.

Neither file says anything about where this machine is. The anchors pose the
question ("how far am I from each of these known points?"); the gazetteer only
labels the answer afterwards.

Attribution: city data is GeoNames (https://www.geonames.org/), CC BY 4.0.
Anchor coordinates are published by RIPE NCC and Ookla respectively.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .anchors import Anchor
from .geo import great_circle_km, latlon_to_unit

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANCHORS_PATH = os.path.join(DATA_DIR, "anchors.json")
CITIES_PATH = os.path.join(DATA_DIR, "cities.json")

_COMPASS = ("north", "north-east", "east", "south-east",
            "south", "south-west", "west", "north-west")


@lru_cache(maxsize=1)
def baked_anchors() -> tuple[Anchor, ...]:
    """The frozen global anchor set shipped with the package."""
    if not os.path.exists(ANCHORS_PATH):
        raise FileNotFoundError(
            f"bundled anchor set missing at {ANCHORS_PATH}; the package data "
            f"directory was not installed")
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return tuple(Anchor.from_dict(a) for a in d["anchors"])


@lru_cache(maxsize=1)
def anchors_hash() -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps([a.to_dict() for a in baked_anchors()], sort_keys=True).encode()
    ).hexdigest()[:16]


@dataclass(frozen=True)
class Place:
    """A named populated place with its offset from the query point."""

    name: str
    country: str
    country_code: str
    lat: float
    lon: float
    population: int
    distance_km: float
    bearing_deg: float

    @property
    def compass(self) -> str:
        return _COMPASS[int(((self.bearing_deg + 22.5) % 360.0) // 45.0)]

    def describe(self) -> str:
        cc = f", {self.country}" if self.country else ""
        pop = f"  (pop. {self.population:,})" if self.population else ""
        return f"{self.name}{cc} \u2014 {self.distance_km:,.0f} km {self.compass}{pop}"


@lru_cache(maxsize=1)
def _city_table():
    """(names, country_codes, populations, lat, lon, unit_vectors, country_names)."""
    with open(CITIES_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    rows = d["cities"]
    names = [r[0] for r in rows]
    cc = [r[3] for r in rows]
    pop = np.array([r[4] for r in rows], dtype=np.float64)
    lat = np.array([r[1] for r in rows], dtype=np.float64)
    lon = np.array([r[2] for r in rows], dtype=np.float64)
    return names, cc, pop, lat, lon, latlon_to_unit(lat, lon), d.get("countries", {})


def nearest_places(lat: float, lon: float, k: int = 5,
                   min_population: int = 0) -> list[Place]:
    """The ``k`` nearest populated places to a coordinate, nearest first."""
    names, cc, pop, clat, clon, units, countries = _city_table()
    here = latlon_to_unit(lat, lon)
    d = great_circle_km(here[None, :], units)
    if min_population > 0:
        d = np.where(pop >= min_population, d, np.inf)

    # Local east/north frame at the query point, for a correct bearing anywhere
    # on the sphere (including across the antimeridian).
    east = np.cross(np.array([0.0, 0.0, 1.0]), here)
    if float(np.linalg.norm(east)) < 1e-9:
        east = np.array([1.0, 0.0, 0.0])
    east = east / np.linalg.norm(east)
    north = np.cross(here, east)

    out: list[Place] = []
    for i in np.argsort(d)[:max(0, k)]:
        if not math.isfinite(float(d[i])):
            continue
        v = units[i]
        delta = v - here * float(np.dot(here, v))
        bearing = math.degrees(math.atan2(float(np.dot(delta, east)),
                                          float(np.dot(delta, north)))) % 360.0
        out.append(Place(
            name=names[i], country=countries.get(cc[i], cc[i]), country_code=cc[i],
            lat=float(clat[i]), lon=float(clon[i]), population=int(pop[i]),
            distance_km=float(d[i]), bearing_deg=bearing))
    return out


def describe_point(lat: float, lon: float, k: int = 3) -> str:
    """A one-line human description of a coordinate."""
    places = nearest_places(lat, lon, k=k)
    if not places:
        return f"{lat:.3f}, {lon:.3f}"
    p = places[0]
    if p.distance_km < 15.0:
        return f"at {p.name}, {p.country}"
    return f"{p.distance_km:,.0f} km {p.compass} of {p.name}, {p.country}"
