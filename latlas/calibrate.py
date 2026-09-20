"""Calibrate the delay model on independent, ground-truthed measurements.

Fitting the detour law from the client's own measurement campaign is a
bootstrapping problem: the model parameters depend on where the client is, and
the location estimate depends on the parameters. Left unconstrained, the fit
resolves that ridge badly -- during development it settled on a combination that
placed the estimate at the very edge of the physically feasible region, 840 km
from the answer.

The anchor mesh removes the circularity. It contains ~145,000 round-trip times
between two endpoints whose coordinates are BOTH published, so every pair gives a
directly observed (distance, RTT) sample with no unknown in it. Fitting here is
therefore an ordinary regression on ground truth, and the resulting parameters
describe how the public internet actually behaves rather than how one client's
network happens to look.

The client's own data can still refine the fit locally; this module supplies the
prior it refines from, and the two are compared in validation.
"""

from __future__ import annotations

import json
import os
import numpy as np

from .geo import haversine_km
from .model import DelayParams, K_FIBER, fit_params
from .mesh_truth import VantageCase

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
#: The calibrated delay law is shipped inside the package: it is derived from
#: public RIPE Atlas measurements and a fresh install must not silently fall back
#: to uncalibrated defaults just because a build-time artifact was missing.
BUNDLED_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "delay_params.json")
PARAMS_PATH = os.path.join(DATA_DIR, "delay_params.json")


def pairs_from_cases(cases: list[VantageCase]) -> tuple[np.ndarray, np.ndarray]:
    """(distance_km, rtt_ms) for every probe/target pair in the mesh."""
    d, r = [], []
    for c in cases:
        for (_key, la, lo, rtt) in c.observations:
            d.append(float(haversine_km(c.lat, c.lon, la, lo)))
            r.append(float(rtt))
    return np.asarray(d), np.asarray(r)


def calibrate_from_mesh(cases: list[VantageCase], progress=print) -> DelayParams:
    d, r = pairs_from_cases(cases)
    progress(f"  calibrating on {len(d):,} ground-truthed pairs")
    # Start from a physically sensible prior so the grid brackets the right
    # region: ~1.8x stretch on the propagation term plus a few ms of overhead.
    base = DelayParams(k=K_FIBER, beta=3.0, mu=float(np.log(1.8)), sigma=0.45)
    params = fit_params(
        d, r, base=base,
        beta_grid=np.concatenate([[0.0], np.linspace(0.2, 8.0, 24)]),
        mu_grid=np.linspace(-0.4, 1.6, 26),
        sigma_grid=np.linspace(0.1, 1.2, 24),
    )
    progress(f"  fitted beta={params.beta:.2f} ms  median detour="
             f"{np.exp(params.mu):.3f}x  sigma={params.sigma:.3f}")
    return params


def save_params(params: DelayParams, path: str = PARAMS_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params.as_dict(), f, indent=1)
    return path


def load_params(path: str | None = None) -> DelayParams | None:
    """Load the delay law, preferring a local recalibration over the bundled one."""
    for candidate in ([path] if path else [PARAMS_PATH, BUNDLED_PARAMS]):
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                d = json.load(f)
            return DelayParams(
                k=float(d.get("k_km_per_ms", K_FIBER)),
                beta=float(d["beta_ms"]),
                mu=float(d["mu_log_detour"]),
                sigma=float(d["sigma_log_detour"]),
                pi_out=float(d.get("pi_outlier", 0.04)),
                mu_out=float(d.get("mu_out_log_rtt", 3.6)),
                sigma_out=float(d.get("sigma_out_log_rtt", 1.0)),
            )
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return None


def main(argv=None) -> int:
    from .mesh_truth import load_mesh_cases, mesh_summary
    import argparse
    ap = argparse.ArgumentParser(description="calibrate the delay model on the anchor mesh")
    ap.add_argument("--cases", type=int, default=None)
    args = ap.parse_args(argv)
    cases = load_mesh_cases()
    if args.cases:
        cases = cases[:args.cases]
    print(json.dumps(mesh_summary(cases), indent=1))
    params = calibrate_from_mesh(cases)
    p = save_params(params)
    print(f"wrote {p}")
    print(json.dumps(params.as_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
