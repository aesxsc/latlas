"""Command line interface.

The default action needs no arguments and no setup:

    python -m latlas                 # scan, print coordinates and nearby places
    python -m latlas web             # live map at http://0.0.0.0:8080
    python -m latlas --json          # same answer, machine-readable

The anchor set is bundled, so there is nothing to discover or download first.
The remaining subcommands are the measurement and validation harness used to
produce and check the accuracy figures; they are not needed to get an answer.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time

from . import __version__
from .anchors import cache_dir
from .geo_data import (ANCHORS_PATH, anchors_hash, baked_anchors,
                       nearest_places)
from .sense import DEFAULT_SAMPLES, DEFAULT_TIMEOUT_MS, DEFAULT_WORKERS, SenseResult, sense

#: Artifacts are written to a directory that is actually writable: the repo's
#: data/ when running from a checkout, the user cache when installed.
VALIDATION_PATH = os.path.join(cache_dir(), "validation.json")
MESH_PATH = os.path.join(cache_dir(), "mesh_cases.json")
RESULT_PATH = os.path.join(cache_dir(), "estimate.json")


def _fmt_latlon(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}\u00b0{ns}, {abs(lon):.4f}\u00b0{ew}"


def _calibration() -> dict | None:
    if not os.path.exists(VALIDATION_PATH):
        return None
    try:
        with open(VALIDATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("mesh")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Default action: just answer the question
# --------------------------------------------------------------------------


def cmd_sense(args) -> int:
    if not args.json:
        print("latlas \u2014 locating this machine from round-trip times only")
        print("no IP geolocation, no client address, no Wi-Fi/GPS/timezone/ISP data\n")

    res = sense(samples=args.samples, workers=args.workers,
                timeout_ms=args.timeout, quick=args.quick,
                places=args.places,
                progress=(lambda s: None) if args.json else print)

    if args.json:
        payload = res.to_dict()
        payload["calibration"] = _calibration()
        print(json.dumps(payload, indent=1))
        return 0

    _print_answer(res)
    if args.report:
        print()
        _print_full_report(res)
    return 0


def _print_answer(res: SenseResult) -> None:
    W = 74
    print("\u2500" * W)
    print(f"  POSITION      {_fmt_latlon(res.lat, res.lon)}")
    print(f"  NEAREST PLACE {res.place}")
    hard = res.estimate.certificate.get("is_hard_bound", True)
    if hard:
        print(f"  ACCURACY      \u00b1{res.certificate_radius_km:,.0f} km "
              f"(speed-of-light bound; holds regardless of routing or model error)")
    else:
        print(f"  ACCURACY      ~{res.certificate_radius_km:,.0f} km "
              f"(NOT a bound: the anchor set contradicts itself here)")
    print("\u2500" * W)

    print("\n  nearest places to the estimate")
    for p in res.nearest:
        pop = f"pop. {p.population:,}" if p.population else ""
        print(f"    {p.distance_km:>8,.0f} km {p.compass:<11s} "
              f"{p.name + ', ' + p.country:<38s} {pop}")

    print(f"\n  measured      {res.anchors_used}/{res.anchors_total} anchors replied "
          f"({100*res.response_rate:.1f}%), closest {res.min_rtt_ms:.2f} ms, "
          f"{res.elapsed_s:.0f} s")

    # The model-refined estimate is reported, but not led with: validation showed
    # it is less accurate than the certificate centre, sometimes much less.
    if res.separation_km > 150:
        print(f"  model estimate {_fmt_latlon(res.model_lat, res.model_lon)} "
              f"({res.separation_km:,.0f} km from the estimate above)")
        print("                 \u26a0 the fitted delay model disagrees with the "
              "physical bound here;")
        print("                   on this network it is the less reliable of the two")

    cal = _calibration()
    if cal:
        fc = (cal.get("feasible_centroid_error_km") or {}).get("median")
        cont = (cal.get("containment") or {}).get("certificate")
        bits = []
        if fc is not None:
            bits.append(f"typical error median {fc:,.0f} km")
        if cont is not None:
            bits.append(f"bound contained the truth {100*cont:.1f}% of the time")
        if bits:
            print(f"\n  measured on {cal.get('n_ok', '?')} independent real vantage "
                  f"points: {', '.join(bits)}")
    else:
        print("\n  (run `python -m latlas validate mesh` once to measure this "
              "tool's accuracy)")


def _print_full_report(res: SenseResult) -> None:
    from .report import render_full_map, summarise
    d = res.to_dict()
    # summarise() expects the estimator's own schema plus its metadata.
    result = res.estimate.to_dict()
    meta = {"backend": res.campaign.backend,
            "samples_per_anchor": res.campaign.samples_per_anchor,
            "duration_s": round(res.elapsed_s, 1)}
    ameta = {"count": res.anchors_total, "hash": anchors_hash(),
             "countries": len({a.country for a in baked_anchors() if a.country}),
             "distinct_operators": len({a.operator for a in baked_anchors()
                                        if a.operator})}
    print(summarise(result, campaign_meta=meta, anchors_meta=ameta,
                    calibration=_calibration()))
    print(render_full_map(result, [(a.lat, a.lon) for a in baked_anchors()]))


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------


def cmd_web(args) -> int:
    from .web import serve
    serve(host=args.host, port=args.port, samples=args.samples,
          workers=args.workers, quick=args.quick, autostart=not args.no_scan)
    return 0


# --------------------------------------------------------------------------
# Informational / harness commands
# --------------------------------------------------------------------------


def cmd_anchors(args) -> int:
    a = baked_anchors()
    countries: dict[str, int] = {}
    for x in a:
        countries[x.country] = countries.get(x.country, 0) + 1
    print(f"bundled anchor set: {len(a)} endpoints, {len(countries)} countries, "
          f"{len({x.operator for x in a if x.operator})} distinct networks")
    print(f"  hash     {anchors_hash()}")
    print(f"  file     {ANCHORS_PATH}")
    print(f"  sources  " + ", ".join(
        f"{s}={sum(1 for x in a if x.source == s)}"
        for s in sorted({x.source for x in a})))
    print("  top countries: " + ", ".join(
        f"{k} {v}" for k, v in sorted(countries.items(), key=lambda kv: -kv[1])[:12]))
    if args.nearest_to:
        lat, lon = args.nearest_to
        print(f"\nanchors nearest to {lat},{lon}:")
        for p in nearest_places(lat, lon, k=10):
            print(f"  {p.distance_km:>8,.0f} km  {p.name}, {p.country}")
    return 0


def cmd_isolation(args) -> int:
    from .groundtruth import check_isolation
    bad = check_isolation()
    if bad:
        print("ISOLATION VIOLATION: estimator modules import ground truth:", bad)
        return 1
    print("isolation OK: no estimator module imports latlas.groundtruth")
    return 0


def cmd_validate(args) -> int:
    from .measure import Campaign
    from .estimate import estimate_location
    from .validate import (MeshValidation, containment,
                           nearest_anchor_jackknife, sample_bootstrap,
                           validate_mesh)

    kwargs = {"exploration": 15000, "final_points": 60000, "fit_iterations": 2}
    payload: dict = {}
    if os.path.exists(VALIDATION_PATH):
        try:
            with open(VALIDATION_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            payload = {}
    payload["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if args.what in ("mesh", "all"):
        from .mesh_truth import load_mesh_cases, mesh_summary
        if not os.path.exists(MESH_PATH):
            print("no mesh cases; run: python -m latlas mesh")
            return 2
        cases = load_mesh_cases(MESH_PATH)
        print(f"mesh validation over {len(cases)} real vantage points")
        print(json.dumps(mesh_summary(cases), indent=1))
        res = validate_mesh(cases, limit=args.cases, **kwargs)
        payload["mesh"] = res.summary()
        payload["mesh_rows"] = res.rows
        payload["mesh_failures"] = res.failures[:40]
        print(json.dumps(payload["mesh"], indent=1))

    if args.what in ("self", "all"):
        from .groundtruth import lookup
        print("scanning this machine")
        res = sense(samples=args.samples, workers=args.workers,
                    timeout_ms=args.timeout, quick=True, progress=print)
        truth = lookup(progress=lambda s: print(s)).to_dict()
        print(f"independent ground truth: {truth['lat']}, {truth['lon']} "
              f"(spread {truth['spread_km']} km)")
        c = containment(res.estimate, truth["lat"], truth["lon"])
        payload["self"] = {
            "truth": truth,
            "error_km": round(float(c["error_km"]), 1),
            "feasible_centroid_error_km": round(float(c["feasible_centroid_error_km"]), 1),
            "cap_centre_error_km": round(float(c["cap_centre_error_km"]), 1),
            "containment": c,
            "result": res.estimate.to_dict(),
        }
        print(f"  feasible-centroid error "
              f"{payload['self']['feasible_centroid_error_km']} km"
              f" | cap-centre {payload['self']['cap_centre_error_km']} km"
              f" | model-estimate {payload['self']['error_km']} km"
              f" | inside bound: {c['in_certificate']}")
        print("  jackknife over nearest anchors")
        payload["jackknife"] = nearest_anchor_jackknife(
            res.campaign.measurements, truth["lat"], truth["lon"], **kwargs)
        print("  sample bootstrap")
        payload["bootstrap"] = sample_bootstrap(
            res.campaign.measurements, truth["lat"], truth["lon"], **kwargs)
        print(json.dumps(payload["bootstrap"], indent=1))

    with open(VALIDATION_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nvalidation written to {VALIDATION_PATH}")
    return 0


def cmd_mesh(args) -> int:
    from .mesh_truth import build_mesh_matrix, load_mesh_cases, mesh_summary
    if args.reuse and os.path.exists(MESH_PATH):
        cases = load_mesh_cases(MESH_PATH)
        print(f"reused {len(cases)} vantage cases")
    else:
        cases = build_mesh_matrix(max_measurements=args.max_measurements,
                                  min_targets=args.min_targets)
    print(json.dumps(mesh_summary(cases), indent=1))
    return 0


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate_from_mesh, save_params
    from .mesh_truth import load_mesh_cases, mesh_summary
    cases = load_mesh_cases(MESH_PATH)
    if args.cases:
        cases = cases[:args.cases]
    print(json.dumps(mesh_summary(cases), indent=1))
    params = calibrate_from_mesh(cases)
    print(f"wrote {save_params(params)}")
    print(json.dumps(params.as_dict(), indent=1))
    return 0


def cmd_discover(args) -> int:
    from .build_anchors import main as build_main
    argv = ["--target", str(args.target), "--samples", str(args.samples),
            "--workers", str(args.workers)]
    if args.reuse:
        argv.append("--reuse")
    return build_main(argv)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="latlas",
        description="Find where this machine is using only ICMP round-trip times "
                    "to known anchors. The anchor set is bundled; just run it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python -m latlas                 scan and print the position\n"
               "  python -m latlas web             live map on 0.0.0.0:8080\n"
               "  python -m latlas --json          machine-readable output\n"
               "  python -m latlas validate mesh   measure this tool's accuracy\n")
    p.add_argument("--version", action="version", version=f"latlas {__version__}")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help="ICMP echoes per anchor (default %(default)s)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="parallel probes (default %(default)s)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MS,
                   help="per-echo timeout in ms (default %(default)s)")
    p.add_argument("--quick", action="store_true",
                   help="coarser candidate grid; noticeably faster")
    p.add_argument("--json", action="store_true", help="emit JSON, no prose")
    p.add_argument("--report", action="store_true",
                   help="also print the full technical report and world map")
    p.add_argument("--places", type=int, default=5,
                   help="how many nearby places to name (default %(default)s)")
    p.set_defaults(func=cmd_sense)

    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("web", help="live map UI (default host 0.0.0.0:8080)")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    sp.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    sp.add_argument("--quick", action="store_true")
    sp.add_argument("--no-scan", action="store_true",
                    help="serve the page but wait for the scan button")
    sp.set_defaults(func=cmd_web)

    sp = sub.add_parser("anchors", help="describe the bundled anchor set")
    sp.add_argument("--nearest-to", type=float, nargs=2, metavar=("LAT", "LON"))
    sp.set_defaults(func=cmd_anchors)

    sp = sub.add_parser("isolation", help="prove ground truth cannot reach the estimator")
    sp.set_defaults(func=cmd_isolation)

    sp = sub.add_parser("validate", help="accuracy / calibration / robustness harness")
    sp.add_argument("what", choices=["mesh", "self", "all"], nargs="?", default="all")
    sp.add_argument("--cases", type=int, default=None)
    sp.add_argument("--samples", type=int, default=8)
    sp.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    sp.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_MS)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("mesh", help="build the independent validation set")
    sp.add_argument("--reuse", action="store_true")
    sp.add_argument("--max_measurements", type=int, default=200)
    sp.add_argument("--min_targets", type=int, default=12)
    sp.set_defaults(func=cmd_mesh)

    sp = sub.add_parser("calibrate", help="refit the delay law from ground-truthed pairs")
    sp.add_argument("--cases", type=int, default=None)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("discover", help="rebuild the anchor set (network-heavy; rarely needed)")
    sp.add_argument("--target", type=int, default=1600)
    sp.add_argument("--samples", type=int, default=3)
    sp.add_argument("--workers", type=int, default=96)
    sp.add_argument("--reuse", action="store_true")
    sp.set_defaults(func=cmd_discover)
    return p


def _silence_stdout() -> None:
    """Point stdout at the null device so the interpreter's final flush cannot fail."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except Exception:  # noqa: BLE001 - best effort, we are already exiting
        pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code = args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (BrokenPipeError, OSError) as e:
        # `latlas | head` closes the pipe early, which is ordinary usage rather
        # than a failure. POSIX surfaces it as BrokenPipeError; Windows reports
        # the same condition as OSError EINVAL, which would otherwise print a
        # traceback for a completely normal pipeline.
        if isinstance(e, BrokenPipeError) or getattr(e, "errno", None) in (
                errno.EPIPE, errno.EINVAL):
            _silence_stdout()
            return 0
        raise
    try:
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        _silence_stdout()
        return 0
    return code


if __name__ == "__main__":
    raise SystemExit(main())
