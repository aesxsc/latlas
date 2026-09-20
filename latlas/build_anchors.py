"""Stage 1: discover, filter and freeze the global anchor set.

    python -m latlas.build_anchors [--target 1500] [--samples 3] [--workers 96]

Pipeline:
  fetch (RIPE Atlas + Speedtest) -> dedupe -> spatial/network diversity select
  -> liveness pre-screen by ICMP -> freeze to data/anchors.json with a hash.

The liveness screen is what makes the frozen set trustworthy: a directory entry
that never answers contributes nothing but a wasted probe slot, so a cheap
3-sample pre-screen drops dead entries before the expensive campaign runs.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latlas.anchors import (Anchor, cache_dir, fetch_ripe_atlas, fetch_speedtest,
                            load_anchors, save_anchors, select_diverse)

DATA_DIR = cache_dir()
from latlas.icmp import Prober


def dedupe(anchors: list[Anchor]) -> list[Anchor]:
    """Drop duplicate targets keyed by resolved host string."""
    seen: dict[str, Anchor] = {}
    for a in anchors:
        k = a.ip.lower()
        prev = seen.get(k)
        if prev is None:
            seen[k] = a
            continue
        # Prefer the entry we trust more.
        if prev.location_confidence != "high" and a.location_confidence == "high":
            seen[k] = a
    return list(seen.values())


def prescreen(anchors: list[Anchor], samples: int, timeout_ms: int,
              workers: int, progress=print) -> list[Anchor]:
    """Keep only anchors that answer ICMP, annotated with the screen RTT."""
    prober = Prober()
    progress(f"  pre-screen backend: {prober.name}")
    keep: list[Anchor] = []
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def chk(a: Anchor):
        nonlocal done
        try:
            s = prober.series(a.ip, n=samples, interval=0.01, timeout_ms=timeout_ms)
        except Exception:
            s = None
        with lock:
            done += 1
            if done % 200 == 0:
                progress(f"    screened {done}/{len(anchors)} ({time.time()-t0:.0f}s)")
        if s is not None and s.received > 0:
            return a
        return None

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(chk, anchors):
                if r is not None:
                    keep.append(r)
    finally:
        prober.close()
    progress(f"  pre-screen: {len(keep)}/{len(anchors)} alive "
             f"({100*len(keep)/max(1,len(anchors)):.1f}%) in {time.time()-t0:.0f}s")
    return keep


def describe(anchors: list[Anchor]) -> dict:
    cc = collections.Counter(a.country for a in anchors)
    ops = collections.Counter(a.operator for a in anchors if a.operator)
    src = collections.Counter(a.source for a in anchors)
    conf = collections.Counter(a.location_confidence for a in anchors)
    return {
        "count": len(anchors),
        "countries": len(cc),
        "distinct_operators": len(ops),
        "by_source": dict(src),
        "by_confidence": dict(conf),
        "top_countries": cc.most_common(10),
        "top_operators": ops.most_common(8),
    }


def nearest_neighbour_stats(anchors: list[Anchor]) -> dict:
    """Median/percentile great-circle distance to each anchor's nearest peer."""
    import math

    def hav(a: Anchor, b: Anchor) -> float:
        p1, p2 = math.radians(a.lat), math.radians(b.lat)
        dp = p2 - p1
        dl = math.radians(b.lon - a.lon)
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))

    # Cheap O(n^2) is fine at n ~ 2000; use a lat-sorted window to be safe.
    d = []
    for i, a in enumerate(anchors):
        best = min((hav(a, b) for j, b in enumerate(anchors) if j != i), default=0.0)
        d.append(best)
    d.sort()
    return {
        "nn_median_km": round(statistics.median(d), 1) if d else None,
        "nn_p10_km": round(d[int(0.1 * len(d))], 1) if d else None,
        "nn_p90_km": round(d[int(0.9 * len(d))], 1) if d else None,
        "nn_max_km": round(d[-1], 1) if d else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500,
                    help="desired number of anchors in the frozen set")
    ap.add_argument("--samples", type=int, default=3, help="pre-screen samples")
    ap.add_argument("--timeout", type=int, default=900, help="pre-screen timeout ms")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "anchors.json"))
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing frozen set without re-screening")
    args = ap.parse_args(argv)

    print("== anchor discovery ==")
    if args.reuse and os.path.exists(args.out):
        an = load_anchors(args.out)
        print(f"reused {len(an)} anchors from {args.out}")
        print(json.dumps(describe(an), indent=1))
        return 0

    print("fetching sources")
    ripe = fetch_ripe_atlas()
    print(f"  ripe-atlas: {len(ripe)} anchors")
    st = fetch_speedtest()
    print(f"  speedtest:  {len(st)} anchors (alias hosts removed)")
    pool = dedupe([*ripe, *st])
    print(f"  merged unique targets: {len(pool)}")
    from collections import Counter
    print("  by source:", dict(Counter(a.source for a in pool)))

    # Screen the *entire* pool for liveness, then choose the frozen set from what
    # actually answers. Screening before selection is what makes the frozen set
    # trustworthy: a directory entry that never replies would otherwise consume
    # a share of the diversity budget and contribute nothing.
    alive = prescreen(pool, args.samples, args.timeout, args.workers)
    print("  alive by source:",
          dict(Counter(a.source for a in alive)))

    # Choose the frozen set, keeping spatial and network spread. Caps are relaxed
    # progressively: a tight cap that cannot fill the quota would silently return
    # a smaller set, so we loosen until the requested count is reachable.
    alive = dedupe(alive)
    chosen = alive
    if len(alive) > args.target:
        for cell_cap, op_cap in ((24, 8), (40, 12), (60, 20), (100, 40),
                                 (10**9, 10**9)):
            trial = select_diverse(alive, args.target, max_per_operator=op_cap,
                                   max_per_cell=cell_cap)
            if len(trial) >= args.target:
                chosen = trial
                break
            chosen = trial
    chosen = dedupe(chosen)
    chosen.sort(key=lambda a: (a.source, a.id))

    h = save_anchors(chosen, args.out)
    print(f"\nfroze {len(chosen)} anchors -> {args.out}  (hash {h})")
    print(json.dumps(describe(chosen), indent=1))
    print(json.dumps(nearest_neighbour_stats(chosen), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
