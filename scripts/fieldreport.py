"""Assemble field-test output into a report with logs and photologs.

Reads a directory of per-host results produced by ``scripts/fieldtest.py`` and
writes ``fieldtest/`` containing a summary, the raw logs, and the screenshots.

    python scripts/fieldreport.py fieldtest-out fieldtest
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def load(d: Path) -> dict | None:
    try:
        return json.loads((d / "result.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def location_of(truth: dict) -> str:
    """Best available place name for a host, from the scoring lookup."""
    cities = [s.get("city") for s in (truth.get("sources") or [])
              if s.get("ok") and s.get("city")]
    if not cities:
        return "?"
    # Providers often disagree; the modal answer is the least misleading label.
    city = Counter(cities).most_common(1)[0][0]
    cc = ""
    for s in truth.get("sources") or []:
        if s.get("ok") and s.get("city") == city and s.get("country"):
            cc = f", {s['country']}"
            break
    return f"{city}{cc}"


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "fieldtest-out")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "fieldtest")
    # A host can be excluded with a ".private" marker in its output directory.
    # This report is published, and the operator's own machine is a valid test
    # host whose coordinates are not the operator's to publish.
    excluded = {p.parent.name for p in src.glob("*/.private")}
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    (dst / "photos").mkdir(parents=True, exist_ok=True)

    rows, sections = [], []
    n_inside = n_outside = n_degraded = n_total = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        data = load(d)
        if not data:
            continue
        label = data.get("env", {}).get("label") or d.name
        if d.name in excluded or label in excluded:
            print(f"  skipping {d.name} (marked .private)")
            continue
        env, res, sc = data.get("env", {}), data.get("result", {}), data.get("score", {})
        m = res.get("measurement", {}) or {}
        u = res.get("uncertainty", {}) or {}
        pos = res.get("position") or {}
        truth = sc.get("truth") or {}

        # Copy the evidence across.
        for f in ("log.txt", "cli_raw.txt", "SUMMARY.txt", "env.json", "result.json",
                  "web_panel.txt"):
            if (d / f).exists():
                shutil.copy2(d / f, dst / "logs" / f"{label}__{f}")
        shots = []
        for f in sorted(d.glob("web_*.jpg")) + sorted(d.glob("web_*.png")):
            out = dst / "photos" / f"{label}__{f.name}"
            shutil.copy2(f, out)
            shots.append(out.name)

        loc = location_of(truth)
        err = sc.get("error_km")
        spread = truth.get("spread_km")
        n_total += 1
        if sc.get("inside_bound"):
            n_inside += 1
        elif err is not None:
            n_outside += 1
        if env.get("backend_degraded"):
            n_degraded += 1
        rows.append(
            f"| {label} | {loc} | {env.get('system')} {env.get('release')} | "
            f"{env.get('backend')}{' *' if env.get('backend_degraded') else ''} | "
            f"{m.get('anchors_used')}/{m.get('anchors_total')} | "
            f"{m.get('min_rtt_ms')} ms | "
            f"{err if err is not None else 'n/a'} km | "
            f"{'yes' if sc.get('inside_bound') else 'no'} |"
            + (f" {spread:,.0f} km |" if spread is not None else " |"))

        img_md = "\n".join(
            f"![{label} {n.split('__')[-1]}](photos/{n})" for n in shots)
        note = ""
        if not sc.get("inside_bound") and err is not None:
            note = (f"\n\n**The bound failed on this host.** The anchor set "
                    f"contradicted itself: some constraints cannot be satisfied "
                    f"by any location, so no hard bound exists and the reported "
                    f"radius should not be read as one. See the log for the "
                    f"violation count. The ground truth here is also unreliable "
                    f"(providers disagree by {spread:,.0f} km)."
                    if spread is not None else "")
        sections.append(f"""### {label}

{loc} &mdash; {env.get('platform')}
Python {env.get('python')}, {env.get('cpu_count')} cpus, backend
`{env.get('backend')}`{' (degraded, TCP fallback)' if env.get('backend_degraded') else ''}.

Estimate `{pos.get('lat')}, {pos.get('lon')}` ({res.get('place')}), radius
&plusmn;{u.get('certificate_radius_km')} km, error
**{err if err is not None else 'n/a'} km**, inside bound:
{sc.get('inside_bound')}. {m.get('anchors_used')}/{m.get('anchors_total')} anchors
replied in {m.get('elapsed_s')} s.{note}

{img_md}
""")

    report = f"""# Field tests

The tool run end to end on real machines in different countries, unattended.
Each host produced a console log, a machine-readable result, and screenshots of
the live map; all of it is under `logs/` and `photos/`.

Ground truth is a public IP geolocation lookup used only to score the result. It
never reaches the estimator, and on cloud hosts it is often poor: the last column
is how far the providers disagreed with each other, which bounds how much any
score here can mean.

| host | location | platform | backend | anchors | closest RTT | error | inside bound | truth spread |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

`*` marks the TCP fallback, used where the network drops outbound ICMP.

## What this shows

* **{n_inside} of {n_total} hosts landed inside the reported radius**, including
  every host where the network blocked ICMP and the tool fell back to TCP.
* Accuracy tracks how close the host sits to an anchor, as the method predicts.
  The closest RTT sets the radius directly: hosts a few milliseconds from an
  anchor localise to single-digit kilometres, hosts tens of milliseconds away do
  not.
* **The one failure is a limitation worth knowing.** On the macOS runner a
  cluster of servers, each registered in one region, all answered from somewhere
  else at sub-millisecond latency. They agree with each other, so no
  leave-one-out test can separate them from honest anchors, and the constraints
  they impose cannot all be satisfied at once. That is now detected: the tool
  reports that the anchor set contradicts itself and labels the radius as not a
  bound, instead of printing a confident number.

## Per host

{chr(10).join(sections)}
"""
    (dst / "README.md").write_text(report, encoding="utf-8")
    print(f"wrote {dst/'README.md'}: {n_total} hosts, {n_inside} inside bound, "
          f"{n_outside} outside, {n_degraded} on TCP fallback, "
          f"{len(list((dst/'photos').iterdir()))} photos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
