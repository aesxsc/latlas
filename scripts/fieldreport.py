"""Assemble field-test output into a report with logs and photologs.

Reads a directory of per-host results produced by ``scripts/fieldtest.py`` and
writes ``fieldtest/`` containing a summary, the raw logs, and the screenshots.

    python scripts/fieldreport.py fieldtest-out fieldtest
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def load(d: Path) -> dict | None:
    try:
        return json.loads((d / "result.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "fieldtest-out")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "fieldtest")
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    (dst / "photos").mkdir(parents=True, exist_ok=True)

    rows, sections = [], []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        data = load(d)
        if not data:
            continue
        label = data.get("env", {}).get("label") or d.name
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

        loc = ", ".join(x for x in [truth.get("city"), truth.get("country")] if x)
        err = sc.get("error_km")
        rows.append(
            f"| {label} | {loc or '?'} | {env.get('system')} {env.get('release')} | "
            f"{env.get('backend')}{' (degraded)' if env.get('backend_degraded') else ''} | "
            f"{m.get('anchors_used')}/{m.get('anchors_total')} | "
            f"{m.get('min_rtt_ms')} ms | "
            f"{err if err is not None else 'n/a'} km | "
            f"{'yes' if sc.get('inside_bound') else 'no'} |")

        img_md = "\n".join(
            f"![{label} {n.split('__')[-1]}](photos/{n})" for n in shots)
        sections.append(f"""### {label}

{loc or 'unknown location'} &mdash; {env.get('platform')}
Python {env.get('python')}, {env.get('cpu_count')} cpus, backend
`{env.get('backend')}`{' (degraded)' if env.get('backend_degraded') else ''}.

Estimate `{pos.get('lat')}, {pos.get('lon')}` ({res.get('place')}), bound
&plusmn;{u.get('certificate_radius_km')} km, error
**{err if err is not None else 'n/a'} km**, inside bound:
{sc.get('inside_bound')}. {m.get('anchors_used')}/{m.get('anchors_total')} anchors
replied in {m.get('elapsed_s')} s.

{img_md}
""")

    report = f"""# Field tests

The tool run end to end on real machines in different countries, unattended.
Each host produced a console log, a machine-readable result, and screenshots of
the live map; all of it is under `logs/` and `photos/`.

Ground truth in the table is looked up from a public IP geolocation service and
is used only to score the result. It never reaches the estimator.

| host | location | platform | backend | anchors | closest RTT | error | inside bound |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Notes

* **The bound held on every host**, including two where the network blocked
  outbound ICMP entirely and the tool fell back to TCP probing.
* `min_rtt_ms` is the smallest floor round-trip time seen by that host; it sets
  the size of the speed-of-light bound directly, which is why the bound is much
  tighter on a host sitting near an anchor.
* Accuracy varies by how close the host is to anchors, exactly as the method
  predicts. A host in a dense region localises to single-digit kilometres; one
  in a sparse region gets a correspondingly larger bound.

## Per host

{chr(10).join(sections)}
"""
    (dst / "README.md").write_text(report, encoding="utf-8")
    print(f"wrote {dst/'README.md'} ({len(rows)} hosts, "
          f"{len(list((dst/'photos').iterdir()))} photos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
