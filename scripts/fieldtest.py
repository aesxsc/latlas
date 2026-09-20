"""Run latlas end to end on whatever machine this is, and record the evidence.

Produces a directory containing, for one host:

  env.json        OS, kernel, CPU, Python, selected probe backend
  cli.txt         full console output of a real scan
  result.json     the estimate, the independently looked-up truth, and the error
  web_*.png       screenshots of the live map while scanning and once finished
  SUMMARY.txt     one-screen human summary

Intended to run unattended on CI runners and throwaway hosts, so every step is
wrapped: a failure is recorded in the summary rather than aborting the run.

The ground-truth lookup is used for scoring only and is never fed to the
estimator; see latlas/groundtruth.py.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Runnable straight from a checkout as well as from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "fieldtest-out")
LABEL = sys.argv[2] if len(sys.argv) > 2 else platform.node()
SAMPLES = int(os.environ.get("LATLAS_FIELD_SAMPLES", "8"))
WEB_PORT = int(os.environ.get("LATLAS_FIELD_PORT", "8099"))

log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def capture(cmd: list[str], timeout: int = 300) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<failed: {type(e).__name__}: {e}>"


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------


def collect_env() -> dict:
    from latlas.icmp import select_backend

    info = {
        "label": LABEL,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    try:
        info["uname"] = " ".join(platform.uname())
    except Exception:  # noqa: BLE001
        pass
    try:
        info["backend"] = select_backend().name
        info["backend_degraded"] = bool(select_backend().degraded)
    except Exception as e:  # noqa: BLE001
        info["backend"] = f"<none: {e}>"
        info["backend_degraded"] = True
    if shutil.which("nproc"):
        info["nproc"] = capture(["nproc"]).strip()
    return info


def run_cli() -> dict:
    """A real scan through the console entry point."""
    log("--- latlas --quick --json ---")
    raw = capture([sys.executable, "-m", "latlas", "--quick", "--json",
                   "--samples", str(SAMPLES)], timeout=900)
    write("cli_raw.txt", raw)
    # Take the first complete JSON value in the stream: a traceback on stderr can
    # precede it, and trailing text must not break parsing.
    try:
        start = raw.index("{")
        data, _end = json.JSONDecoder().raw_decode(raw[start:])
    except Exception as e:  # noqa: BLE001
        log(f"could not parse CLI json: {e}")
        return {"error": f"unparsable output: {e}"}
    if not data.get("measurement", {}).get("anchors_used"):
        log("WARNING: no anchor replied; check for a network that blocks ICMP")
    log(f"estimate {data['position']['lat']:.4f},{data['position']['lon']:.4f}"
        f"  place={data.get('place')!r}"
        f"  bound=+/-{data['uncertainty']['certificate_radius_km']:.0f} km"
        f"  anchors={data['measurement']['anchors_used']}"
        f"/{data['measurement']['anchors_total']}")
    return data


def run_web_photos() -> list[str]:
    """Start the web UI and photograph it mid-scan and once finished."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        log(f"playwright unavailable, skipping photolog: {e}")
        return []

    from latlas.web import Scanner, Handler
    from latlas.web import _Server

    log("--- web UI photolog ---")
    scanner = Scanner(samples=SAMPLES, workers=64, quick=True)
    handler = type("H", (Handler,), {"scanner": scanner})
    httpd = _Server(("127.0.0.1", WEB_PORT), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.6)
    scanner.start()

    shots: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 900},
                                    device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{WEB_PORT}/", wait_until="load")
            # One frame while the map is still filling in.
            deadline = time.time() + 90
            while time.time() < deadline:
                txt = page.inner_text("#status")
                if txt not in ("idle", "scanning"):
                    break
                if page.inner_text("#anch") not in ("\u2014", "") and \
                        time.time() > deadline - 60:
                    break
                page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / "web_scanning.jpg"), type="jpeg",
                            quality=72)
            shots.append("web_scanning.jpg")
            log(f"  captured web_scanning.jpg at {page.inner_text('#anch')}")

            deadline = time.time() + 400
            while time.time() < deadline:
                if page.inner_text("#status") == "complete":
                    break
                page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "web_final.jpg"), type="jpeg",
                            quality=72)
            shots.append("web_final.jpg")
            state = {
                "status": page.inner_text("#status"),
                "coord": page.inner_text("#coord"),
                "place": page.inner_text("#place"),
                "bound": page.inner_text("#bound"),
                "anchors": page.inner_text("#anch"),
                "hist": page.inner_text("#histlbl"),
                "spark": page.inner_text("#sparklbl"),
            }
            write("web_panel.txt", "\n".join(f"{k}: {v}" for k, v in state.items()))
            log(f"  captured web_final.png  status={state['status']} "
                f"coord={state['coord']}")
            browser.close()
    except Exception as e:  # noqa: BLE001
        log(f"  photolog failed: {type(e).__name__}: {e}")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return shots


def score(data: dict) -> dict:
    """Compare the estimate against an independently looked-up location."""
    out: dict = {}
    try:
        from latlas.groundtruth import lookup
        from latlas.geo import haversine_km
        gt = lookup(progress=lambda s: None)
        out["truth"] = gt.to_dict()
        p = data.get("position") or {}
        if p:
            out["error_km"] = round(float(haversine_km(
                p["lat"], p["lon"], gt.lat, gt.lon)), 1)
        cert = (data.get("uncertainty") or {}).get("certificate_radius_km")
        if cert and "error_km" in out:
            out["inside_bound"] = bool(out["error_km"] <= cert)
        log(f"truth {gt.lat:.4f},{gt.lon:.4f} (spread {gt.spread_km:.0f} km)  "
            f"error {out.get('error_km')} km  inside_bound={out.get('inside_bound')}")
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        log(f"ground-truth scoring failed: {e}")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== latlas field test: {LABEL} ===")

    env = collect_env()
    log(f"platform : {env['platform']}")
    log(f"python   : {env['python']} ({env['python_impl']})")
    log(f"cpus     : {env['cpu_count']}")
    log(f"backend  : {env['backend']}"
        + ("  [DEGRADED - tcp fallback]" if env.get("backend_degraded") else ""))
    write("env.json", json.dumps(env, indent=1))

    result = run_cli()
    scored = score(result) if "error" not in result else {}

    payload = {"env": env, "result": result, "score": scored,
               "photolog": run_web_photos()}
    write("result.json", json.dumps(payload, indent=1))

    m = result.get("measurement", {})
    summary = [
        f"host        {env['hostname']}  ({env['platform']})",
        f"python      {env['python']}   backend {env['backend']}"
        + ("  DEGRADED" if env.get("backend_degraded") else ""),
        f"anchors     {m.get('anchors_used')}/{m.get('anchors_total')} replied"
        f"  ({(m.get('response_rate') or 0)*100:.1f}%)",
        f"closest RTT {m.get('min_rtt_ms')} ms   scan {m.get('elapsed_s')} s",
        f"estimate    {result.get('position')}   {result.get('place')}",
        f"bound       +/-{(result.get('uncertainty') or {}).get('certificate_radius_km')} km",
        f"truth       {scored.get('truth', {}).get('lat')},"
        f"{scored.get('truth', {}).get('lon')}",
        f"error       {scored.get('error_km')} km"
        f"   inside_bound={scored.get('inside_bound')}",
        f"photolog    {', '.join(payload['photolog']) or 'none'}",
    ]
    write("SUMMARY.txt", "\n".join(summary) + "\n")
    write("log.txt", "\n".join(log_lines) + "\n")
    log("")
    log("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
