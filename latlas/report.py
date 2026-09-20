"""Human-readable reporting: terminal summary plus a coarse world map.

The map is an equirectangular sketch whose only job is to let a reader sanity
check the answer at a glance: where the anchors are, how far the reported
uncertainty reaches, and which way the estimate leans. It is not a precision
instrument, and the numeric tables are the authoritative output.

Character cells are roughly twice as tall as wide, so a 4:1 cell grid renders an
approximately correct 2:1 equirectangular world.
"""

from __future__ import annotations

import math
from typing import Sequence

from .geo import haversine_km, latlon_to_unit, unit_from_bearing

_MARK_ANCHOR = {1: "\u00b7", 2: ":", 3: "*"}   # . : * by occupancy


def render_map(anchor_latlon: Sequence[tuple[float, float]],
               est_lat: float, est_lon: float, *,
               certificate: dict | None = None,
               p90: dict | None = None,
               modes: Sequence[tuple[float, float]] = (),
               truth: tuple[float, float] | None = None,
               width: int = 128, height: int = 32,
               title: str = "") -> str:
    """Render an equirectangular ASCII map of the estimate and its uncertainty."""
    w, h = width, height
    grid = [[" "] * w for _ in range(h)]

    def plot(lat: float, lon: float, ch: str) -> None:
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return
        x = int((lon + 180.0) / 360.0 * (w - 1))
        y = int((90.0 - lat) / 180.0 * (h - 1))
        if 0 <= x < w and 0 <= y < h:
            grid[y][x] = ch

    # Anchor occupancy gives the reader the coverage context.
    counts: dict[tuple[int, int], int] = {}
    for lat, lon in anchor_latlon:
        x = int((lon + 180.0) / 360.0 * (w - 1))
        y = int((90.0 - lat) / 180.0 * (h - 1))
        counts[(x, y)] = counts.get((x, y), 0) + 1
    for (x, y), n in counts.items():
        grid[y][x] = _MARK_ANCHOR.get(min(n, 3), "*")

    def ring(centre_lat: float, centre_lon: float, radius_km: float, ch: str) -> None:
        if radius_km <= 0 or radius_km > 20000:
            return
        c = latlon_to_unit(centre_lat, centre_lon)
        # The ring must follow the *edge of the region the system actually
        # reported*, which is a spherical cap, not a screen-space ellipse.
        deg_step = max(1.0, 360.0 / (radius_km / 20.0 + 8.0))
        b = 0.0
        while b < 360.0:
            p = unit_from_bearing(c, b, radius_km)
            u = p / max(1e-12, float((p ** 2).sum()) ** 0.5)
            lat = math.degrees(math.asin(max(-1.0, min(1.0, float(u[2])))))
            lon = math.degrees(math.atan2(float(u[1]), float(u[0])))
            plot(lat, lon, ch)
            b += deg_step

    if certificate:
        ring(certificate["centre_lat"], certificate["centre_lon"],
             certificate["radius_km"], "#")
    if p90:
        ring(p90["centre_lat"], p90["centre_lon"],
             p90["enclosing_cap_radius_km"], "+")
    for lat, lon in modes:
        plot(lat, lon, "o")
    if truth:
        plot(truth[0], truth[1], "X")
    plot(est_lat, est_lon, "O")

    lines = []
    if title:
        lines.append(title)
    lines.append("+" + "-" * w + "+")
    for y, row in enumerate(grid):
        label = ""
        if y % 8 == 0:
            label = f"{int(90 - 180.0 * y / (h - 1)):+03d}"
        lines.append("|" + "".join(row) + "|" + label)
    lon_axis = [" "] * w
    for lon in range(-180, 181, 30):
        x = int((lon + 180.0) / 360.0 * (w - 1))
        s = f"{lon:+d}"
        for i, ch in enumerate(s):
            if 0 <= x + i < w:
                lon_axis[x + i] = ch
    lines.append("+" + "-" * w + "+")
    lines.append(" " + "".join(lon_axis))
    lines.append(" legend:  " + " ".join(
        [". : * anchor density", "# speed-of-light certificate",
         "+ 90% credible cap", "O estimate", "o mode", "X true location"]))
    return "\n".join(lines)


def _fmt_km(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.0f} km"


def _fmt_latlon(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}\u00b0{ns}, {abs(lon):.4f}\u00b0{ew}"


def _calibration_section(cal: dict) -> list[str]:
    """State the system's measured reliability, from its own validation run.

    Reporting uncertainty is where a system like this is most likely to mislead,
    so the empirically observed behaviour is printed next to the theoretical
    claim. The separation matters: the certificate is a bound and held up, while
    the model-based credible regions did not.
    """
    out: list[str] = []
    n = cal.get("n_ok") or cal.get("n_cases")
    if not n:
        return out
    out.append(f"CALIBRATION (measured on {n} independent real vantage points)")
    fc = cal.get("feasible_centroid_error_km") or {}
    if fc:
        out.append(f"  reported estimate error  : median {fc.get('median')} km, "
                   f"p75 {fc.get('p75')} km, p90 {fc.get('p90')} km")
    cc = cal.get("cap_centre_error_km") or {}
    if cc:
        out.append(f"  (region centre would be  : median {cc.get('median')} km)")
    pm = cal.get("posterior_mean_error_km") or {}
    if pm:
        out.append(f"  (model-refined would be  : median {pm.get('median')} km)")
    cr = (cal.get("certificate_radius_km") or {}).get("median")
    if cr is not None:
        out.append(f"  median uncertainty radius: {cr:,.0f} km")
    cont = cal.get("containment") or {}
    if cont.get("certificate") is not None:
        out.append(f"  certificate containment   : {100 * cont['certificate']:.1f}% "
                   f"(theory requires it to hold always)")
    for key, label in (("p50_region", "50%"), ("p90_region", "90%")):
        v = cont.get(key)
        if v is not None:
            nominal = 0.5 if key == "p50_region" else 0.9
            flag = ("  <-- OVER-CONFIDENT; not a coverage guarantee"
                    if v < nominal - 0.15 else "")
            out.append(f"  {label} credible coverage    : {100 * v:.1f}% "
                       f"(nominal {100 * nominal:.0f}%){flag}")
    out.append("")
    return out


def summarise(result: dict, *, campaign_meta: dict | None = None,
              anchors_meta: dict | None = None, truth: dict | None = None,
              calibration: dict | None = None) -> str:
    """Render the full text report for one localisation result."""
    est = result["estimate"]
    cert = result.get("certificate", {})
    cred = result.get("credible_regions", {})
    diag = result.get("diagnostics", {})
    dm = result.get("delay_model", {})

    out: list[str] = []
    A = out.append

    A("=" * 78)
    A("LATENCY-ONLY GEOLOCATION REPORT")
    A("=" * 78)

    if anchors_meta:
        A(f"anchor set      : {anchors_meta.get('count', '?'):,} endpoints, "
          f"{anchors_meta.get('countries', '?')} countries, "
          f"{anchors_meta.get('distinct_operators', '?')} networks "
          f"(hash {anchors_meta.get('hash', '?')})")
    if campaign_meta:
        A(f"measurement     : backend {campaign_meta.get('backend', '?')}, "
          f"{campaign_meta.get('samples_per_anchor', '?')} echoes/endpoint, "
          f"{campaign_meta.get('duration_s', '?')} s")
        A(f"response        : {diag.get('responding_anchors', '?')}/"
          f"{diag.get('attempted_anchors', '?')} endpoints replied "
          f"({100 * diag.get('response_rate', 0):.1f}%)")
    A("")

    A("POSITION")
    cc = result.get("certificate_centroid", {})
    pm = result.get("posterior_mean", {})
    cap = result.get("certificate_cap_centre", {})
    sep = result.get("estimate_separation_km")
    if cc:
        A(f"  estimate             : {_fmt_latlon(cc['lat'], cc['lon'])}")
        A(f"      centroid of the feasible region; rests on nothing but the")
        A(f"      speed of light, and the most stable of the available estimates")
    if pm:
        A(f"  model-refined        : {_fmt_latlon(pm['lat'], pm['lon'])}")
        A(f"      posterior mean under the fitted delay model")
    if cap:
        A(f"  region centre        : {_fmt_latlon(cap['lat'], cap['lon'])}")
        A(f"      centre of the enclosing cap; sits on the region boundary and")
        A(f"      moves more between scans, so it is not used as the answer")
    if sep is not None:
        A(f"  the model estimate is {sep:,.0f} km from the reported position")
    if truth:
        d = haversine_km(cc["lat"], cc["lon"], truth["lat"], truth["lon"]) if cc else float("nan")
        d_pm = (haversine_km(pm["lat"], pm["lon"], truth["lat"], truth["lon"])
                if pm else float("nan"))
        A(f"  true location        : {_fmt_latlon(truth['lat'], truth['lon'])}")
        A(f"      [validation only -- never an estimator input]")
        A(f"  error, estimate      : {d:,.1f} km")
        if pm:
            A(f"  error, model-refined : {d_pm:,.1f} km")
    A("")

    A("UNCERTAINTY")
    tol = cert.get("consensus_region")
    strict = cert.get("strict_region")
    A(f"  certificate (speed-of-light cones, no statistical assumptions)")
    if tol:
        A(f"    region         : cap centred {_fmt_latlon(tol['centre_lat'], tol['centre_lon'])}")
        A(f"    radius         : {_fmt_km(tol['radius_km'])}  "
          f"(diameter {_fmt_km(tol['diameter_km'])})")
    A(f"    validity       : holds provided no more than "
      f"{cert.get('min_violations_achievable', '?')} of the "
      f"{cert.get('anchors_used', '?')} anchors are mislocated or anycast "
      f"({100 * cert.get('tolerance_frac', 0):.2f}%)")
    A(f"    propagation    : {cert.get('max_km_per_ms', '?')} km per ms of RTT "
      f"(vacuum light speed; valid for fibre and satellite alike)")
    if strict:
        A(f"    strict variant : radius {_fmt_km(strict['radius_km'])} "
          f"(contradicts no anchor at all)")
    A("")
    for key, label in (("p50", "50% credible region"), ("p90", "90% credible region")):
        r = cred.get(key)
        if not r:
            continue
        A(f"  {label}")
        A(f"    cap radius     : {_fmt_km(r['enclosing_cap_radius_km'])}   "
          f"area {r['area_km2']:,.0f} km\u00b2")
        A(f"    centre         : {_fmt_latlon(r['centre_lat'], r['centre_lon'])}")
    if cred:
        A("    note: these regions come from the fitted delay model and validation")
        A("          shows they are over-confident as coverage statements; the")
        A("          certificate above is the bound you should rely on")
    modes = result.get("modes", [])
    if len(modes) > 1:
        A(f"  multimodality: {len(modes)} distinct modes")
        for m in modes:
            A(f"    {100 * m['posterior_mass']:5.1f}% of mass near "
              f"{_fmt_latlon(m['lat'], m['lon'])}, radius {_fmt_km(m['radius_km'])}")
    A("")
    if calibration:
        out.extend(_calibration_section(calibration))

    A("NETWORK MODEL FITTED FROM THIS MEASUREMENT")
    A(f"  median detour    : {dm.get('implied_median_detour', '?')}x great-circle")
    A(f"  log-detour sd    : {dm.get('sigma_log_detour', '?')}")
    A(f"  minimum delay    : {dm.get('beta_ms', '?')} ms")
    A(f"  outlier weight   : {dm.get('pi_outlier', '?')}")
    A("")

    A("CONSISTENCY CHECKS")
    A(f"  nearest endpoint : {diag.get('nearest_anchor', {}).get('city', '?')} "
      f"({diag.get('nearest_anchor', {}).get('country', '?')}) at "
      f"{diag.get('min_observed_rtt_ms', '?')} ms -> certificate radius "
      f"{_fmt_km(diag.get('nearest_anchor', {}).get('certificate_radius_km'))}")
    A(f"  RTT range        : {diag.get('min_observed_rtt_ms')} / "
      f"{diag.get('median_observed_rtt_ms')} / {diag.get('max_observed_rtt_ms')} ms "
      f"(min/median/max)")
    A(f"  constraints violated at certificate: "
      f"{diag.get('constraints_violated_at_certificate', '?')} of "
      f"{cert.get('anchors_used', '?')} (hard, model-free)")
    A(f"  anchors the delay model struggles to explain: "
      f"{diag.get('model_inconsistent_anchors', 0)} "
      f"({100 * diag.get('model_inconsistent_frac', 0):.2f}%) (soft)")
    susp = diag.get("suspected_mislocated_or_anycast", [])[:6]
    if susp:
        A("    worst explained (RTT implies far less distance than advertised):")
        for s in susp:
            A(f"      {s['key'][:40]:40s} rtt {s['rtt_ms']:>7.2f} ms  "
              f"advertised {s['advertised_km']:>9,.0f} km  "
              f"implies {s['rtt_implied_km']:>9,.0f} km  "
              f"p(informative)={s['responsibility']:.3f}")
    A("")
    return "\n".join(out)


def render_full_map(result: dict, anchor_latlon, truth: dict | None = None) -> str:
    est = result["estimate"]
    cert = result.get("certificate", {}).get("consensus_region")
    p90 = result.get("credible_regions", {}).get("p90")
    modes = [(m["lat"], m["lon"]) for m in result.get("modes", [])]
    return render_map(
        anchor_latlon, est["lat"], est["lon"],
        certificate=cert, p90=p90, modes=modes,
        truth=(truth["lat"], truth["lon"]) if truth else None,
        title="(equirectangular sketch; anchors '.', certificate '#', "
              "90% credible '+')",
    )
