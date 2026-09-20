"""Web UI: a live map of where this machine appears to be.

    latlas web                 # http://0.0.0.0:8080
    latlas web --port 9000

The page opens on a map and starts a scan immediately. Every anchor is streamed
to the browser the moment it answers, so the map fills in as the scan runs rather
than appearing all at once, and a cheap incremental estimate is pushed alongside
so the position visibly converges.

Design notes
------------
* Standard library only on the server: ``http.server`` plus Server-Sent Events.
  No framework, no build step, no dependency for someone who just wants an answer.
* One scan at a time, guarded by a lock. Two concurrent scans would compete for
  the same uplink and corrupt each other's timings, so a second request is
  refused rather than allowed to start a rival run.
* Markers are added incrementally and never cleared; only the estimate, its trail
  and the range beams are redrawn. Rebuilding a thousand markers on every update
  is what makes live maps feel slow.
* Tiles come from OpenStreetMap, which needs no API key. The page degrades to the
  printed numbers if tiles fail to load.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .geo_data import anchors_hash, baked_anchors, nearest_places
from .live import QuickLocator
from .measure import run_campaign
from .sense import DEFAULT_SAMPLES, DEFAULT_TIMEOUT_MS, DEFAULT_WORKERS

#: How many anchors to accumulate before recomputing the preview. Small enough to
#: feel live, large enough that the preview is not noise.
ESTIMATE_EVERY = 40
#: Range beams drawn from the estimate to the closest anchors.
BEAM_COUNT = 48


class Scanner:
    """Runs one measurement at a time and fans events out to every listener."""

    def __init__(self, samples: int, workers: int, quick: bool,
                 timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self.samples = samples
        self.workers = workers
        self.quick = quick
        self.timeout_ms = timeout_ms
        self._lock = threading.Lock()
        self._subs: set[queue.Queue] = set()
        self._sub_lock = threading.Lock()
        self._result: dict | None = None
        self._state = "idle"
        self._started = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- fan-out -------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4096)
        with self._sub_lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            self._subs.discard(q)

    def _emit(self, event: dict) -> None:
        with self._sub_lock:
            targets = list(self._subs)
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass          # a stalled browser must not slow the scan down

    # ---- lifecycle -----------------------------------------------------
    def status(self) -> dict:
        return {"state": self._state, "result": self._result,
                "started": self._started, "samples": self.samples,
                "workers": self.workers, "quick": self.quick,
                "timeout_ms": self.timeout_ms}

    def start(self, force: bool = False) -> dict:
        with self._lock:
            if self._state == "scanning":
                return {"started": False, "state": self._state,
                        "note": "a scan is already running"}
            self._state = "scanning"
            self._started = time.time()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return {"started": True, "state": self._state}

    def stop(self) -> dict:
        self._stop.set()
        return {"stopping": True, "state": self._state}

    def _run(self) -> None:
        t0 = time.time()
        anchors = list(baked_anchors())
        total = len(anchors)
        locator = QuickLocator()
        locator.reset(t0)
        live = [0]
        last_fix = [0.0]
        self._emit({"type": "start", "total": total,
                    "samples": self.samples, "workers": self.workers,
                    "quick": self.quick, "hash": anchors_hash(),
                    "started": t0})

        def on_anchor(m) -> None:
            rtt = m.floor_ms()
            locator.add(m.lat, m.lon, rtt)
            live[0] += 1
            self._emit({
                "type": "anchor", "i": live[0], "total": total,
                "key": m.key, "lat": round(m.lat, 5), "lon": round(m.lon, 5),
                "rtt_ms": None if rtt is None else round(rtt, 2),
                "median_ms": (None if m.median_ms is None else round(m.median_ms, 2)),
                "loss": round(m.loss, 3), "source": m.source,
                "city": m.city, "country": m.country,
            })
            # Recompute the preview on a schedule, not per anchor.
            if live[0] % ESTIMATE_EVERY == 0:
                fix = locator.estimate()
                if fix is not None and time.time() - last_fix[0] >= 0.25:
                    last_fix[0] = time.time()
                    payload = fix.to_dict()
                    # Name the places around the running fix too, so the panel is
                    # useful during the scan rather than only at the end.
                    payload["nearest"] = [
                        {"name": p.name, "country": p.country,
                         "distance_km": round(p.distance_km, 1),
                         "compass": p.compass}
                        for p in nearest_places(fix.lat, fix.lon, k=4)
                    ]
                    self._emit({"type": "estimate", "data": payload,
                                "elapsed_s": round(time.time() - t0, 1)})

        try:
            campaign = run_campaign(
                anchors, samples=self.samples, timeout_ms=self.timeout_ms,
                max_workers=self.workers, anchors_hash=anchors_hash(),
                progress=lambda s: None, on_anchor=on_anchor)

            self._emit({"type": "stage", "stage": "refining",
                        "elapsed_s": round(time.time() - t0, 1)})

            from .estimate import estimate_location
            kwargs = ({"exploration": 15000, "final_points": 60000,
                       "fit_iterations": 2} if self.quick else {})
            est = estimate_location(campaign.measurements,
                                    progress=lambda s: None, **kwargs)
            head = nearest_places(est.cert_lat, est.cert_lon, k=6)
            model = nearest_places(est.lat, est.lon, k=1)
            payload = {
                "position": {"lat": round(est.cert_lat, 5),
                             "lon": round(est.cert_lon, 5)},
                "place": (head[0].name if head else ""),
                "certificate_radius_km": round(est.cert_radius_km, 1),
                "credible_radius_km": ((est.credible.get("p90") or {})
                                       .get("enclosing_cap_radius_km")),
                "region_centre": {"lat": round(est.cap_lat, 5),
                                  "lon": round(est.cap_lon, 5)},
                "model": {"lat": round(est.lat, 5), "lon": round(est.lon, 5),
                          "place": (model[0].name if model else ""),
                          "separation_km": round(est.separation_km(), 1)},
                "nearest": [{"name": p.name, "country": p.country,
                             "lat": round(p.lat, 5), "lon": round(p.lon, 5),
                             "distance_km": round(p.distance_km, 1),
                             "compass": p.compass, "population": p.population}
                            for p in head],
                "anchors_total": total,
                "anchors_used": sum(1 for m in campaign.measurements
                                    if m.received > 0),
                "elapsed_s": round(time.time() - t0, 1),
                "backend": campaign.backend,
                "delay_model": est.params.as_dict(),
                "min_rtt_ms": round(float(min(
                    (m.floor_ms() for m in campaign.measurements
                     if m.floor_ms() is not None), default=0.0)), 2),
                "samples_per_anchor": campaign.samples_per_anchor,
            }
            self._result = payload
            self._state = "done"
            self._emit({"type": "done", "data": payload})
        except Exception as e:  # noqa: BLE001 - report, never kill the server
            self._state = "error"
            self._emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>latlas</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root{
    color-scheme:dark;
    --bg:#0b0e13; --panel:rgba(14,18,24,.93); --line:#222b36; --line2:#2f3a47;
    --fg:#e6edf3; --dim:#8b96a5; --dimmer:#5d6773;
    --accent:#2f81f7; --good:#3fb950; --warn:#d29922; --bad:#f85149; --violet:#a371f7;
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);
    font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    -webkit-font-smoothing:antialiased}
  #map{position:absolute;inset:0;background:var(--bg)}
  .leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.86) contrast(.92) saturate(.55)}
  .leaflet-container{background:var(--bg)}
  .panel{position:absolute;z-index:1000;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;backdrop-filter:blur(6px);box-shadow:0 10px 30px rgba(0,0,0,.5)}
  #left{top:12px;left:12px;width:318px;max-height:calc(100% - 24px);overflow:auto}
  #right{top:12px;right:12px;width:238px;max-height:calc(100% - 24px);overflow:auto}
  #bottom{left:12px;right:12px;bottom:12px;height:74px;display:flex;align-items:center;
    gap:14px;padding:8px 12px}
  .sect{padding:11px 13px;border-bottom:1px solid var(--line)}
  .sect:last-child{border-bottom:0}
  h1{margin:0;font-size:13px;letter-spacing:.6px;font-weight:600}
  .sub{color:var(--dimmer);font-size:10.5px;letter-spacing:.3px;text-transform:uppercase}
  .row{display:flex;justify-content:space-between;gap:8px;padding:2px 0}
  .row .k{color:var(--dim)} .row .v{text-align:right;font-variant-numeric:tabular-nums}
  .big{font-size:23px;font-weight:600;letter-spacing:.4px;font-variant-numeric:tabular-nums}
  .place{color:var(--good);font-size:12.5px;margin-top:2px;min-height:17px}
  .pill{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;padding:2px 7px;
    border-radius:99px;border:1px solid var(--line2);color:var(--dim)}
  .dot{width:6px;height:6px;border-radius:50%;background:var(--dim)}
  .dot.live{background:var(--accent);animation:pulse 1.1s infinite}
  .dot.ok{background:var(--good)} .dot.err{background:var(--bad)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  #bar{height:3px;background:#1b232d;border-radius:2px;overflow:hidden;margin:9px 0 7px}
  #barfill{height:100%;width:0;background:linear-gradient(90deg,#2f81f7,#3fb950);
    transition:width .25s}
  .spark{display:block;width:100%;height:44px}
  ul{margin:6px 0 0;padding:0;list-style:none}
  li{display:flex;justify-content:space-between;gap:8px;padding:2px 0;
    border-top:1px solid var(--line);font-size:12px}
  li:first-child{border-top:0}
  li .n{color:var(--fg)} li .d{color:var(--dim);font-variant-numeric:tabular-nums}
  .ctl{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:4px 0}
  .ctl label{color:var(--dim);font-size:12px}
  input[type=range]{width:104px;accent-color:var(--accent)}
  select{background:#141a22;color:var(--fg);border:1px solid var(--line2);
    border-radius:5px;padding:3px 6px;font:inherit;font-size:11.5px}
  input[type=number]{width:62px;background:#141a22;color:var(--fg);border:1px solid var(--line2);
    border-radius:5px;padding:3px 6px;font:inherit;font-size:11.5px}
  .btn{display:block;width:100%;margin-top:8px;padding:7px;border:0;border-radius:6px;
    background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
  .btn.sec{background:#1b232d;color:var(--fg);border:1px solid var(--line2);font-weight:400}
  .btn:disabled{opacity:.4;cursor:default}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--dim);align-items:center}
  .sw{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px;
    vertical-align:middle}
  .hist{flex:1;height:56px;min-width:120px}
  .muted{color:var(--dimmer);font-size:11px}
  #err{color:var(--bad)}
  .kv{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);padding:1px 0}
</style></head><body>
<div id="map"></div>

<section id="left" class="panel">
  <div class="sect">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h1>LATLAS</h1>
      <span class="pill"><span id="dot" class="dot"></span><span id="status">idle</span></span>
    </div>
    <div class="sub" style="margin-top:3px">position from round-trip time only</div>
  </div>

  <div class="sect">
    <div class="big" id="coord">&mdash;</div>
    <div class="place" id="place">starting scan&hellip;</div>
    <div id="bar"><div id="barfill"></div></div>
    <div class="row"><span class="k">bound</span><span class="v" id="bound">&mdash;</span></div>
    <div class="row"><span class="k">anchors</span><span class="v" id="anch">&mdash;</span></div>
    <div class="row"><span class="k">closest RTT</span><span class="v" id="minrtt">&mdash;</span></div>
    <div class="row"><span class="k">elapsed</span><span class="v" id="elapsed">&mdash;</span></div>
    <div class="row"><span class="k">rate</span><span class="v" id="rate">&mdash;</span></div>
  </div>

  <div class="sect">
    <div class="sub">uncertainty while scanning</div>
    <canvas id="spark" class="spark"></canvas>
    <div class="muted" id="sparklbl">bound shrinks as closer anchors answer</div>
  </div>

  <div class="sect" id="finalbox" style="display:none">
    <div class="sub">refined result</div>
    <div class="kv"><span>model estimate</span><span id="fmodel">&mdash;</span></div>
    <div class="kv"><span>90% region</span><span id="fcred">&mdash;</span></div>
    <div class="kv"><span>backend</span><span id="fbackend">&mdash;</span></div>
  </div>

  <div class="sect">
    <div class="sub">nearest places</div>
    <ul id="cities"><li><span class="n muted">waiting for a fix</span></li></ul>
  </div>
</section>

<section id="right" class="panel">
  <div class="sect">
    <div class="sub">scan options</div>
    <div class="ctl"><label>echoes / anchor</label>
      <input type="number" id="oSamples" min="1" max="40" value="10"></div>
    <div class="ctl"><label>parallel probes</label>
      <input type="number" id="oWorkers" min="4" max="256" value="64"></div>
    <div class="ctl"><label>fast grid</label>
      <input type="checkbox" id="oQuick" checked></div>
    <button class="btn" id="bScan">scan</button>
    <button class="btn sec" id="bStop">stop</button>
  </div>
  <div class="sect">
    <div class="sub">display</div>
    <div class="ctl"><label>basemap</label>
      <select id="oBase">
        <option value="dark">dark</option>
        <option value="light">light</option>
        <option value="none">none</option>
      </select></div>
    <div class="ctl"><label>anchor dots</label>
      <input type="checkbox" id="oDots" checked></div>
    <div class="ctl"><label>range beams</label>
      <input type="checkbox" id="oBeams" checked></div>
    <div class="ctl"><label>trail</label>
      <input type="checkbox" id="oTrail" checked></div>
    <div class="ctl"><label>follow estimate</label>
      <input type="checkbox" id="oFollow"></div>
    <div class="ctl"><label>max RTT shown</label>
      <input type="range" id="oMaxRtt" min="20" max="400" step="10" value="400"></div>
    <div class="muted" id="maxrttlbl">400 ms</div>
  </div>
  <div class="sect">
    <div class="sub">legend</div>
    <div class="legend">
      <span><i class="sw" style="background:#3fb950"></i>&lt;40 ms</span>
      <span><i class="sw" style="background:#2f81f7"></i>&lt;90</span>
      <span><i class="sw" style="background:#d29922"></i>&lt;180</span>
      <span><i class="sw" style="background:#f85149"></i>&ge;180</span>
      <span><i class="sw" style="background:#5d6773"></i>lost</span>
    </div>
    <div class="legend" style="margin-top:8px">
      <span><i class="sw" style="background:#a371f7"></i>estimate</span>
      <span><i class="sw" style="background:#2f81f7"></i>bound</span>
      <span><i class="sw" style="background:#3fb950"></i>90% region</span>
    </div>
  </div>
</section>

<section id="bottom" class="panel">
  <div style="min-width:150px">
    <div class="sub">RTT distribution</div>
    <div class="muted" id="histlbl">no data yet</div>
  </div>
  <canvas id="hist" class="hist"></canvas>
  <div style="min-width:150px;text-align:right">
    <div class="sub">reply rate</div>
    <div class="big" id="replyrate" style="font-size:17px">&mdash;</div>
  </div>
</section>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const BASEMAPS = {
  dark:  {url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png', cls:'dark'},
  light: {url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png', cls:'light'},
};
const map = L.map('map',{worldCopyJump:true,preferCanvas:true,zoomControl:true})
              .setView([30,10],3);
const canvas = L.canvas({padding:.4});
let tileLayer = null, currentBase = 'dark';

function setBase(name){
  if(tileLayer){ map.removeLayer(tileLayer); tileLayer=null; }
  document.body.classList.toggle('lightmap', name==='light');
  if(name==='none') return;
  tileLayer = L.tileLayer(BASEMAPS[name].url,{maxZoom:19,
    attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
}
setBase('dark');

const anchorLayer = L.layerGroup().addTo(map);
const beamLayer   = L.layerGroup().addTo(map);
const fixLayer    = L.layerGroup().addTo(map);
const trailLayer  = L.layerGroup().addTo(map);

let markers = new Map();      // key -> {marker, rtt}
let ripples = [];
let trailPts = [], fitted = false, scanning = false, userPanned = false;
let t0 = 0, lastCount = 0, lastT = 0, joinedMidScan = false;
let cur = null, prevFix = null;

const $ = id => document.getElementById(id);
const fmt = (n,d=0) => (n==null||isNaN(n)) ? '\u2014'
  : Number(n).toLocaleString(undefined,{maximumFractionDigits:d});

function rttColor(rtt, ok){
  if(!ok || rtt==null) return '#5d6773';
  if(rtt < 40) return '#3fb950';
  if(rtt < 90) return '#2f81f7';
  if(rtt < 180) return '#d29922';
  return '#f85149';
}
function rttRadius(rtt, ok){
  if(!ok || rtt==null) return 2.5;
  return Math.max(3, Math.min(11, 2.6 + Math.log2(rtt+1)*1.15));
}

function addAnchor(e){
  const ok = e.rtt_ms != null;
  const m = L.circleMarker([e.lat,e.lon],{
    renderer:canvas, radius:rttRadius(e.rtt_ms,ok),
    color:rttColor(e.rtt_ms,ok), fillColor:rttColor(e.rtt_ms,ok),
    weight:ok?1:0.6, opacity:ok?.95:.5, fillOpacity:ok?.8:.25
  });
  m.bindTooltip(
    `<b>${e.key}</b><br>${ok?fmt(e.rtt_ms,2)+' ms':'no reply'}`
    + (e.city?`<br>${e.city}, ${e.country}`:'')
    + (e.loss?`<br>loss ${(e.loss*100).toFixed(0)}%`:''),
    {direction:'top',offset:[0,-4]});
  anchorLayer.addLayer(m);
  markers.set(e.key,{marker:m,rtt:e.rtt_ms});
  // Brief expanding ring so arrivals read as pings rather than a static
  // scatter. Capped in count and lifetime so a 1600-anchor scan stays smooth.
  if(ok && ripples.length < 26){
    const r = L.circleMarker([e.lat,e.lon],{renderer:canvas,radius:3,
      color:rttColor(e.rtt_ms,ok),weight:1.4,fill:false,opacity:.9}).addTo(anchorLayer);
    ripples.push({layer:r, born:Date.now(), lat:e.lat, lon:e.lon,
                  col:rttColor(e.rtt_ms,ok)});
  }
}

function stepRipples(){
  const now = Date.now();
  ripples = ripples.filter(r=>{
    const age = (now-r.born)/700;
    if(age >= 1){ anchorLayer.removeLayer(r.layer); return false; }
    r.layer.setRadius(3 + age*13);
    r.layer.setStyle({opacity:.9*(1-age)});
    return true;
  });
}
setInterval(stepRipples, 90);

function drawFix(fix, final){
  fixLayer.clearLayers();
  beamLayer.clearLayers();
  const c = [fix.lat, fix.lon];
  if(fix.cap_radius_km){
    L.circle(c,{radius:fix.cap_radius_km*1000,color:'#2f81f7',weight:1.4,
      opacity:.75,fillOpacity:.04,dashArray:'5,7'}).addTo(fixLayer);
  }
  if(final && final.credible_radius_km){
    L.circle(c,{radius:final.credible_radius_km*1000,color:'#3fb950',weight:1.4,
      opacity:.8,fillOpacity:.10}).addTo(fixLayer);
  }
  // Range beams: lines to the closest anchors, which are the ones that carry the
  // information. Redrawn each fix so they visibly sweep as the position settles.
  if($('oBeams').checked){
    const near = [...markers.values()].filter(x=>x.rtt!=null)
                   .sort((a,b)=>a.rtt-b.rtt).slice(0,48);
    for(const {marker} of near){
      const ll = marker.getLatLng();
      L.polyline([c,[ll.lat,ll.lng]],{color:'#a371f7',weight:.7,opacity:.22,
        renderer:canvas}).addTo(beamLayer);
    }
  }
  L.circleMarker(c,{renderer:canvas,radius:7,color:'#fff',weight:2,
    fillColor:'#a371f7',fillOpacity:1}).addTo(fixLayer)
    .bindTooltip(`estimate<br>${fmt(fix.lat,5)}, ${fmt(fix.lon,5)}`
      +(fix.cap_radius_km?`<br>bound \u00b1${fmt(fix.cap_radius_km)} km`:''),
      {direction:'top',offset:[0,-6]});
  if(final && final.model){
    L.circleMarker([final.model.lat,final.model.lon],{renderer:canvas,radius:5,
      color:'#d29922',weight:2,fillOpacity:.85}).addTo(fixLayer)
      .bindTooltip('model-refined estimate');
  }
  if($('oTrail').checked && prevFix){
    const moved = map.distance([prevFix.lat,prevFix.lon],c)/1000;
    if(moved > 1){
      L.polyline([[prevFix.lat,prevFix.lon],c],{color:'#a371f7',weight:1.6,
        opacity:.5,dashArray:'3,4',renderer:canvas}).addTo(trailLayer);
    }
  }
  prevFix = {lat:fix.lat, lon:fix.lon};
}

function sparkline(){
  const cv = $('spark'); if(!cv) return;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*devicePixelRatio; cv.height = h*devicePixelRatio;
  const g = cv.getContext('2d'); g.scale(devicePixelRatio,devicePixelRatio);
  g.clearRect(0,0,w,h);
  if(trailPts.length < 2){ $('sparklbl').textContent='waiting for the first fix'; return; }
  $('sparklbl').textContent = 'region extent, '+trailPts.length+' updates';
  const vals = trailPts.map(p=>p.r);
  const mx = Math.max(...vals), mn = Math.min(...vals);
  const span = Math.max(mx-mn, 1);
  g.strokeStyle='#2f81f7'; g.lineWidth=1.6; g.beginPath();
  trailPts.forEach((p,i)=>{
    const x = i/(trailPts.length-1)*(w-2)+1;
    const y = h-3 - (p.r-mn)/span*(h-8);
    i?g.lineTo(x,y):g.moveTo(x,y);
  });
  g.stroke();
  g.fillStyle='#8b96a5'; g.font='10px ui-monospace,monospace';
  g.fillText(fmt(mx)+' km', 2, 10);
  g.fillText(fmt(mn)+' km', 2, h-2);
}

const HIST_BINS = 40;
let hist = new Array(HIST_BINS).fill(0);
function histogram(){
  const cv = $('hist'); if(!cv) return;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*devicePixelRatio; cv.height = h*devicePixelRatio;
  const g = cv.getContext('2d'); g.scale(devicePixelRatio,devicePixelRatio);
  g.clearRect(0,0,w,h);
  const mx = Math.max(...hist, 1);
  const bw = w/HIST_BINS;
  for(let i=0;i<HIST_BINS;i++){
    const bh = hist[i]/mx*(h-12);
    g.fillStyle = i<HIST_BINS*0.15?'#3fb950':i<HIST_BINS*0.45?'#2f81f7':
                  i<HIST_BINS*0.75?'#d29922':'#f85149';
    g.fillRect(i*bw+0.5, h-bh-1, bw-1, bh);
  }
  g.fillStyle='#5d6773'; g.font='10px ui-monospace,monospace';
  g.fillText('0', 2, h-2); g.fillText('400+ ms', w-52, h-2);
  const n = hist.reduce((a,b)=>a+b,0);
  $('histlbl').textContent = n ? n+' replies' : 'no data yet';
}
histogram();

function elapsedS(){ return t0 ? (Date.now()-t0)/1000 : 0; }

function reset(){
  anchorLayer.clearLayers(); beamLayer.clearLayers(); fixLayer.clearLayers();
  trailLayer.clearLayers();
  markers.clear(); trailPts=[]; hist=new Array(HIST_BINS).fill(0); ripples=[];
  fitted=false; prevFix=null; lastCount=0; lastT=0;
  histogram(); sparkline();
  $('cities').innerHTML='<li><span class="n muted">waiting for a fix</span></li>';
  $('finalbox').style.display='none';
}

function onAnchor(e){
  addAnchor(e);
  const rtt = e.rtt_ms;
  if(rtt!=null){
    const b = Math.min(HIST_BINS-1, Math.floor(rtt/400*HIST_BINS));
    hist[b]++; histogram();
  }
  const rate = lastT ? (e.i-lastCount)/((Date.now()-lastT)/1000) : 0;
  if(Date.now()-lastT > 700){ lastCount=e.i; lastT=Date.now(); }
  $('anch').textContent = e.i+' / '+e.total;
  $('barfill').style.width = (100*e.i/e.total)+'%';
  $('elapsed').textContent = fmt(elapsedS(),0)+' s';
  if(rate>0) $('rate').textContent = fmt(rate,0)+' /s';
  const ok = [...markers.values()].filter(m=>m.rtt!=null).length;
  $('replyrate').textContent = (100*ok/markers.size).toFixed(1)+'%';
}

function onFix(fix){
  cur = fix;
  trailPts.push({lat:fix.lat, lon:fix.lon,
                 r:(fix.region_radius_km||fix.cap_radius_km)});
  if(trailPts.length>400) trailPts.shift();
  sparkline();
  drawFix(fix, null);
  $('coord').textContent =
    fmt(Math.abs(fix.lat),4)+'\u00b0'+(fix.lat>=0?'N':'S')+', '+
    fmt(Math.abs(fix.lon),4)+'\u00b0'+(fix.lon>=0?'E':'W');
  $('bound').textContent = '\u00b1'+fmt(fix.cap_radius_km)+' km';
  $('minrtt').textContent = fmt(fix.min_rtt_ms,2)+' ms';
  $('place').textContent = 'converging\u2026';
  if(fix.nearest && fix.nearest.length){
    $('cities').innerHTML = fix.nearest.map(c=>
      `<li><span class="n">${c.name}, ${c.country}</span>`
      +`<span class="d">${fmt(c.distance_km)} km ${c.compass}</span></li>`).join('');
  }
  if($('oFollow').checked && !userPanned){
    map.setView([fix.lat,fix.lon], Math.max(map.getZoom(), 3), {animate:true});
  }
}

function onDone(d){
  scanning=false;
  $('dot').className='dot ok'; $('status').textContent='complete';
  $('bScan').disabled=false; $('bStop').disabled=true;
  $('coord').textContent =
    fmt(Math.abs(d.position.lat),4)+'\u00b0'+(d.position.lat>=0?'N':'S')+', '+
    fmt(Math.abs(d.position.lon),4)+'\u00b0'+(d.position.lon>=0?'E':'W');
  $('place').textContent = 'at ' + d.place;
  const hard = d.uncertainty ? d.uncertainty.is_hard_bound !== false : true;
  $('bound').textContent = (hard ? '\u00b1' : '~') + fmt(d.certificate_radius_km)
      + ' km' + (hard ? '' : '  (not a bound)');
  if(!hard){
    $('place').textContent += ' \u2014 anchor set self-contradictory, treat as unreliable';
    $('place').style.color = '#d29922';
  }
  $('minrtt').textContent = fmt(d.min_rtt_ms,2)+' ms';
  $('anch').textContent = d.anchors_used+' / '+d.anchors_total;
  $('fmodel').textContent = fmt(d.model.lat,3)+', '+fmt(d.model.lon,3)
      +'  ('+fmt(d.model.separation_km)+' km)';
  $('fcred').textContent = d.credible_radius_km ? fmt(d.credible_radius_km)+' km' : '\u2014';
  $('fbackend').textContent = d.backend;
  $('finalbox').style.display='block';
  $('cities').innerHTML = d.nearest.map(c=>
    `<li><span class="n">${c.name}, ${c.country}</span>`
    +`<span class="d">${fmt(c.distance_km)} km ${c.compass}</span></li>`).join('');
  drawFix({lat:d.position.lat,lon:d.position.lon,cap_radius_km:d.certificate_radius_km},
          d);
  map.fitBounds(L.latLng(d.position.lat,d.position.lon)
    .toBounds(Math.max(d.certificate_radius_km*1000*2.6, 8e5)));
}

map.on('dragstart', ()=>{ userPanned = true; });
map.on('zoomstart', e=>{ if(e.originalEvent) userPanned = true; });

function connect(){
  const es = new EventSource('/events');
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if(m.type==='start'){
      reset(); scanning=true; t0 = m.started ? m.started*1000 : Date.now();
      $('dot').className='dot live'; $('status').textContent='scanning';
      $('bScan').disabled=true; $('bStop').disabled=false;
      $('place').textContent='probing '+m.total+' anchors\u2026';
    }
    else if(m.type==='anchor') onAnchor(m);
    else if(m.type==='estimate') onFix(m.data);
    else if(m.type==='stage'){
      $('status').textContent=m.stage; $('place').textContent='refining position\u2026';
      $('barfill').style.width='100%';
    }
    else if(m.type==='done') onDone(m.data);
    else if(m.type==='error'){
      scanning=false; $('dot').className='dot err'; $('status').textContent='error';
      $('place').innerHTML='<span id="err">'+m.message+'</span>';
      $('bScan').disabled=false; $('bStop').disabled=true;
    }
    else if(m.type==='hello' && m.state==='scanning'){
      // Joined a scan already in flight: adopt the server's clock so the
      // elapsed readout is real rather than seconds since 1970.
      scanning=true; joinedMidScan=true;
      // `started` is the server's Unix time in seconds, so it converts directly
      // to the same epoch-millisecond scale Date.now() uses.
      if(m.started) t0 = m.started*1000;
      $('dot').className='dot live'; $('status').textContent='scanning';
      $('bScan').disabled=true; $('bStop').disabled=false;
    }
  };
}

async function start(){
  const p = new URLSearchParams({
    samples:$('oSamples').value, workers:$('oWorkers').value,
    quick:$('oQuick').checked?'1':'0'});
  const r = await fetch('/scan?'+p,{method:'POST'});
  const j = await r.json();
  if(!j.started) $('status').textContent = j.note || 'busy';
}
$('bScan').onclick = start;
$('bStop').onclick = ()=>fetch('/stop',{method:'POST'});

$('oBase').onchange = e => setBase(e.target.value);
$('oMaxRtt').oninput = e => {
  $('maxrttlbl').textContent = e.target.value+' ms';
  const lim = +e.target.value;
  for(const {marker,rtt} of markers.values())
    marker.setStyle({opacity: (rtt==null||rtt<=lim)?1:0.06,
                     fillOpacity: (rtt==null||rtt<=lim)?0.8:0.04});
};
$('oDots').onchange = e => {
  if(e.target.checked) map.addLayer(anchorLayer); else map.removeLayer(anchorLayer);
};
$('oBeams').onchange = e => {
  if(e.target.checked) map.addLayer(beamLayer); else map.removeLayer(beamLayer);
};
$('oTrail').onchange = e => {
  if(e.target.checked) map.addLayer(trailLayer); else map.removeLayer(trailLayer);
};
window.addEventListener('resize', ()=>{ sparkline(); histogram(); });
connect();
start();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    scanner: Scanner
    server_version = "latlas"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass  # a scan logs hundreds of requests otherwise

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, json.dumps(self.scanner.status()).encode(),
                       "application/json")
        elif path == "/events":
            self._stream()
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        params = {}
        for part in query.split("&"):
            k, _, v = part.partition("=")
            if k:
                params[k] = v
        if path == "/scan":
            for key, attr, cast in (("samples", "samples", int),
                                    ("workers", "workers", int),
                                    ("quick", "quick", lambda s: s == "1")):
                if key in params:
                    try:
                        setattr(self.scanner, attr, cast(params[key]))
                    except (TypeError, ValueError):
                        pass
            out = self.scanner.start()
            self._send(200, json.dumps(out).encode(), "application/json")
        elif path == "/stop":
            self._send(200, json.dumps(self.scanner.stop()).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = self.scanner.subscribe()
        try:
            st = self.scanner.status()
            self._emit(self._sse({"type": "hello", "state": st["state"],
                                  "started": st["started"],
                                  "samples": st["samples"],
                                  "workers": st["workers"],
                                  "quick": st["quick"]}))
            if st["state"] == "done" and st["result"]:
                self._emit(self._sse({"type": "done", "data": st["result"]}))
            while True:
                try:
                    ev = q.get(timeout=15.0)
                except queue.Empty:
                    self._emit(": keepalive\n\n")   # defeat idle proxy timeouts
                    continue
                self._emit(self._sse(ev))
                if ev.get("type") in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError,
                OSError):
            pass
        finally:
            self.scanner.unsubscribe(q)

    @staticmethod
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def _emit(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    """HTTP server that refuses to share its port.

    ``HTTPServer`` sets ``allow_reuse_address``, which on Windows maps to
    ``SO_REUSEADDR`` and lets a *second* process bind a port that is already
    listening. A stale instance then silently shadows the new one and keeps
    serving its old page, which is exactly what happened during development and
    cost a confusing debugging round. Failing loudly is the correct behaviour.
    """

    allow_reuse_address = False
    daemon_threads = True


def serve(host: str = "0.0.0.0", port: int = 8080, *,
          samples: int = DEFAULT_SAMPLES, workers: int = DEFAULT_WORKERS,
          quick: bool = False, autostart: bool = True,
          progress=print) -> None:
    """Serve the live map until interrupted."""
    scanner = Scanner(samples=samples, workers=workers, quick=quick)
    handler = type("BoundHandler", (Handler,), {"scanner": scanner})
    try:
        httpd = _Server((host, port), handler)
    except OSError as e:
        raise SystemExit(
            f"cannot bind {host}:{port}: {e}\n"
            f"another latlas web instance is probably still running "
            f"(try a different --port, or stop it)") from e
    shown = "0.0.0.0" if host in ("", "0.0.0.0") else host
    progress(f"latlas web UI on http://{shown}:{port}/  "
             f"(reachable at http://<this-machine-ip>:{port}/ from other devices)")
    progress(f"  streaming per-anchor results; {samples} echoes/anchor, "
             f"{workers} parallel workers")
    if autostart:
        scanner.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        progress("\nstopping")
    finally:
        httpd.server_close()
