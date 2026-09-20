"""Web UI: a live map of where this machine appears to be.

    python -m latlas web                 # http://0.0.0.0:8080
    python -m latlas web --port 9000

The page opens on a map and starts a scan immediately. Progress is streamed over
Server-Sent Events as anchors reply, so the readout moves while the measurement
runs rather than sitting on a spinner, and the finished scan drops a marker, the
speed-of-light certificate circle, and the neighbouring settlements onto the map.

Design notes
------------
* ``http.server`` from the standard library only -- no framework, no build step,
  no extra dependency for someone who just wants an answer.
* One scan at a time, guarded by a lock. Two concurrent scans would compete for
  the same narrow uplink and corrupt each other's timings, so a second request
  waits for the result already in flight instead of starting a rival run.
* The page is a single self-contained document; Leaflet and its tiles come from a
  CDN because this tool only works on a machine that has a network, and the map
  degrades to the printed coordinates if tiles fail to load.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .sense import DEFAULT_SAMPLES, DEFAULT_WORKERS, SenseResult, sense

# --------------------------------------------------------------------------
# Scan coordination
# --------------------------------------------------------------------------


class Scanner:
    """Runs one measurement at a time and fans the events out to all listeners."""

    def __init__(self, samples: int, workers: int, quick: bool) -> None:
        self.samples = samples
        self.workers = workers
        self.quick = quick
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()
        self._sub_lock = threading.Lock()
        self._result: dict | None = None
        self._state = "idle"
        self._started = 0.0
        self._thread: threading.Thread | None = None

    # ---- fan-out -------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._sub_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            self._subscribers.discard(q)

    def _emit(self, event: dict) -> None:
        with self._sub_lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    # ---- lifecycle -----------------------------------------------------
    def status(self) -> dict:
        return {"state": self._state, "result": self._result,
                "started": self._started,
                "samples": self.samples, "workers": self.workers}

    def start(self, force: bool = False) -> dict:
        """Begin a scan unless one is already running."""
        with self._lock:
            if self._state == "scanning" and not force:
                return {"started": False, "state": self._state}
            if self._state == "scanning" and force:
                return {"started": False, "state": self._state,
                        "note": "a scan is already running"}
            self._state = "scanning"
            self._started = time.time()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return {"started": True, "state": self._state}

    def _run(self) -> None:
        self._emit({"type": "stage", "stage": "probing"})

        def on_progress(ev: dict) -> None:
            self._emit(ev)

        try:
            res = sense(samples=self.samples, workers=self.workers,
                        quick=self.quick, on_progress=on_progress)
            self._result = res.to_dict()
            self._state = "done"
            self._emit({"type": "done", "data": self._result})
        except Exception as e:  # noqa: BLE001 - report, never kill the server
            self._state = "error"
            self._emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>latlas &mdash; where am I?</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0e1116; color:#e6edf3;
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  #map { position:absolute; inset:0; background:#0e1116; }
  /* OpenStreetMap's raster tiles need no API key, which CARTO's now do. They are
     light-themed, so the tile pane is inverted to keep the dark UI coherent. */
  .leaflet-tile-pane { filter: invert(1) hue-rotate(180deg) brightness(.92) contrast(.92) saturate(.7); }
  #panel { position:absolute; top:14px; left:14px; z-index:1000; width:330px;
           max-height:calc(100% - 28px); overflow:auto;
           background:rgba(16,20,27,.93); border:1px solid #2b3441;
           border-radius:10px; padding:14px 16px;
           box-shadow:0 8px 28px rgba(0,0,0,.55); }
  h1 { font-size:15px; margin:0 0 2px; letter-spacing:.4px; }
  .sub { color:#7d8896; font-size:11px; margin-bottom:12px; }
  .coord { font-size:21px; font-weight:600; letter-spacing:.3px; }
  .place { color:#7ee787; font-size:13px; margin:4px 0 12px; word-break:break-word; }
  .row { display:flex; justify-content:space-between; gap:10px;
         padding:3px 0; border-top:1px solid #1e2632; }
  .row span:first-child { color:#8b96a5; }
  .row span:last-child { text-align:right; }
  .note { color:#7d8896; font-size:11px; margin-top:10px; }
  .warn { color:#e3b341; font-size:11px; margin-top:6px; }
  ul { margin:6px 0 0; padding-left:16px; }
  li { margin:2px 0; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         margin-right:6px; vertical-align:middle; }
  button { background:#1f6feb; color:#fff; border:0; border-radius:6px;
           padding:7px 12px; font:inherit; cursor:pointer; margin-top:12px; }
  button:disabled { background:#28313d; color:#68727f; cursor:default; }
  #bar { height:4px; background:#1e2632; border-radius:2px; overflow:hidden;
         margin:10px 0 2px; }
  #barfill { height:100%; width:0%; background:#1f6feb; transition:width .3s; }
</style></head><body>
<div id="map"></div>
<div id="panel">
  <h1>latlas</h1>
  <div class="sub">position from ICMP round-trip times only</div>
  <div id="coord" class="coord">&mdash;</div>
  <div id="place" class="place">starting scan&hellip;</div>
  <div id="bar"><div id="barfill"></div></div>
  <div id="status" class="note">idle</div>
  <div id="detail" style="display:none">
    <div class="row"><span>uncertainty</span><span id="unc">&mdash;</span></div>
    <div class="row"><span>anchors replied</span><span id="anch">&mdash;</span></div>
    <div class="row"><span>closest RTT</span><span id="rtt">&mdash;</span></div>
    <div class="row"><span>scan time</span><span id="dur">&mdash;</span></div>
    <div class="row"><span>model estimate</span><span id="model">&mdash;</span></div>
    <div class="note" style="margin-top:10px">Nearest places to the estimate:</div>
    <ul id="cities"></ul>
    <div id="cal" class="warn"></div>
  </div>
  <button id="again">scan again</button>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map', {worldCopyJump:true}).setView([25,10], 2);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, crossOrigin: true,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
let marker=null, certCircle=null, credCircle=null, cityLayer=null, modelMarker=null;

function clear(){ [marker,modelMarker,certCircle,credCircle].forEach(o=>{if(o)map.removeLayer(o);});
  if(cityLayer){map.removeLayer(cityLayer);cityLayer=null;}
  marker=modelMarker=certCircle=credCircle=null; }

function fmt(n,d=0){ return n==null?'n/a':Number(n).toLocaleString(undefined,{maximumFractionDigits:d}); }

function show(d){
  clear();
  const p = d.position;
  const cert = d.uncertainty.certificate_radius_km;
  const cred = d.uncertainty.credible_radius_km;
  if (cert) certCircle = L.circle([p.lat,p.lon], {radius:cert*1000, color:'#58a6ff',
      weight:1.5, fillOpacity:0.05, dashArray:'5,6'}).addTo(map);
  if (cred) credCircle = L.circle([p.lat,p.lon], {radius:cred*1000, color:'#7ee787',
      weight:1.5, fillOpacity:0.10}).addTo(map);
  marker = L.circleMarker([p.lat,p.lon], {radius:8, color:'#fff', weight:2,
      fillColor:'#1f6feb', fillOpacity:1}).addTo(map);
  marker.bindPopup('<b>estimate</b><br>'+fmt(p.lat,4)+', '+fmt(p.lon,4)).openPopup();
  modelMarker = L.circleMarker([d.model_refined.lat,d.model_refined.lon],
      {radius:5, color:'#e3b341', weight:2, fillOpacity:.8}).addTo(map);
  modelMarker.bindPopup('model-refined estimate');
  const items = (d.nearest||[]).map(c =>
      L.circleMarker([c.lat,c.lon],{radius:3,color:'#8b96a5',weight:1,fillOpacity:.7})
       .bindPopup('<b>'+c.name+'</b><br>'+fmt(c.distance_km,0)+' km '+c.compass));
  cityLayer = L.layerGroup(items).addTo(map);

  document.getElementById('coord').textContent =
      fmt(Math.abs(p.lat),4)+'\u00b0'+(p.lat>=0?'N':'S')+', '+
      fmt(Math.abs(p.lon),4)+'\u00b0'+(p.lon>=0?'E':'W');
  document.getElementById('place').textContent = d.place;
  document.getElementById('unc').textContent =
      '\u00b1'+fmt(cert)+' km (speed of light)' +
      (cred ? '  |  '+fmt(cred)+' km (model)' : '');
  document.getElementById('anch').textContent =
      d.measurement.anchors_used+' / '+d.measurement.anchors_total+
      '  ('+(100*d.measurement.response_rate).toFixed(1)+'%)';
  document.getElementById('rtt').textContent = fmt(d.measurement.min_rtt_ms,2)+' ms';
  document.getElementById('dur').textContent = fmt(d.measurement.elapsed_s,1)+' s';
  document.getElementById('model').textContent =
      fmt(d.model_refined.lat,3)+', '+fmt(d.model_refined.lon,3)+
      '  ('+fmt(d.model_refined.separation_km)+' km away)';
  document.getElementById('cities').innerHTML = (d.nearest||[]).map(c =>
      '<li>'+c.name+', '+c.country+' &mdash; '+fmt(c.distance_km)+' km '+c.compass+'</li>')
      .join('');
  document.getElementById('detail').style.display='block';
  map.fitBounds(L.latLng(p.lat,p.lon).toBounds(Math.max(cert*1000*2.6, 400000)));
}

function connect(){
  const es = new EventSource('/events');
  es.onmessage = (e)=>{
    const m = JSON.parse(e.data);
    if (m.type==='progress'){
      const pct = m.total? Math.round(100*m.probed/m.total):0;
      document.getElementById('barfill').style.width = pct+'%';
      document.getElementById('status').textContent =
          'probing '+m.probed+' / '+m.total+' anchors ('+m.elapsed_s+' s)';
    } else if (m.type==='stage'){
      document.getElementById('status').textContent = m.stage+'...';
      if (m.stage==='estimating') document.getElementById('barfill').style.width='100%';
    } else if (m.type==='done'){
      document.getElementById('status').textContent = 'scan complete';
      document.getElementById('again').disabled = false;
      show(m.data);
    } else if (m.type==='error'){
      document.getElementById('status').textContent = 'error: '+m.message;
      document.getElementById('again').disabled = false;
    }
  };
  es.onerror = ()=>{ /* browser retries automatically */ };
}

document.getElementById('again').onclick = async ()=>{
  document.getElementById('again').disabled = true;
  document.getElementById('detail').style.display='none';
  document.getElementById('place').textContent='scanning\u2026';
  document.getElementById('barfill').style.width='0%';
  clear();
  await fetch('/scan', {method:'POST'});
};
connect();
fetch('/scan', {method:'POST'}).then(r=>r.json()).then(j=>{
  if(!j.started) document.getElementById('again').disabled=false;
});
</script></body></html>
"""

_CALIBRATION_NOTE = (
    "Uncertainty is a hard bound: it comes from the speed of light, not from a "
    "fitted model, and it holds regardless of routing. The tighter model-based "
    "circle depends on an empirically calibrated delay law and has been measured "
    "to be over-confident on some networks -- treat it as indicative."
)


class Handler(BaseHTTPRequestHandler):
    scanner: Scanner
    server_version = "latlas"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass  # quiet: a scan logs hundreds of requests otherwise

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
        if self.path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(self.scanner.status()).encode(),
                       "application/json")
        elif self.path == "/events":
            self._stream()
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/scan":
            out = self.scanner.start()
            self._send(200, json.dumps(out).encode(), "application/json")
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
            # Bring a late subscriber up to date before streaming live events.
            st = self.scanner.status()
            self._emit(self._sse({"type": "hello", "state": st["state"]}))
            if st["state"] == "done" and st["result"]:
                self._emit(self._sse({"type": "done", "data": st["result"]}))
            while True:
                try:
                    ev = q.get(timeout=15.0)
                except queue.Empty:
                    self._emit(": keepalive\n\n")   # defeat idle proxies/timeouts
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


def serve(host: str = "0.0.0.0", port: int = 8080, *,
          samples: int = DEFAULT_SAMPLES, workers: int = DEFAULT_WORKERS,
          quick: bool = False, autostart: bool = True,
          progress=print) -> None:
    """Serve the map UI until interrupted."""
    scanner = Scanner(samples=samples, workers=workers, quick=quick)
    handler = type("BoundHandler", (Handler,), {"scanner": scanner})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    shown = "0.0.0.0" if host in ("", "0.0.0.0") else host
    progress(f"latlas web UI on http://{shown}:{port}/  "
             f"(reachable at http://<this-machine-ip>:{port}/ from other devices)")
    progress(f"  scanning with {samples} echoes/anchor, {workers} parallel workers")
    if autostart:
        scanner.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        progress("\nstopping")
    finally:
        httpd.server_close()
