"""Peregrine Autopilot Controller — local web app for sim use."""

from __future__ import annotations
import subprocess, signal, os, time, json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ── Config ────────────────────────────────────────────────────────
AUTOPILOT_DIR  = Path(__file__).parent.parent
VENV_PYTHON    = AUTOPILOT_DIR / ".venv/bin/python"
LOG_FILE       = Path("/tmp/peregrine.log")
PID_FILE       = Path("/tmp/peregrine.pid")
MODULE         = "uav.main"
START_ARGS     = ["--wait-for-fly"]

# ── Process helpers ───────────────────────────────────────────────

def _pid() -> int | None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "uav.main"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.split() if p.strip()]
        return pids[0] if pids else None
    except Exception:
        return None

def start_autopilot() -> dict:
    if _pid():
        return {"ok": False, "msg": "Already running"}
    log = open(LOG_FILE, "a")
    proc = subprocess.Popen(
        [str(VENV_PYTHON), "-B", "-u", "-m", MODULE, *START_ARGS],
        cwd=str(AUTOPILOT_DIR),
        stdout=log, stderr=log,
        start_new_session=True,
    )
    time.sleep(0.5)
    pid = _pid()
    return {"ok": bool(pid), "msg": f"Started PID {pid}" if pid else "Failed to start", "pid": pid}

def stop_autopilot() -> dict:
    pid = _pid()
    if not pid:
        return {"ok": False, "msg": "Not running"}
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.8)
        if _pid():
            os.kill(pid, signal.SIGKILL)
        return {"ok": True, "msg": f"Stopped PID {pid}"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

def reset_autopilot() -> dict:
    stop_autopilot()
    time.sleep(1.0)
    return start_autopilot()

def status() -> dict:
    pid = _pid()
    last_lines = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 6000))
                raw = f.read().decode("utf-8", errors="replace")
                last_lines = raw.strip().splitlines()[-80:]
        except Exception:
            pass
    return {"running": bool(pid), "pid": pid, "log": last_lines}

def clear_log() -> dict:
    try:
        open(LOG_FILE, "w").close()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

# ── HTTP handler ─────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # silence access log

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            self._html(HTML)
        elif p == "/api/status":
            self._json(status())
        elif p == "/api/start":
            self._json(start_autopilot())
        elif p == "/api/stop":
            self._json(stop_autopilot())
        elif p == "/api/reset":
            self._json(reset_autopilot())
        elif p == "/api/clear_log":
            self._json(clear_log())
        else:
            self._json({"error": "not found"}, 404)


# ── UI ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Peregrine Autopilot Controller</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --text:    #e6edf3;
    --muted:   #8b949e;
    --green:   #3fb950;
    --red:     #f85149;
    --yellow:  #d29922;
    --blue:    #58a6ff;
    --purple:  #bc8cff;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; min-height: 100vh; }

  header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 24px; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .logo { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; color: var(--text); }
  .logo span { color: var(--blue); }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;
    border: 1px solid;
  }
  .pill.online  { color: var(--green);  border-color: var(--green);  background: rgba(63,185,80,.1); }
  .pill.offline { color: var(--red);    border-color: var(--red);    background: rgba(248,81,73,.1); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
  .dot.pulse { animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .pid { margin-left: auto; font-size: 12px; color: var(--muted); font-family: monospace; }

  main { max-width: 1100px; margin: 0 auto; padding: 24px; display: grid; gap: 20px; }

  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px;
  }
  .card-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); margin-bottom: 16px; }

  .btn-group { display: flex; gap: 10px; flex-wrap: wrap; }
  button {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 9px 18px; border-radius: 7px; font-size: 14px; font-weight: 600;
    border: none; cursor: pointer; transition: opacity .15s, transform .1s;
  }
  button:hover  { opacity: .85; }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .4; cursor: not-allowed; }

  .btn-start  { background: var(--green);  color: #000; }
  .btn-stop   { background: var(--red);    color: #fff; }
  .btn-reset  { background: var(--yellow); color: #000; }
  .btn-clear  { background: var(--border); color: var(--muted); font-size: 12px; padding: 6px 12px; }

  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat { display: flex; flex-direction: column; gap: 4px; }
  .stat-label { font-size: 11px; color: var(--muted); }
  .stat-value { font-size: 18px; font-weight: 700; font-family: monospace; }

  .log-wrap {
    background: #0d1117; border: 1px solid var(--border); border-radius: 8px;
    height: 420px; overflow-y: auto; padding: 12px 14px;
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.6;
  }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-line.err   { color: var(--red); }
  .log-line.warn  { color: var(--yellow); }
  .log-line.phase { color: var(--purple); font-weight: 600; }
  .log-line.ok    { color: var(--green); }
  .log-line.nav   { color: var(--blue); }
  .log-line.dim   { color: var(--muted); }

  .toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--surface); border: 1px solid var(--border);
    padding: 10px 18px; border-radius: 8px; font-size: 13px;
    opacity: 0; transition: opacity .3s; pointer-events: none; z-index: 99;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>

<header>
  <div class="logo">Peregrine <span>//</span> Autopilot</div>
  <div id="status-pill" class="pill offline">
    <div class="dot" id="status-dot"></div>
    <span id="status-text">Offline</span>
  </div>
  <div class="pid" id="pid-label"></div>
</header>

<main>
  <!-- Controls -->
  <div class="card">
    <div class="card-title">Controls</div>
    <div class="btn-group">
      <button class="btn-start" onclick="api('start')">▶ Start</button>
      <button class="btn-stop"  onclick="api('stop')">■ Stop</button>
      <button class="btn-reset" onclick="api('reset')">↺ Reset</button>
    </div>
  </div>

  <div class="row">
    <!-- Stats -->
    <div class="card">
      <div class="card-title">Live Status</div>
      <div class="stat-grid">
        <div class="stat">
          <div class="stat-label">State</div>
          <div class="stat-value" id="s-mode">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Altitude</div>
          <div class="stat-value" id="s-alt">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Speed</div>
          <div class="stat-value" id="s-spd">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Vert Speed</div>
          <div class="stat-value" id="s-vs">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Heading</div>
          <div class="stat-value" id="s-hdg">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">AGL</div>
          <div class="stat-value" id="s-agl">—</div>
        </div>
      </div>
    </div>

    <!-- Controls output -->
    <div class="card">
      <div class="card-title">Control Surfaces</div>
      <div class="stat-grid">
        <div class="stat">
          <div class="stat-label">Throttle</div>
          <div class="stat-value" id="s-thr">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Pitch</div>
          <div class="stat-value" id="s-pitch">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Roll</div>
          <div class="stat-value" id="s-roll">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Brake</div>
          <div class="stat-value" id="s-brake">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Gear</div>
          <div class="stat-value" id="s-gear">—</div>
        </div>
        <div class="stat">
          <div class="stat-label">Accuracy</div>
          <div class="stat-value" id="s-acc">—</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Log -->
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <div class="card-title" style="margin:0">Live Log</div>
      <button class="btn-clear" onclick="api('clear_log')">Clear</button>
    </div>
    <div class="log-wrap" id="log-box"></div>
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
let autoScroll = true;
const logBox = document.getElementById('log-box');
logBox.addEventListener('scroll', () => {
  autoScroll = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 30;
});

// ── Log line colorizer ────────────────────────────────────────────
function colorLine(raw) {
  const line = raw.trim();
  if (!line) return null;
  let cls = 'dim';
  if (/error|traceback|exception|failed/i.test(line)) cls = 'err';
  else if (/warn/i.test(line))          cls = 'warn';
  else if (/Mode=(CLIMB|CRUISE|APPROACH|LAND|FLARE|ROLLOUT|GROUND)/i.test(line)) cls = 'dim';
  else if (/\[GPS\]|\[NAV\]|\[PATH\]/i.test(line))  cls = 'nav';
  else if (/\[PEREGRINE\]|\[PREFLIGHT\]/i.test(line)) cls = 'ok';
  else if (/phase|transition/i.test(line)) cls = 'phase';
  const div = document.createElement('div');
  div.className = 'log-line ' + cls;
  div.textContent = line;
  return div;
}

// ── Parse last Mode= line for stats ──────────────────────────────
function parseStats(lines) {
  // Find last Mode line
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = lines[i].match(/Mode=(\S+)\s+Alt=([\d.]+)ft\s+AGL=([\d.-]+)m\s+VS=([+\-\d]+)fpm\s+Hdg=([\d.]+)deg\s+Spd=([\d.-]+)kts\s+Acc=([\d.]+)%\/([\d.]+)%.*cmd\[T=([\d.]+)\s+P=([+\-\d.]+)\s+R=([+\-\d.]+).*B=([\d.]+).*G=(\w+)/);
    if (m) return {
      mode: m[1], alt: m[2]+'ft', agl: m[3]+'m', vs: m[4]+'fpm',
      hdg: m[5]+'°', spd: m[6]+'kts', acc: m[7]+'%',
      thr: (parseFloat(m[9])*100).toFixed(0)+'%',
      pitch: m[10], roll: m[11], brake: m[12], gear: m[13].toUpperCase()
    };
  }
  return null;
}

// ── Poll ──────────────────────────────────────────────────────────
let lastLogLen = 0;

async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    // Status pill
    const pill = document.getElementById('status-pill');
    const dot  = document.getElementById('status-dot');
    const txt  = document.getElementById('status-text');
    const pidL = document.getElementById('pid-label');
    if (d.running) {
      pill.className = 'pill online';
      dot.className  = 'dot pulse';
      txt.textContent = 'Running';
      pidL.textContent = `PID ${d.pid}`;
    } else {
      pill.className = 'pill offline';
      dot.className  = 'dot';
      txt.textContent = 'Offline';
      pidL.textContent = '';
    }

    // Stats
    const stats = parseStats(d.log || []);
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val ?? '—'; };
    if (stats) {
      set('s-mode',  stats.mode);
      set('s-alt',   stats.alt);
      set('s-agl',   stats.agl);
      set('s-vs',    stats.vs);
      set('s-hdg',   stats.hdg);
      set('s-spd',   stats.spd);
      set('s-thr',   stats.thr);
      set('s-pitch', stats.pitch);
      set('s-roll',  stats.roll);
      set('s-brake', stats.brake);
      set('s-gear',  stats.gear);
      set('s-acc',   stats.acc);
    } else if (!d.running) {
      ['s-mode','s-alt','s-agl','s-vs','s-hdg','s-spd','s-thr','s-pitch','s-roll','s-brake','s-gear','s-acc'].forEach(id => set(id, '—'));
    }

    // Log
    const lines = d.log || [];
    if (lines.length !== lastLogLen) {
      // Append only new lines
      const newLines = lines.slice(lastLogLen);
      const frag = document.createDocumentFragment();
      newLines.forEach(l => {
        const el = colorLine(l);
        if (el) frag.appendChild(el);
      });
      logBox.appendChild(frag);
      lastLogLen = lines.length;
      if (autoScroll) logBox.scrollTop = logBox.scrollHeight;
    }

  } catch(e) {}
}

// ── API call ──────────────────────────────────────────────────────
async function api(action) {
  try {
    const r = await fetch(`/api/${action}`);
    const d = await r.json();
    showToast(d.msg || d.ok);
    // If clear_log, reset log state
    if (action === 'clear_log') { logBox.innerHTML = ''; lastLogLen = 0; }
    await poll();
  } catch(e) { showToast('Error: ' + e.message); }
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = String(msg);
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

setInterval(poll, 800);
poll();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = 7777
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Peregrine Controller → http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
