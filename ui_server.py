from flask import Flask, request, Response
import json
import time
import threading

app = Flask(__name__)

_state_lock = threading.Lock()
_state = {
    "status": "Starting…",
    "body": "",
    "ts": time.time(),
}

_clients = []
_clients_lock = threading.Lock()


def _broadcast(state: dict):
    payload = f"data: {json.dumps(state)}\n\n"
    with _clients_lock:
        dead = []
        for q in _clients:
            try:
                q.append(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            _clients.remove(q)


@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SimpleBot</title>
<style>
html, body {
  margin:0; padding:0; width:100%; height:100%;
  background:#000; color:#fff;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
.wrap {
  height:100%;
  display:flex;
  flex-direction:column;
  justify-content:center;
  align-items:center;
  text-align:center;
  padding:24px;
  gap:14px;
}
#status {
  font-size:54px;
  font-weight:700;
}
#body {
  font-size:40px;
  color:#cfcfcf;
}
.dot {
  position:fixed;
  right:16px; bottom:16px;
  width:10px; height:10px;
  border-radius:50%;
  background:#666;
}
</style>
</head>
<body>
  <div class="wrap">
    <div id="status">Starting…</div>
    <div id="body"></div>
  </div>
  <div class="dot" id="dot"></div>

<script>
const statusEl = document.getElementById("status");
const bodyEl = document.getElementById("body");
const dotEl = document.getElementById("dot");

let t=0;
setInterval(() => {
  t = (t+1)%60;
  dotEl.style.background = (t<30) ? "#aaa" : "#444";
}, 250);

const ev = new EventSource("/events");
ev.onmessage = (msg) => {
  try {
    const s = JSON.parse(msg.data);
    statusEl.textContent = s.status || "";
    bodyEl.textContent = s.body || "";
  } catch (e) {}
};
</script>
</body>
</html>
"""


@app.get("/events")
def events():
    q = []
    with _clients_lock:
        _clients.append(q)

    with _state_lock:
        initial = f"data: {json.dumps(_state)}\n\n"
    q.append(initial)

    def gen():
        try:
            while True:
                if q:
                    yield q.pop(0)
                else:
                    time.sleep(0.05)
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)

    return Response(gen(), mimetype="text/event-stream")


@app.post("/update")
def update():
    data = request.get_json(force=True, silent=True) or {}
    with _state_lock:
        _state["status"] = (data.get("status") or "").strip()
        _state["body"] = (data.get("body") or "").strip()
        _state["ts"] = time.time()
        snapshot = dict(_state)

    _broadcast(snapshot)
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)


