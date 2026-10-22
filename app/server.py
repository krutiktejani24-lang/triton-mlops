"""
Dashboard gateway (Flask).

Serves the browser dashboard and exposes a small JSON API. Each prediction is
forwarded to Triton (REST or gRPC, chosen in the UI). If Triton is not
reachable the gateway falls back to ONNX Runtime ("demo mode").

Environment variables
  TRITON_HTTP_URL  default localhost:8000
  TRITON_GRPC_URL  default localhost:8001
  MODEL_REPO       default ../model_repository
  FORCE_DEMO=1     never talk to Triton
  PORT             default 5006
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from backends import CHURN, DIGIT, DemoBackend, TritonBackend

BASE = Path(__file__).resolve().parent
MODEL_REPO = Path(os.environ.get("MODEL_REPO", BASE.parent / "model_repository"))
TRITON_HTTP = os.environ.get("TRITON_HTTP_URL", "localhost:8000")
TRITON_GRPC = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
FORCE_DEMO = os.environ.get("FORCE_DEMO") == "1"

META = json.loads((BASE / "model_info.json").read_text())
FEATURES = META["churn"]["features"]

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")

demo_backend = DemoBackend(MODEL_REPO)
try:
    triton_backend = None if FORCE_DEMO else TritonBackend(TRITON_HTTP, TRITON_GRPC)
except ImportError:
    triton_backend = None   # tritonclient not installed -> demo mode only

_state = {"t": 0.0, "backend": demo_backend}


def get_backend():
    """Pick Triton when it is up (re-checked every 3 s), otherwise the demo backend."""
    now = time.time()
    if now - _state["t"] > 3:
        _state["t"] = now
        ok = triton_backend is not None and triton_backend.is_ready()
        _state["backend"] = triton_backend if ok else demo_backend
    return _state["backend"]


# ---------------------------------------------------------------- helpers ----

def predict(model, array, protocol):
    backend = get_backend()
    t0 = time.perf_counter()
    probs = backend.infer(model, array, protocol)
    ms = (time.perf_counter() - t0) * 1000
    return backend, probs, ms


def random_input(model, rng):
    if model == CHURN:
        lo = np.array([f["min"] for f in FEATURES])
        hi = np.array([f["max"] for f in FEATURES])
        return np.round(rng.uniform(lo, hi)).astype(np.float32).reshape(1, -1)
    return rng.uniform(0, 1, size=(1, 1, 8, 8)).astype(np.float32)


# ----------------------------------------------------------------- routes ----

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def api_meta():
    return jsonify(META)


@app.get("/api/status")
def api_status():
    b = get_backend()
    return jsonify({
        "mode": b.name,
        "triton_http": TRITON_HTTP,
        "triton_grpc": TRITON_GRPC,
        "models": [CHURN, DIGIT],
    })


@app.get("/api/config/<model>")
def api_config(model):
    if model not in (CHURN, DIGIT):
        return jsonify(error="unknown model"), 404
    return jsonify(text=(MODEL_REPO / model / "config.pbtxt").read_text())


@app.get("/api/stats")
def api_stats():
    b = get_backend()
    if b.name != "triton":
        return jsonify(available=False)
    try:
        return jsonify(available=True, models=b.stats())
    except Exception as e:  # noqa: BLE001
        return jsonify(available=False, error=str(e))


@app.post("/api/predict/churn")
def api_predict_churn():
    body = request.get_json(force=True)
    feats = body.get("features", [])
    if len(feats) != len(FEATURES):
        return jsonify(error=f"expected {len(FEATURES)} features"), 400
    protocol = body.get("protocol", "rest")
    try:
        x = np.array(feats, dtype=np.float32).reshape(1, -1)
        backend, probs, ms = predict(CHURN, x, protocol)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 502
    p = probs.reshape(-1).tolist()
    return jsonify(probabilities=p, churn_probability=p[1],
                   label="churn" if p[1] >= 0.5 else "stay",
                   latency_ms=round(ms, 2), protocol=protocol, mode=backend.name)


@app.post("/api/predict/digit")
def api_predict_digit():
    body = request.get_json(force=True)
    pixels = body.get("pixels", [])
    if len(pixels) != 64:
        return jsonify(error="expected 64 pixels"), 400
    protocol = body.get("protocol", "rest")
    try:
        x = np.clip(np.array(pixels, dtype=np.float32), 0, 1).reshape(1, 1, 8, 8)
        backend, probs, ms = predict(DIGIT, x, protocol)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 502
    p = probs.reshape(-1).tolist()
    return jsonify(probabilities=p, digit=int(np.argmax(p)),
                   latency_ms=round(ms, 2), protocol=protocol, mode=backend.name)


@app.post("/api/benchmark")
def api_benchmark():
    """Fire many concurrent requests at the server and report latency/throughput."""
    body = request.get_json(force=True)
    model = body.get("model", "both")
    protocol = body.get("protocol", "rest")
    n = max(1, min(int(body.get("requests", 200)), 5006))
    conc = max(1, min(int(body.get("concurrency", 16)), 128))

    backend = get_backend()
    rng = np.random.default_rng(0)
    models = [CHURN, DIGIT] if model == "both" else [CHURN if model == "churn" else DIGIT]
    jobs = [(models[i % len(models)], random_input(models[i % len(models)], rng))
            for i in range(n)]

    def work(job):
        m, x = job
        t0 = time.perf_counter()
        try:
            backend.infer(m, x, protocol)
            return m, (time.perf_counter() - t0) * 1000, None
        except Exception as e:  # noqa: BLE001
            return m, (time.perf_counter() - t0) * 1000, str(e)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(work, jobs))
    wall = time.perf_counter() - t_start

    def summarize(rows):
        lat = np.array([r[1] for r in rows if r[2] is None])
        if lat.size == 0:
            return {"ok": 0, "errors": len(rows)}
        return {"ok": int(lat.size), "errors": len(rows) - int(lat.size),
                "p50_ms": round(float(np.percentile(lat, 50)), 2),
                "p95_ms": round(float(np.percentile(lat, 95)), 2),
                "p99_ms": round(float(np.percentile(lat, 99)), 2),
                "mean_ms": round(float(lat.mean()), 2)}

    ok_lat = np.array([r[1] for r in results if r[2] is None])
    hist = []
    if ok_lat.size:
        counts, edges = np.histogram(ok_lat, bins=20)
        hist = [{"from": round(float(edges[i]), 2), "to": round(float(edges[i + 1]), 2),
                 "count": int(counts[i])} for i in range(len(counts))]
    first_err = next((r[2] for r in results if r[2]), None)

    return jsonify({
        "mode": backend.name, "protocol": protocol, "requests": n, "concurrency": conc,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(n / wall, 1) if wall else 0,
        "overall": summarize(results),
        "per_model": {m: summarize([r for r in results if r[0] == m]) for m in models},
        "histogram": hist,
        "first_error": first_err,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5006))
    print(f"Dashboard on http://localhost:{port}  (Triton: {TRITON_HTTP} / {TRITON_GRPC})")
    app.run(host="0.0.0.0", port=port, threaded=True)
