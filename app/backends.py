"""
Inference backends used by the dashboard gateway.

TritonBackend : talks to a real Triton Inference Server over REST (HTTP, :8000)
                or gRPC (:8001) using the official `tritonclient` package.
DemoBackend   : fallback that runs the very same ONNX files with ONNX Runtime
                so the dashboard still works when Triton is not running.
"""
import threading
from pathlib import Path

import numpy as np

CHURN = "churn_model"
DIGIT = "digit_image_model"
INPUT_NAME = {CHURN: "features", DIGIT: "image"}
OUTPUT_NAME = "probabilities"


class TritonBackend:
    name = "triton"

    def __init__(self, http_url="localhost:8000", grpc_url="localhost:8001"):
        import tritonclient.http as httpclient   # noqa: F401  (fail early if missing)
        import tritonclient.grpc as grpcclient   # noqa: F401
        self.http_url, self.grpc_url = http_url, grpc_url
        self._local = threading.local()           # one client per thread

    # -- clients ---------------------------------------------------------------
    def _client(self, protocol):
        import tritonclient.http as httpclient
        import tritonclient.grpc as grpcclient
        cache = self._local.__dict__.setdefault("clients", {})
        if protocol not in cache:
            if protocol == "grpc":
                cache[protocol] = grpcclient.InferenceServerClient(self.grpc_url)
            else:
                cache[protocol] = httpclient.InferenceServerClient(
                    self.http_url, connection_timeout=2.0, network_timeout=10.0)
        return cache[protocol]

    # -- health ----------------------------------------------------------------
    def is_ready(self):
        try:
            c = self._client("rest")
            return bool(c.is_server_ready() and c.is_model_ready(CHURN)
                        and c.is_model_ready(DIGIT))
        except Exception:
            return False

    # -- inference -------------------------------------------------------------
    def infer(self, model, array, protocol="rest"):
        array = np.ascontiguousarray(array, dtype=np.float32)
        if protocol == "grpc":
            import tritonclient.grpc as mod
            inp = mod.InferInput(INPUT_NAME[model], list(array.shape), "FP32")
            inp.set_data_from_numpy(array)
            out = mod.InferRequestedOutput(OUTPUT_NAME)
        else:
            import tritonclient.http as mod
            inp = mod.InferInput(INPUT_NAME[model], list(array.shape), "FP32")
            inp.set_data_from_numpy(array, binary_data=True)
            out = mod.InferRequestedOutput(OUTPUT_NAME, binary_data=True)
        result = self._client(protocol).infer(model, [inp], outputs=[out])
        return result.as_numpy(OUTPUT_NAME)

    # -- server-side statistics (shows dynamic batching in action) -------------
    def stats(self):
        c = self._client("rest")
        out = {}
        for m in (CHURN, DIGIT):
            s = c.get_inference_statistics(m)["model_stats"][0]
            inf, exe = int(s.get("inference_count", 0)), int(s.get("execution_count", 0))
            batches = sorted(
                ({"batch_size": int(b["batch_size"]),
                  "count": int(b["compute_infer"]["count"])}
                 for b in s.get("batch_stats", [])),
                key=lambda b: b["batch_size"])
            ns = int(s.get("inference_stats", {}).get("compute_infer", {}).get("ns", 0))
            cnt = int(s.get("inference_stats", {}).get("compute_infer", {}).get("count", 0))
            out[m] = {
                "inference_count": inf,
                "execution_count": exe,
                "avg_batch_size": round(inf / exe, 2) if exe else 0,
                "avg_compute_ms": round(ns / cnt / 1e6, 3) if cnt else 0,
                "batches": batches,
            }
        return out


class DemoBackend:
    name = "demo"

    def __init__(self, model_repo: Path):
        import onnxruntime as ort
        self._sessions = {
            m: ort.InferenceSession(str(model_repo / m / "1" / "model.onnx"),
                                    providers=["CPUExecutionProvider"])
            for m in (CHURN, DIGIT)
        }

    def is_ready(self):
        return True

    def infer(self, model, array, protocol="rest"):
        array = np.ascontiguousarray(array, dtype=np.float32)
        return self._sessions[model].run([OUTPUT_NAME], {INPUT_NAME[model]: array})[0]

    def stats(self):
        return None   # only a real Triton server reports batching statistics
