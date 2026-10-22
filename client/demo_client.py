"""
Stand-alone command-line client for the Triton server.

Shows the same inference call over REST (HTTP :8000) and gRPC (:8001), then
runs a concurrent test where BOTH models are queried at once from a single
client application.

    python client/demo_client.py                 # REST + gRPC demo + concurrent test
    python client/demo_client.py --requests 1000 --workers 64
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

CHURN, DIGIT = "churn_model", "digit_image_model"


def infer_rest(client, model, input_name, array):
    inp = httpclient.InferInput(input_name, list(array.shape), "FP32")
    inp.set_data_from_numpy(array, binary_data=True)
    out = httpclient.InferRequestedOutput("probabilities", binary_data=True)
    return client.infer(model, [inp], outputs=[out]).as_numpy("probabilities")


def infer_grpc(client, model, input_name, array):
    inp = grpcclient.InferInput(input_name, list(array.shape), "FP32")
    inp.set_data_from_numpy(array)
    out = grpcclient.InferRequestedOutput("probabilities")
    return client.infer(model, [inp], outputs=[out]).as_numpy("probabilities")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", default="localhost:8000")
    ap.add_argument("--grpc", default="localhost:8001")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    rest = httpclient.InferenceServerClient(args.http)
    grpc = grpcclient.InferenceServerClient(args.grpc)
    assert rest.is_server_ready(), "Triton is not ready - run `docker compose up` first"

    print("Models in repository:")
    for m in rest.get_model_repository_index():
        print("  -", m["name"], m.get("state", ""))

    # one at-risk customer: tenure 3, $105, 7 calls, monthly contract, 4 late, 85 GB, age 29, 1 service
    customer = np.array([[3, 105, 7, 0, 4, 85, 29, 1]], dtype=np.float32)
    # a blank 8x8 image with a vertical bar (looks like a "1")
    image = np.zeros((1, 1, 8, 8), dtype=np.float32)
    image[0, 0, 1:7, 4] = 1.0

    for name, fn, c in (("REST", infer_rest, rest), ("gRPC", infer_grpc, grpc)):
        churn = fn(c, CHURN, "features", customer)[0]
        digit = fn(c, DIGIT, "image", image)[0]
        print(f"\n[{name}] churn probability = {churn[1]:.3f}   digit prediction = {int(digit.argmax())}")

    # ---- concurrent test: both models, one client application -------------------
    rng = np.random.default_rng(0)

    def job(i):
        if i % 2 == 0:
            x = np.round(rng.uniform([0, 20, 0, 0, 0, 0, 18, 1], [72, 120, 10, 2, 6, 100, 80, 8])).astype(np.float32)[None, :]
            return infer_grpc(grpc, CHURN, "features", x)
        return infer_grpc(grpc, DIGIT, "image", rng.uniform(0, 1, (1, 1, 8, 8)).astype(np.float32))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(job, range(args.requests)))
    dt = time.perf_counter() - t0
    print(f"\nConcurrent test (gRPC, both models): {args.requests} requests, {args.workers} workers "
          f"-> {dt:.2f}s, {args.requests / dt:.0f} req/s")

    for m in (CHURN, DIGIT):
        s = rest.get_inference_statistics(m)["model_stats"][0]
        inf, exe = int(s["inference_count"]), int(s["execution_count"])
        print(f"  {m}: inferences={inf} executions={exe} avg batch={inf / max(exe, 1):.2f}")


if __name__ == "__main__":
    main()
