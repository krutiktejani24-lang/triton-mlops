"""
Train the two models and export them to ONNX inside the Triton model repository.

  1. churn_model        - tabular customer-churn predictor (8 features)
  2. digit_image_model  - 8x8 grayscale image classifier (digits 0-9)

Run from the project root:   python scripts/export_models.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import shift as nd_shift
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from onnx_writer import build_mlp_onnx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "model_repository"
APP = ROOT / "app"
RNG = np.random.default_rng(42)

CHURN_FEATURES = [
    # name, label, min, max, step, default
    ("tenure_months",   "Tenure (months)",                0,  72, 1, 24),
    ("monthly_charges", "Monthly charges ($)",           20, 120, 1, 70),
    ("support_calls",   "Support calls (last 90 days)",   0,  10, 1, 2),
    ("contract_type",   "Contract (0=monthly, 1=1y, 2=2y)", 0, 2, 1, 0),
    ("late_payments",   "Late payments",                  0,   6, 1, 1),
    ("data_usage_gb",   "Data usage (GB / month)",        0, 100, 1, 40),
    ("age",             "Customer age",                  18,  80, 1, 40),
    ("num_services",    "Number of services",             1,   8, 1, 3),
]
PRESETS = {
    "Loyal customer": [60, 45, 0, 2, 0, 30, 50, 5],
    "At-risk customer": [3, 105, 7, 0, 4, 85, 29, 1],
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def verify(onnx_path, X, expected_proba, in_name):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {in_name: X.astype(np.float32)})[0]
    err = float(np.abs(got - expected_proba).max())
    assert err < 1e-4, f"ONNX output differs from sklearn (max err {err})"
    return err


# --------------------------------------------------------------- churn model --

def make_churn_data(n=8000):
    lo = np.array([f[2] for f in CHURN_FEATURES], dtype=float)
    hi = np.array([f[3] for f in CHURN_FEATURES], dtype=float)
    X = RNG.uniform(lo, hi, size=(n, len(CHURN_FEATURES)))
    X = np.round(X)
    tenure, monthly, calls, contract, late, usage, age, services = X.T
    z = (-0.045 * tenure + 0.035 * (monthly - 60) + 0.40 * calls
         - 0.85 * contract + 0.45 * late + 0.006 * (usage - 50)
         - 0.01 * (age - 45) - 0.15 * (services - 3))
    # calibrate the intercept so ~30% of customers churn
    a, b = -10.0, 10.0
    for _ in range(50):
        mid = (a + b) / 2
        if sigmoid(z + mid).mean() > 0.30:
            b = mid
        else:
            a = mid
    y = (RNG.uniform(size=n) < sigmoid(z + a)).astype(int)
    return X, y


def train_churn():
    X, y = make_churn_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(16,), activation="relu",
                        max_iter=600, random_state=1)
    clf.fit(scaler.transform(Xtr), ytr)

    p_te = clf.predict_proba(scaler.transform(Xte))[:, 1]
    acc = float(((p_te > 0.5) == yte).mean())
    auc = float(roc_auc_score(yte, p_te))

    # fold the StandardScaler into the first layer so the ONNX model takes raw features
    W1, b1 = clf.coefs_[0], clf.intercepts_[0]
    W1f = W1 / scaler.scale_[:, None]
    b1f = b1 - (scaler.mean_ / scaler.scale_) @ W1
    # binary logistic output -> two logits [0, z]; softmax gives [1-p, p]
    W2 = np.hstack([np.zeros((W1.shape[1], 1)), clf.coefs_[1]])
    b2 = np.array([0.0, clf.intercepts_[1][0]])

    onnx_bytes = build_mlp_onnx(
        [(W1f, b1f), (W2, b2)], "features", "probabilities",
        ["batch", len(CHURN_FEATURES)], graph_name="churn_model")
    path = REPO / "churn_model" / "1" / "model.onnx"
    path.write_bytes(onnx_bytes)

    expected = clf.predict_proba(scaler.transform(Xte[:200]))
    err = verify(path, Xte[:200], expected, "features")
    print(f"[churn_model]       acc={acc:.3f} auc={auc:.3f}  onnx-vs-sklearn max err={err:.2e}")
    return {"accuracy": round(acc, 4), "auc": round(auc, 4)}


# --------------------------------------------------------------- image model --

def augment(images, labels, copies=2):
    xs, ys = [images], [labels]
    for _ in range(copies):
        out = np.empty_like(images)
        for i, img in enumerate(images):
            dy, dx = RNG.integers(-1, 2, size=2)
            out[i] = nd_shift(img.reshape(8, 8), (dy, dx), order=0, cval=0).ravel()
        xs.append(out)
        ys.append(labels)
    return np.vstack(xs), np.concatenate(ys)


def train_digits():
    d = load_digits()
    X = d.data / 16.0                     # pixels 0..1
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.2,
                                          random_state=1, stratify=d.target)
    Xaug, yaug = augment(Xtr, ytr)
    clf = MLPClassifier(hidden_layer_sizes=(64,), activation="relu",
                        max_iter=400, random_state=1)
    clf.fit(Xaug, yaug)
    acc = float(clf.score(Xte, yte))

    layers = [(clf.coefs_[0], clf.intercepts_[0]), (clf.coefs_[1], clf.intercepts_[1])]
    onnx_bytes = build_mlp_onnx(layers, "image", "probabilities",
                                ["batch", 1, 8, 8], flatten=True,
                                graph_name="digit_image_model")
    path = REPO / "digit_image_model" / "1" / "model.onnx"
    path.write_bytes(onnx_bytes)

    expected = clf.predict_proba(Xte[:200])
    err = verify(path, Xte[:200].reshape(-1, 1, 8, 8), expected, "image")
    print(f"[digit_image_model] acc={acc:.3f}              onnx-vs-sklearn max err={err:.2e}")

    # a few real test images for the dashboard's "load sample" button
    idx = RNG.choice(len(Xte), size=40, replace=False)
    samples = [{"pixels": [round(float(v), 4) for v in Xte[i]], "label": int(yte[i])}
               for i in idx]
    return {"accuracy": round(acc, 4)}, samples


def main():
    churn_metrics = train_churn()
    digit_metrics, samples = train_digits()
    info = {
        "churn": {
            "features": [dict(name=n, label=l, min=a, max=b, step=s, default=dflt)
                         for n, l, a, b, s, dflt in CHURN_FEATURES],
            "presets": PRESETS,
            **churn_metrics,
        },
        "digit": {**digit_metrics},
        "samples": samples,
    }
    (APP / "model_info.json").write_text(json.dumps(info))
    print("Models written to", REPO)


if __name__ == "__main__":
    main()
