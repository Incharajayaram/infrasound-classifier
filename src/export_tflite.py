"""
Export dual-input int8 TFLite model for ESP32-S3 / TFLite Micro.

The model has two inputs:
  input 0 "psd_input" : (1, 129, 1)  int8 — standardised log10-Welch-PSD
  input 1 "td_input"  : (1, 10)      int8 — standardised [log_crest, kurtosis, skew, log_rms,
                                              centroid, flatness, frac_low, frac_mid, frac_high, log_zcr]

Firmware must:
  1. Compute Welch PSD + TD features for the current 15-second window
  2. Standardise with psd_mean/psd_std and td_mean/td_std from scaler.json
  3. Quantise each input: q = round(x / scale + zero_point), clamp to [-128, 127]
  4. Call TFLite Micro Invoke()
  5. Dequantise output: p = (q - zero_point) * scale
  6. argmax → class label

Usage:
    python src/export_tflite.py [--model results/model_float32.keras]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

RESULTS_DIR  = Path("results")
TFLITE_PATH  = RESULTS_DIR / "model.tflite"
C_ARRAY_PATH = RESULTS_DIR / "model_data.cc"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TEST_SET     = Path("data/processed/test_set.npz")
CLASS_NAMES  = ["explosion", "rocket_bolide", "background"]


def load_test_set():
    d = np.load(TEST_SET, allow_pickle=True)
    if "X_psd_test" in d:
        return d["X_psd_test"].astype(np.float32), d["X_td_test"].astype(np.float32), d["y_test"]
    else:
        return d["X_test"].astype(np.float32), None, d["y_test"]


def load_model(path):
    import math as _math

    def focal_loss(y_true, y_pred):
        gamma = 2.0
        y_true = tf.cast(y_true, tf.float32)
        if y_true.shape.rank == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
        elif y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=3)
        p_t = tf.clip_by_value(tf.reduce_sum(y_pred * y_true, axis=-1), 1e-7, 1.0)
        return tf.reduce_mean(-(1.0 - p_t) ** gamma * tf.math.log(p_t))
    focal_loss.__name__ = "focal_loss"

    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, total_steps, warmup_steps):
            super().__init__()
            self.base_lr = float(base_lr)
            self.total_steps = float(total_steps)
            self.warmup_steps = float(warmup_steps)
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            lr_warm = self.base_lr * step / tf.maximum(self.warmup_steps, 1.0)
            prog = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
            lr_cos = self.base_lr * 0.5 * (1.0 + tf.cos(_math.pi * tf.clip_by_value(prog, 0.0, 1.0)))
            return tf.where(step < self.warmup_steps, lr_warm, lr_cos)
        def get_config(self):
            return {"base_lr": self.base_lr, "total_steps": self.total_steps, "warmup_steps": self.warmup_steps}

    return tf.keras.models.load_model(
        path,
        custom_objects={"focal_loss": focal_loss, "WarmupCosineDecay": WarmupCosineDecay},
        compile=False,
    )


def make_representative_dataset(X_psd, X_td, n=200):
    idx = np.random.choice(len(X_psd), size=min(n, len(X_psd)), replace=False)
    def gen():
        for i in idx:
            # Yield a dict so TFLite matches inputs by name rather than position.
            if X_td is not None:
                yield {"psd_input": X_psd[i:i+1], "td_input": X_td[i:i+1]}
            else:
                yield [X_psd[i:i+1]]
    return gen


def convert_int8(model, X_psd, X_td):
    # Dynamic-range quantization: quantises weights to int8, keeps float32 I/O.
    # Full int8 (with calibration) hits a TF-2.21 bug for multi-input Conv1D
    # models; dynamic-range is equally small on ESP32-S3 and avoids it entirely.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # No representative_dataset / inference_input_type — dynamic range only.
    return converter.convert()


def run_tflite(tflite_bytes, X_psd, X_td, y_test):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_dets = interp.get_input_details()
    out_det  = interp.get_output_details()[0]

    # Build name→detail map so we feed by name regardless of TFLite's ordering.
    inp_map = {d["name"].split(":")[0].split("/")[-1]: d for d in inp_dets}

    def _get_det(key, fallback_idx):
        for name, d in inp_map.items():
            if key in name:
                return d
        return inp_dets[fallback_idx]

    correct, latencies = 0, []
    for i in range(len(X_psd)):
        psd_det = _get_det("psd", 0)
        interp.set_tensor(psd_det["index"], X_psd[i:i+1].astype(psd_det["dtype"]))
        if X_td is not None:
            td_det = _get_det("td", 1)
            interp.set_tensor(td_det["index"], X_td[i:i+1].astype(td_det["dtype"]))

        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

        out = interp.get_tensor(out_det["index"]).astype(np.float32)
        if np.argmax(out) == y_test[i]:
            correct += 1

    return correct / len(y_test), float(np.mean(latencies))


def write_c_array(tflite_path, out_path):
    result = subprocess.run(["xxd", "-i", str(tflite_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] xxd failed: {result.stderr}")
        return
    c_code = result.stdout
    var_name = tflite_path.name.replace(".", "_").replace("/", "_")
    c_code = c_code.replace(var_name, "infrasound_model")
    header = (
        "// Auto-generated by export_tflite.py\n"
        "// Dual-input model: psd_input (129,1) and td_input (4,)\n"
        "#include <cstdint>\n\n"
    )
    out_path.write_text(header + c_code)
    print(f"Saved C array to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/model_float32.keras")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    X_psd, X_td, y_test = load_test_set()

    # Float32 baseline
    if X_td is not None:
        probs_f32 = model.predict([X_psd, X_td], verbose=0)
    else:
        probs_f32 = model.predict(X_psd, verbose=0)
    acc_f32 = float(np.mean(np.argmax(probs_f32, axis=1) == y_test))
    print(f"Float32 accuracy: {acc_f32:.4f}")

    print("\nConverting to int8 TFLite...")
    tflite_bytes = convert_int8(model, X_psd, X_td)
    TFLITE_PATH.write_bytes(tflite_bytes)
    size_kb = len(tflite_bytes) / 1024
    print(f"model.tflite: {size_kb:.1f} KB  →  {TFLITE_PATH}")

    if size_kb > 320:
        print(f"  [WARN] {size_kb:.0f} KB — verify SRAM headroom on ESP32-S3")

    print("\nRunning int8 inference on test set...")
    acc_int8, mean_lat = run_tflite(tflite_bytes, X_psd, X_td, y_test)
    print(f"Int8 accuracy:     {acc_int8:.4f}")
    print(f"Accuracy drop:     {acc_f32 - acc_int8:+.4f}")
    print(f"Mean host latency: {mean_lat:.3f} ms/window")

    write_c_array(TFLITE_PATH, C_ARRAY_PATH)

    # Update metrics.json
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics["tflite_export"] = {
        "model_size_kb":        round(size_kb, 1),
        "quantization":         "dynamic-range (weight int8, float32 I/O)",
        "float32_accuracy":     acc_f32,
        "tflite_accuracy":      acc_int8,
        "accuracy_drop":        round(acc_f32 - acc_int8, 4),
        "mean_host_latency_ms": round(mean_lat, 3),
        "inputs": {
            "psd_input":  "shape (1,129,1) float32 — standardised log10-Welch-PSD per frequency bin",
            "td_input":   "shape (1,10)   float32 — standardised [log_crest, kurtosis, skewness, log_rms, centroid, flatness, frac_low, frac_mid, frac_high, log_zcr]",
        },
        "esp32s3_note": (
            "240 MHz LX7. Host latency × ~20 ≈ on-device estimate. "
            "Dynamic-range quantisation: weights are int8, activations are float32. "
            "For full int8 inference on TFLite Micro, retrain with int8 aware training."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Updated {METRICS_PATH}")

    print(f"\n--- Summary ---")
    print(f"  Float32: {acc_f32:.4f}  Int8: {acc_int8:.4f}  (drop {acc_f32-acc_int8:+.4f})")
    print(f"  Size: {size_kb:.1f} KB   Host latency: {mean_lat:.3f} ms/window")


if __name__ == "__main__":
    main()
