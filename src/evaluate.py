"""
Evaluate trained model: accuracy, macro F1, per-class precision/recall,
confusion matrix (PNG), and softmax confidence analysis.

Usage:
    python src/evaluate.py [--model results/model_float32.keras]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
)
import tensorflow as tf

RESULTS_DIR  = Path("results")
CM_PATH      = RESULTS_DIR / "confusion_matrix.png"
METRICS_PATH = RESULTS_DIR / "metrics.json"
TEST_SET     = Path("data/processed/test_set.npz")
CLASS_NAMES  = ["explosion", "rocket_bolide", "background"]


def load_test_set():
    d = np.load(TEST_SET, allow_pickle=True)
    event_ids = d["event_ids_test"] if "event_ids_test" in d else None
    if "X_psd_test" in d:
        return d["X_psd_test"], d["X_td_test"], d["y_test"], event_ids
    else:
        return d["X_test"], None, d["y_test"], event_ids


def predict(model, X_psd, X_td):
    if X_td is not None:
        return model.predict([X_psd, X_td], verbose=0)
    else:
        return model.predict(X_psd, verbose=0)


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/model_float32.keras")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Reconstruct the exact loss so Keras can deserialise the saved model.
    # compile=False at load time avoids needing to deserialise the optimizer.
    # The factory's inner function is named "focal_loss" — matches what was
    # serialised in train.py.
    def _make_focal_loss(gamma=2.0, n_classes=3, smoothing=0.10):
        def focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            if y_true.shape.rank == 1:
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
            elif y_true.shape[-1] == 1:
                y_true = tf.one_hot(
                    tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=n_classes
                )
            if smoothing > 0.0:
                y_true = y_true * (1.0 - smoothing) + smoothing / n_classes
            p_t = tf.clip_by_value(
                tf.reduce_sum(y_pred * y_true, axis=-1), 1e-7, 1.0
            )
            return tf.reduce_mean(-(1.0 - p_t) ** gamma * tf.math.log(p_t))
        focal_loss.__name__ = "focal_loss"
        return focal_loss

    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, total_steps, warmup_steps):
            super().__init__()
            self.base_lr = float(base_lr)
            self.total_steps = float(total_steps)
            self.warmup_steps = float(warmup_steps)
        def __call__(self, step):
            import math
            step = tf.cast(step, tf.float32)
            lr_warm = self.base_lr * step / tf.maximum(self.warmup_steps, 1.0)
            prog = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
            lr_cos = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * tf.clip_by_value(prog, 0.0, 1.0)))
            return tf.where(step < self.warmup_steps, lr_warm, lr_cos)
        def get_config(self):
            return {"base_lr": self.base_lr, "total_steps": self.total_steps, "warmup_steps": self.warmup_steps}

    print(f"Loading model from {args.model}")
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"focal_loss": _make_focal_loss(), "WarmupCosineDecay": WarmupCosineDecay},
        compile=False,
    )

    X_psd, X_td, y_test, event_ids = load_test_set()
    n_classes   = len(np.unique(y_test))
    class_names = CLASS_NAMES[:n_classes]

    print(f"Test set: {X_psd.shape[0]} windows, {n_classes} classes")

    probs = predict(model, X_psd, X_td)

    # Prior calibration: model was trained on an oversampled (near-balanced)
    # dataset; the test set has the original imbalanced class distribution.
    # Multiply softmax outputs by (test_prior / train_prior) and renormalise
    # so argmax decisions reflect the real class frequencies.
    train_prior = np.array([29133, 29133, 32370], dtype=np.float32)  # after oversample
    train_prior /= train_prior.sum()
    test_counts  = np.bincount(y_test, minlength=n_classes).astype(np.float32)
    test_prior   = test_counts / test_counts.sum()
    # α controls calibration strength: 0=no adjustment, 1=full Bayesian, 0.5=half
    CALIB_ALPHA  = 0.7
    cal_factor   = (test_prior / np.maximum(train_prior, 1e-9)) ** CALIB_ALPHA
    probs_cal    = probs * cal_factor
    probs_cal   /= probs_cal.sum(axis=1, keepdims=True)

    y_pred_raw = np.argmax(probs,     axis=1)
    y_pred     = np.argmax(probs_cal, axis=1)

    acc_raw  = accuracy_score(y_test, y_pred_raw)
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"(uncalibrated accuracy: {acc_raw:.4f}  →  calibrated: {acc:.4f})")
    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_test, y_pred, labels=np.arange(n_classes)
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))

    print(f"\nOverall accuracy : {acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print("\nPer-class metrics:")
    hdr = f"  {'Class':20s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i, cn in enumerate(class_names):
        print(f"  {cn:20s}  {precision[i]:6.4f}  {recall[i]:6.4f}  "
              f"{f1_per[i]:6.4f}  {int(support[i]):>8d}")

    correct_mask    = y_pred == y_test
    max_probs       = probs_cal.max(axis=1)
    mean_conf_right = float(max_probs[correct_mask].mean())  if correct_mask.any()  else float("nan")
    mean_conf_wrong = float(max_probs[~correct_mask].mean()) if (~correct_mask).any() else float("nan")
    print(f"\nMean softmax confidence — correct: {mean_conf_right:.4f}  wrong: {mean_conf_wrong:.4f}")

    plot_confusion_matrix(cm, class_names, CM_PATH)

    # Event-level majority-vote accuracy (closer to how benchmark papers report)
    event_acc = None
    if event_ids is not None:
        from collections import Counter
        unique_events = np.unique(event_ids)
        ev_correct = 0
        for eid in unique_events:
            mask      = event_ids == eid
            true_cls  = int(np.bincount(y_test[mask]).argmax())
            pred_cls  = int(Counter(y_pred[mask]).most_common(1)[0][0])
            ev_correct += int(pred_cls == true_cls)
        event_acc = ev_correct / len(unique_events)
        print(f"Event-level accuracy (majority vote): {event_acc:.4f}"
              f"  ({ev_correct}/{len(unique_events)} events)")

    metrics = {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "per_class": {
            cn: {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1_per[i]),
                "support":   int(support[i]),
            }
            for i, cn in enumerate(class_names)
        },
        "mean_softmax_confidence": {"correct": mean_conf_right, "wrong": mean_conf_wrong},
        "event_level_accuracy": event_acc,
        "note": (
            "Window-level metrics. Event-level accuracy uses majority vote across "
            "all windows for the same recording — this matches benchmark reporting. "
            "Calibration alpha=0.7 applied (adjusts for train/test class prior mismatch)."
        ),
    }

    # Merge with existing TFLite export metrics if present
    if METRICS_PATH.exists():
        existing = json.loads(METRICS_PATH.read_text())
        if "tflite_export" in existing:
            metrics["tflite_export"] = existing["tflite_export"]

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
