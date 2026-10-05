"""
Train the dual-input 1D-CNN infrasound classifier.

Model inputs:
  psd_input : (129, 1) — log10 Welch PSD, standardised per frequency bin
  td_input  : (10,)    — time-domain + spectral features, standardised

Split strategy: GroupShuffleSplit on (event_id + station_id) — no leakage.
Performance estimate: 5-fold GroupKFold cross-validation on the training set.
Final model: trained on full train+val data, evaluated on held-out test set.

v2 improvements:
  - Oversample minority classes (explosion 3×, rocket_bolide 2×) before aug
  - Class-weighted focal loss (inverse-frequency weights)
  - Deeper model: 4 Conv1D blocks (64-128-64-32) with MaxPooling between blocks
  - Reduced label smoothing (0.05) and mixup alpha (0.15)
  - Longer training (150 epochs, patience 25) and lower dropout (0.25)
  - Two-layer classifier head (128 → 64 → n_classes)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_PATH = Path("data/processed/dataset.npz")
SCALER_PATH  = Path("data/processed/scaler.json")
MODEL_PATH   = Path("results/model_float32.keras")
HISTORY_PATH = Path("results/training_history.json")
CV_PATH      = Path("results/cv_summary.json")

TEST_FRAC      = 0.20
BATCH_SIZE     = 64
MAX_EPOCHS     = 150
PATIENCE       = 25
RANDOM_SEED    = 42
FOCAL_GAMMA    = 2.0
LABEL_SMOOTH   = 0.05   # reduced from 0.10 — less penalty on minority classes
N_CV_FOLDS     = 5
WARMUP_EPOCHS  = 5
WEIGHT_DECAY   = 1e-4
MIXUP_ALPHA    = 0.15   # reduced from 0.30 — less mixing preserves minority signal
DROPOUT_RATE   = 0.25   # reduced from 0.35

CLASS_NAMES = ["explosion", "rocket_bolide", "background"]


# ── Focal loss with per-class weights ─────────────────────────────────────────
def make_focal_loss(gamma: float, n_classes: int, smoothing: float = 0.0,
                    class_weights=None):
    """
    Focal loss with optional label smoothing and class weighting.
    class_weights: list/array of length n_classes, e.g. inverse-frequency weights.
    Handles both integer labels and soft one-hot (from mixup).
    """
    w_tensor = (tf.constant(class_weights, dtype=tf.float32)
                if class_weights is not None else None)

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if y_true.shape.rank == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
        elif y_true.shape[-1] == 1:
            y_true = tf.one_hot(
                tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=n_classes)
        if smoothing > 0.0:
            y_true = y_true * (1.0 - smoothing) + smoothing / n_classes
        p_t  = tf.clip_by_value(
            tf.reduce_sum(y_pred * y_true, axis=-1), 1e-7, 1.0)
        loss = -(1.0 - p_t) ** gamma * tf.math.log(p_t)
        if w_tensor is not None:
            hard_class = tf.argmax(y_true, axis=-1)
            loss = loss * tf.gather(w_tensor, hard_class)
        return tf.reduce_mean(loss)

    focal_loss.__name__ = "focal_loss"
    return focal_loss


# ── Squeeze-and-Excitation block ──────────────────────────────────────────────
def se_block(x, ratio: int = 4):
    n_ch = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(1, n_ch // ratio), activation="relu")(se)
    se = layers.Dense(n_ch, activation="sigmoid")(se)
    se = layers.Reshape((1, n_ch))(se)
    return layers.Multiply()([x, se])


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(n_psd: int, n_td: int, n_classes: int) -> keras.Model:
    """
    4-block Conv1D with SE + MaxPooling, larger TD branch, two-layer head.
    Block filters: 64 → 128 → 64 → 32.  MaxPool after blocks 1 & 2.
    """
    psd_in = keras.Input(shape=(n_psd, 1), name="psd_input")

    # Block 1 — local spectral features
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(psd_in)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling1D(2)(x)           # 129 → 64

    # Block 2 — wider receptive field
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling1D(2)(x)           # 64 → 32

    # Block 3
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)

    # Block 4
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    psd_features = layers.GlobalAveragePooling1D()(x)   # (32,)

    # Time-domain branch
    td_in = keras.Input(shape=(n_td,), name="td_input")
    t = layers.Dense(64, activation="relu")(td_in)
    t = layers.BatchNormalization()(t)
    t = layers.Dropout(0.20)(t)
    td_features = layers.Dense(32, activation="relu")(t)  # (32,)

    # Classifier head
    merged = layers.Concatenate()([psd_features, td_features])  # (64,)
    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    return keras.Model(inputs=[psd_in, td_in], outputs=out)


# ── Scalers ───────────────────────────────────────────────────────────────────
def fit_scalers(X_psd, X_td):
    psd_mean = X_psd.mean(axis=0)
    psd_std  = X_psd.std(axis=0)  + 1e-8
    td_mean  = X_td.mean(axis=0)
    td_std   = X_td.std(axis=0)   + 1e-8
    return psd_mean, psd_std, td_mean, td_std


def apply_scalers(X_psd, X_td, psd_mean, psd_std, td_mean, td_std):
    return (
        ((X_psd - psd_mean) / psd_std).astype(np.float32),
        ((X_td  - td_mean)  / td_std ).astype(np.float32),
    )


def save_scalers(psd_mean, psd_std, td_mean, td_std, freq_axis=None):
    obj = {
        "psd_mean":  psd_mean.tolist(),
        "psd_std":   psd_std.tolist(),
        "td_mean":   td_mean.tolist(),
        "td_std":    td_std.tolist(),
        "td_feature_names": [
            "log10_crest_factor", "excess_kurtosis", "skewness", "log10_rms_pa",
            "spectral_centroid_hz", "spectral_flatness",
            "band_frac_low_lt1hz", "band_frac_mid_1to5hz", "band_frac_high_gt5hz",
            "log10_zcr",
        ],
    }
    if freq_axis is not None:
        obj["freq_axis_hz"] = freq_axis.tolist()
    SCALER_PATH.write_text(json.dumps(obj, indent=2))
    print(f"Saved scaler to {SCALER_PATH}")


# ── Oversampling ──────────────────────────────────────────────────────────────
def oversample_minority(X_psd, X_td, y, rng: np.random.Generator,
                        target_counts: dict) -> tuple:
    """
    Upsample each class until it reaches target_counts[class_idx].
    target_counts: {0: N_explosion, 1: N_rocket, 2: N_background}
    """
    parts_p, parts_t, parts_y = [X_psd], [X_td], [y]
    for cls, target in target_counts.items():
        mask    = y == cls
        current = mask.sum()
        if current >= target:
            continue
        need = target - current
        idxs = rng.choice(np.where(mask)[0], size=need, replace=True)
        parts_p.append(X_psd[idxs])
        parts_t.append(X_td[idxs])
        parts_y.append(y[idxs])
    return (np.concatenate(parts_p),
            np.concatenate(parts_t),
            np.concatenate(parts_y))


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_gain_noise(X_psd, X_td, y, rng: np.random.Generator):
    """Gain offset + additive noise; explosion gets 2× gain variance."""
    Xp, Xt = X_psd.copy(), X_td.copy()
    sigma = np.full(len(y), 0.15, dtype=np.float32)
    sigma[y == 0] = 0.30
    sigma[y == 1] = 0.20
    gain = rng.normal(0, 1, size=len(y)).astype(np.float32) * sigma
    Xp += gain[:, None]
    Xp += rng.normal(0, 0.05, size=Xp.shape).astype(np.float32)
    Xt += rng.normal(0, 0.03, size=Xt.shape).astype(np.float32)
    return (np.concatenate([X_psd, Xp]),
            np.concatenate([X_td,  Xt]),
            np.concatenate([y,     y]))


def mixup(X_psd, X_td, y, n_classes: int, rng: np.random.Generator,
          alpha: float = MIXUP_ALPHA):
    if alpha <= 0.0:
        y_oh = keras.utils.to_categorical(y, n_classes).astype(np.float32)
        return X_psd, X_td, y_oh
    n   = len(y)
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
    idx = rng.permutation(n)
    Xp_mix = lam[:, None] * X_psd + (1 - lam[:, None]) * X_psd[idx]
    Xt_mix = lam[:, None] * X_td  + (1 - lam[:, None]) * X_td[idx]
    y_oh   = keras.utils.to_categorical(y, n_classes).astype(np.float32)
    y_mix  = lam[:, None] * y_oh + (1 - lam[:, None]) * y_oh[idx]
    return Xp_mix, Xt_mix, y_mix


# ── LR schedule ───────────────────────────────────────────────────────────────
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int):
        super().__init__()
        self.base_lr      = float(base_lr)
        self.total_steps  = float(total_steps)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.warmup_steps
        total  = self.total_steps
        lr_w   = self.base_lr * step / tf.maximum(warmup, 1.0)
        prog   = (step - warmup) / tf.maximum(total - warmup, 1.0)
        lr_c   = self.base_lr * 0.5 * (
            1.0 + tf.cos(np.pi * tf.clip_by_value(prog, 0.0, 1.0)))
        return tf.where(step < warmup, lr_w, lr_c)

    def get_config(self):
        return {"base_lr": self.base_lr,
                "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}


# ── Compile ───────────────────────────────────────────────────────────────────
def compile_model(model, n_classes, class_weights=None,
                  steps_per_epoch=None, total_epochs=None):
    if steps_per_epoch and total_epochs:
        lr = WarmupCosineDecay(
            base_lr=1e-3,
            total_steps=steps_per_epoch * total_epochs,
            warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
        )
        optimizer = keras.optimizers.AdamW(learning_rate=lr,
                                           weight_decay=WEIGHT_DECAY)
    else:
        optimizer = keras.optimizers.AdamW(learning_rate=1e-3,
                                           weight_decay=WEIGHT_DECAY)
    model.compile(
        optimizer=optimizer,
        loss=make_focal_loss(FOCAL_GAMMA, n_classes, LABEL_SMOOTH, class_weights),
        metrics=["accuracy"],
    )
    return model


# ── Training helper ───────────────────────────────────────────────────────────
def train_model(model, X_psd_tr, X_td_tr, y_tr,
                X_psd_va, X_td_va, y_va, verbose=0):
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True,
        ),
    ]
    return model.fit(
        [X_psd_tr, X_td_tr], y_tr,
        validation_data=([X_psd_va, X_td_va], y_va),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=verbose,
    )


# ── Cross-validation ──────────────────────────────────────────────────────────
def cross_validate(X_psd, X_td, y, groups, n_classes, class_weights, rng):
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_accs = []
    print(f"\n5-fold GroupKFold cross-validation (train-set only)...")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_psd, y, groups)):
        Xp_tr, Xt_tr, y_tr = X_psd[tr_idx], X_td[tr_idx], y[tr_idx]
        Xp_va, Xt_va, y_va = X_psd[va_idx], X_td[va_idx], y[va_idx]

        pm, ps, tm, ts = fit_scalers(Xp_tr, Xt_tr)
        Xp_tr_s, Xt_tr_s = apply_scalers(Xp_tr, Xt_tr, pm, ps, tm, ts)
        Xp_va_s, Xt_va_s = apply_scalers(Xp_va, Xt_va, pm, ps, tm, ts)

        # Oversample on fold training set
        counts    = {c: int(np.bincount(y_tr).max() * 0.8) for c in range(n_classes)}
        Xp_os, Xt_os, y_os = oversample_minority(Xp_tr_s, Xt_tr_s, y_tr, rng, counts)

        Xp_aug, Xt_aug, y_aug = augment_gain_noise(Xp_os, Xt_os, y_os, rng)
        Xp_mix, Xt_mix, y_mix = mixup(Xp_aug, Xt_aug, y_aug, n_classes, rng)

        y_va_oh = keras.utils.to_categorical(y_va, n_classes).astype(np.float32)

        m = build_model(X_psd.shape[1], X_td.shape[1], n_classes)
        compile_model(m, n_classes, class_weights)
        train_model(m, Xp_mix[..., np.newaxis], Xt_mix, y_mix,
                    Xp_va_s[..., np.newaxis], Xt_va_s, y_va_oh, verbose=0)

        _, va_acc = m.evaluate(
            [Xp_va_s[..., np.newaxis], Xt_va_s], y_va_oh, verbose=0)
        fold_accs.append(float(va_acc))
        print(f"  Fold {fold+1}: val_acc={va_acc:.4f}")

    print(f"  CV mean: {np.mean(fold_accs):.4f}  ±  {np.std(fold_accs):.4f}")
    return fold_accs


def main():
    Path("results").mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible to TF: {gpus}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled.")
    else:
        print("WARNING: No GPU detected — running on CPU.")

    print("Loading dataset...")
    d        = np.load(DATASET_PATH, allow_pickle=True)
    X_psd    = d["X_psd"].astype(np.float32)
    X_td     = d["X_td"].astype(np.float32)
    y        = d["y"].astype(np.int32)

    for arr, name in [(X_psd, "X_psd"), (X_td, "X_td")]:
        bad = ~np.isfinite(arr)
        if bad.any():
            col_means = np.nanmean(arr, axis=0)
            arr[bad]  = np.take(col_means, np.where(bad)[1])
            print(f"  Replaced {bad.sum()} NaN/Inf values in {name}")

    event_ids   = d["event_ids"]
    station_ids = d["station_ids"]
    freq_axis   = d.get("freq_axis", None)
    n_classes   = len(np.unique(y))

    print(f"X_psd: {X_psd.shape},  X_td: {X_td.shape},  classes: {n_classes}")
    counts = np.bincount(y)
    for ci, cn in enumerate(CLASS_NAMES[:n_classes]):
        print(f"  {cn}: {counts[ci]} windows")

    # Inverse-frequency class weights (normalised so max = 1)
    inv_freq     = counts.max() / counts.astype(float)
    class_weights = (inv_freq / inv_freq.max()).tolist()
    print(f"  Class weights: { {CLASS_NAMES[i]: round(class_weights[i], 3) for i in range(n_classes)} }")

    # Train / test split (no leakage)
    groups = np.array([f"{e}__{s}" for e, s in zip(event_ids, station_ids)])
    gss    = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                               random_state=RANDOM_SEED)
    tv_idx, test_idx = next(gss.split(X_psd, y, groups))

    Xp_tv, Xt_tv, y_tv, g_tv = (X_psd[tv_idx], X_td[tv_idx],
                                  y[tv_idx], groups[tv_idx])
    Xp_test, Xt_test, y_test  = X_psd[test_idx], X_td[test_idx], y[test_idx]

    # 5-fold CV
    cv_accs = cross_validate(Xp_tv, Xt_tv, y_tv, g_tv,
                              n_classes, class_weights, rng)
    CV_PATH.write_text(json.dumps({
        "fold_accuracies": cv_accs,
        "mean": float(np.mean(cv_accs)),
        "std":  float(np.std(cv_accs)),
    }, indent=2))

    # Fit scaler on full train+val
    psd_mean, psd_std, td_mean, td_std = fit_scalers(Xp_tv, Xt_tv)
    save_scalers(psd_mean, psd_std, td_mean, td_std, freq_axis)
    Xp_tv_s,   Xt_tv_s   = apply_scalers(Xp_tv,   Xt_tv,
                                          psd_mean, psd_std, td_mean, td_std)
    Xp_test_s, Xt_test_s = apply_scalers(Xp_test, Xt_test,
                                          psd_mean, psd_std, td_mean, td_std)

    # Oversample: bring explosion up to ~rocket_bolide count, rocket_bolide up to ~background
    tv_counts   = np.bincount(y_tv)
    target_exp  = int(tv_counts[2] * 0.90)   # ~90% of background count
    target_rock = int(tv_counts[2] * 0.90)
    os_targets  = {0: target_exp, 1: target_rock}
    print(f"\nOversampling targets: explosion→{target_exp}, rocket_bolide→{target_rock}")
    Xp_os, Xt_os, y_os = oversample_minority(Xp_tv_s, Xt_tv_s, y_tv, rng, os_targets)
    print(f"After oversample: {np.bincount(y_os).tolist()}")

    # Augment then Mixup
    Xp_aug, Xt_aug, y_aug = augment_gain_noise(Xp_os, Xt_os, y_os, rng)
    Xp_mix, Xt_mix, y_mix = mixup(Xp_aug, Xt_aug, y_aug, n_classes, rng)
    print(f"After augmentation + mixup: {len(y_mix)} training windows")

    Xp_mix_c   = Xp_mix[..., np.newaxis]
    Xp_test_sc = Xp_test_s[..., np.newaxis]

    # Validation split (15% of augmented set)
    n_val    = max(32, int(len(y_mix) * 0.15))
    val_idxs = rng.choice(len(y_mix), size=n_val, replace=False)
    tr_mask  = np.ones(len(y_mix), dtype=bool)
    tr_mask[val_idxs] = False

    Xp_tr_f, Xt_tr_f, y_tr_f = (Xp_mix_c[tr_mask], Xt_mix[tr_mask],
                                   y_mix[tr_mask])
    y_va_hard = np.argmax(y_mix[val_idxs], axis=1)
    y_va_oh   = keras.utils.to_categorical(y_va_hard, n_classes).astype(np.float32)
    print(f"Final training: {len(y_tr_f)} windows,  validation: {len(y_va_oh)}")

    steps_per_epoch = max(1, len(y_tr_f) // BATCH_SIZE)
    model = build_model(X_psd.shape[1], X_td.shape[1], n_classes)
    model.summary()
    compile_model(model, n_classes, class_weights, steps_per_epoch, MAX_EPOCHS)

    history = train_model(model, Xp_tr_f, Xt_tr_f, y_tr_f,
                          Xp_mix_c[val_idxs], Xt_mix[val_idxs], y_va_oh,
                          verbose=2)

    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    np.savez_compressed(
        "data/processed/test_set.npz",
        X_psd_test=Xp_test_sc,
        X_td_test=Xt_test_s,
        y_test=y_test,
        event_ids_test=event_ids[test_idx],
    )

    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    HISTORY_PATH.write_text(json.dumps(hist_dict, indent=2))

    y_test_oh = keras.utils.to_categorical(y_test, n_classes).astype(np.float32)
    test_loss, test_acc = model.evaluate(
        [Xp_test_sc, Xt_test_s], y_test_oh, verbose=0)
    print(f"\nTest accuracy (float32): {test_acc:.4f}  loss: {test_loss:.4f}")
    print(f"CV mean accuracy:        {np.mean(cv_accs):.4f}"
          f" ± {np.std(cv_accs):.4f}")
    print("\nRun  python src/evaluate.py  for full metrics and confusion matrix.")


if __name__ == "__main__":
    main()
