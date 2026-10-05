"""
Build feature dataset from downloaded miniSEED waveforms.

Processing chain per waveform:
  1. Merge gaps, resample to 40 Hz
  2. Detrend (linear)
  3. Remove instrument response → Pa  (uses IRIS FDSN for station metadata)
  4. Bandpass 0.01–18 Hz (Butterworth, zero-phase, order 4)
  5. Window into WIN_LEN_S seconds with OVERLAP fraction (stride = WIN_LEN_S × (1-OVERLAP))
  6. Per window — two feature blocks:
       PSD block  : Welch (nperseg=256, noverlap=128) → log10  [129 bins]
       Impulse block: log10_crest, excess_kurtosis, skewness, log10_rms, spectral_centroid,
                      spectral_flatness, band_frac_low, band_frac_mid, band_frac_high, log10_zcr  [10 values]

Rationale for impulse block:
  Explosions are impulsive; kurtosis and crest factor are high (~10-100x background).
  These 4 features give the CNN a direct "spikiness" signal that 60-second PSD windows
  would average away.

Output:
  data/processed/dataset.npz  — X_psd (N,129)  X_td (N,10)  y  event_ids  station_ids
  data/processed/class_map.json

Usage:
    python src/build_dataset.py [--skip-response]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from obspy import read as obs_read, UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import welch
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

# ── Config ───────────────────────────────────────────────────────────────────
WAVEFORM_DIR = Path("data/raw/waveforms")
OUT_DIR      = Path("data/processed")
TARGET_RATE  = 40.0       # Hz
WIN_LEN_S    = 15.0       # seconds — short enough to capture impulsive events cleanly
OVERLAP      = 0.80       # stride = 15 × 0.2 = 3 s
NPERSEG      = 256
NOVERLAP     = 128
BP_LOW       = 0.01       # Hz
BP_HIGH      = 18.0       # Hz  (below Nyquist of 20 Hz)
BP_ORDER     = 4

CLASS_MAP = {
    "explosion":     0,
    "rocket_bolide": 1,
    "background":    2,
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

STATION_META_CACHE = Path("data/raw/station_meta_cache.json")


def get_fdsn_client():
    return Client("IRIS")


def load_station_meta_cache() -> dict:
    if STATION_META_CACHE.exists():
        return json.loads(STATION_META_CACHE.read_text())
    return {}


def save_station_meta_cache(cache: dict):
    STATION_META_CACHE.write_text(json.dumps(cache, indent=2))


def fetch_sensitivity(client, net, sta, loc, cha, t: UTCDateTime, cache: dict):
    key = f"{net}.{sta}.{loc}.{cha}"
    if key in cache:
        return cache[key]
    try:
        inv = client.get_stations(
            network=net, station=sta, location=loc, channel=cha,
            starttime=t - 60, endtime=t + 60,
            level="response",
        )
        sensitivity = inv[0][0][0].response.instrument_sensitivity.value
        cache[key] = sensitivity
        return sensitivity
    except Exception as e:
        warnings.warn(f"Could not get sensitivity for {key}: {e}")
        return None


def process_stream(st, skip_response=False, client=None, meta_cache=None):
    if len(st) == 0:
        return None
    st = st.copy()
    st.merge(fill_value=0)
    tr = st[0]

    if abs(tr.stats.sampling_rate - TARGET_RATE) > 0.5:
        tr.resample(TARGET_RATE)

    tr.detrend("linear")

    if not skip_response and client is not None and meta_cache is not None:
        sensitivity = fetch_sensitivity(
            client,
            tr.stats.network, tr.stats.station,
            tr.stats.location, tr.stats.channel,
            tr.stats.starttime, meta_cache,
        )
        if sensitivity:
            tr.data = tr.data.astype(np.float64) / sensitivity
        else:
            tr.data = tr.data.astype(np.float64)
    else:
        tr.data = tr.data.astype(np.float64)

    tr.filter("bandpass", freqmin=BP_LOW, freqmax=BP_HIGH,
              corners=BP_ORDER, zerophase=True)

    return tr.data, float(tr.stats.sampling_rate)


def impulsive_features(seg: np.ndarray, fs: float = 40.0) -> np.ndarray:
    """
    10 time-domain + spectral features.

    Classic 4  : log_crest, excess_kurtosis, skewness, log_rms
    New 6      : spectral_centroid_hz, spectral_flatness,
                 band_frac_low (<1 Hz), band_frac_mid (1-5 Hz),
                 band_frac_high (5-20 Hz), log_zcr
    """
    rms  = np.sqrt(np.mean(seg ** 2)) + 1e-12
    peak = np.max(np.abs(seg))        + 1e-12
    crest = peak / rms

    std = seg.std()
    if std < 1e-10:
        kurt, skewness = 0.0, 0.0
    else:
        kurt     = float(scipy_kurtosis(seg))
        skewness = float(scipy_skew(seg))
    log_rms = np.log10(rms)

    # Welch PSD for spectral features (same params as the main PSD)
    f, pxx = welch(seg, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    pxx_sum = pxx.sum() + 1e-30

    # Spectral centroid — frequency centre-of-mass
    centroid = float(np.dot(f, pxx) / pxx_sum)

    # Spectral flatness — geometric / arithmetic mean of PSD (0=tonal, 1=noise)
    log_pxx  = np.log(pxx + 1e-30)
    flatness = float(np.exp(log_pxx.mean()) / (pxx.mean() + 1e-30))
    flatness = float(np.clip(flatness, 0.0, 1.0))

    # Band energy fractions (discriminate wind <1 Hz vs event broadband)
    low_mask  = f <  1.0
    mid_mask  = (f >= 1.0) & (f < 5.0)
    high_mask = f >= 5.0
    frac_low  = float(pxx[low_mask].sum()  / pxx_sum)
    frac_mid  = float(pxx[mid_mask].sum()  / pxx_sum)
    frac_high = float(pxx[high_mask].sum() / pxx_sum)

    # Zero-crossing rate (scale-invariant)
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0)
    log_zcr = float(np.log10(zcr + 1e-6))

    return np.array([
        np.log10(max(crest, 1.0)),  # 0 log crest factor
        kurt,                        # 1 excess kurtosis
        skewness,                    # 2 skewness
        log_rms,                     # 3 log RMS (Pa)
        centroid,                    # 4 spectral centroid (Hz)
        flatness,                    # 5 spectral flatness
        frac_low,                    # 6 band energy <1 Hz
        frac_mid,                    # 7 band energy 1–5 Hz
        frac_high,                   # 8 band energy 5–20 Hz
        log_zcr,                     # 9 log zero-crossing rate
    ], dtype=np.float32)


def extract_features(data: np.ndarray, fs: float):
    """
    Returns (X_psd, X_td, freqs) where:
      X_psd : (N_windows, 129)
      X_td  : (N_windows, 10)
      freqs : frequency axis Hz
    """
    win_samples = int(WIN_LEN_S * fs)
    step        = max(1, int(win_samples * (1 - OVERLAP)))

    if len(data) < win_samples:
        empty_psd = np.empty((0, NPERSEG // 2 + 1), dtype=np.float32)
        empty_td  = np.empty((0, 10), dtype=np.float32)
        return empty_psd, empty_td, None

    psd_rows, td_rows = [], []
    i = 0
    while i + win_samples <= len(data):
        seg = data[i : i + win_samples]
        f, pxx = welch(seg, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP)
        psd_rows.append(np.log10(pxx + 1e-30).astype(np.float32))
        td_rows.append(impulsive_features(seg, fs=fs))
        i += step

    return np.array(psd_rows), np.array(td_rows), f


def parse_meta_from_path(path: Path) -> dict:
    label    = path.parent.name
    fname    = path.stem
    parts    = fname.split("_")
    if label == "background":
        station_id = parts[-1]
        event_id   = "bg_" + (parts[1] if len(parts) >= 3 else "unknown")
    else:
        station_id = parts[-1]
        event_id   = parts[0] if len(parts) >= 2 else "unknown"
    return {"label": label, "event_id": event_id, "station_id": station_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-response", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client     = None if args.skip_response else get_fdsn_client()
    meta_cache = load_station_meta_cache() if not args.skip_response else {}

    all_psd, all_td, all_y, all_meta = [], [], [], []
    freq_axis = None
    label_counts = {k: 0 for k in CLASS_MAP}
    errors = 0

    mseed_files = sorted(WAVEFORM_DIR.rglob("*.mseed"))
    print(f"Found {len(mseed_files)} miniSEED files")

    for i, fpath in enumerate(mseed_files):
        file_meta = parse_meta_from_path(fpath)
        label_str = file_meta["label"]
        if label_str not in CLASS_MAP:
            continue

        try:
            st = obs_read(str(fpath))
        except Exception as e:
            print(f"  [ERR] read {fpath.name}: {e}")
            errors += 1
            continue

        result = process_stream(st, skip_response=args.skip_response,
                                client=client, meta_cache=meta_cache)
        if result is None:
            errors += 1
            continue

        data_pa, fs = result
        X_psd, X_td, freqs = extract_features(data_pa, fs)
        if X_psd.shape[0] == 0:
            continue

        if freq_axis is None:
            freq_axis = freqs

        y_val = CLASS_MAP[label_str]
        all_psd.append(X_psd)
        all_td.append(X_td)
        all_y.extend([y_val] * len(X_psd))
        for _ in range(len(X_psd)):
            all_meta.append(file_meta)
        label_counts[label_str] += len(X_psd)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(mseed_files)} files  "
                  f"({sum(label_counts.values())} windows so far)...")

    if not args.skip_response:
        save_station_meta_cache(meta_cache)

    if not all_psd:
        print("ERROR: No features extracted.")
        return

    X_psd = np.concatenate(all_psd, axis=0)
    X_td  = np.concatenate(all_td,  axis=0)
    y     = np.array(all_y, dtype=np.int32)

    print(f"\nTotal windows   : {len(y)}")
    print(f"PSD feature bins: {X_psd.shape[1]}")
    print(f"TD features     : {X_td.shape[1]}  [log_crest, kurtosis, skewness, log_rms, centroid, flatness, frac_low, frac_mid, frac_high, log_zcr]")
    print("Windows per class:")
    for label, cnt in label_counts.items():
        print(f"  {label:20s}: {cnt}")
    print(f"Errors          : {errors}")

    out_path = OUT_DIR / "dataset.npz"
    np.savez_compressed(
        out_path,
        X_psd=X_psd,
        X_td=X_td,
        y=y,
        event_ids=np.array([m["event_id"]   for m in all_meta]),
        station_ids=np.array([m["station_id"] for m in all_meta]),
        label_strs=np.array([m["label"]      for m in all_meta]),
        freq_axis=freq_axis if freq_axis is not None else np.array([]),
    )
    print(f"\nSaved dataset to {out_path}")

    (OUT_DIR / "class_map.json").write_text(json.dumps(CLASS_MAP, indent=2))
    print("Saved class_map.json")

    print("\nNext step: python src/train.py")


if __name__ == "__main__":
    main()
