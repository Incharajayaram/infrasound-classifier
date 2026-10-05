# Infrasound Edge Classifier

A lightweight 1D-CNN that classifies 60-second infrasound pressure windows into three classes, quantised to int8 for deployment on ESP32-S3 with TFLite Micro.

## Classes

| Class | TAIRED source types | Event count |
|---|---|---|
| `explosion` | Explosion, Mining | 16 |
| `rocket_bolide` | Rocket launch, Rocket re-entry, Bolide | 23 |
| `background` | Scraped quiet TA BDF windows | ~200 scraped |

Thunder was omitted: TAIRED contains only 2 meteorological events (a tornado and a derecho), which are wind-field events rather than thunder — insufficient for a class. A thunder class requires locally collected recordings.

## Run order

```bash
# 0. Inspect TAIRED catalog (already done — shows source-type counts and stops)
python src/inspect_taired.py

# 1. Download waveforms from IRIS FDSN (~200 background + 39 event files)
#    Takes 20-40 min. Add --dry-run to preview without fetching.
python src/download_data.py --bg-count 200

# 2. Preprocess and extract Welch-PSD features
python src/build_dataset.py

# 3. Train 1D-CNN (split by event+station to avoid leakage)
python src/train.py

# 4. Full evaluation: accuracy, macro F1, confusion matrix
python src/evaluate.py

# 5. Export int8 TFLite + C array for TFLite Micro
python src/export_tflite.py
```

All scripts must be run from the project root (`infrasound-classifier/`) with the `.venv` interpreter.

## Data pipeline

```
TAIRED catalog (data.earthscope.org)
        │
        ▼
download_data.py  ──── IRIS FDSN (ObsPy)
        │   downloads miniSEED to data/raw/waveforms/{class}/
        ▼
build_dataset.py
        │   detrend → resample 40 Hz → remove response (→ Pa)
        │   bandpass 0.01–20 Hz (Butterworth order 4, zero-phase)
        │   window 60 s / 50% overlap
        │   Welch PSD (nperseg=256, noverlap=128) → log10
        │   save data/processed/dataset.npz
        ▼
train.py
        │   split BY EVENT+STATION (GroupShuffleSplit, no leakage)
        │   fit scaler on training windows only → data/processed/scaler.json
        │   augmentation (training only): gain offset, additive noise
        │   1D-CNN with BatchNorm, GlobalAveragePooling1D, class-weighted loss
        │   save results/model_float32.keras
        ▼
evaluate.py  →  results/metrics.json, results/confusion_matrix.png
        ▼
export_tflite.py  →  results/model.tflite, results/model_data.cc
```

## Model architecture

Input: `(129, 1)` — 129 log10-PSD bins (0–20 Hz at 40 Hz sample rate)

```
Conv1D(16, 5, ReLU) → BatchNorm
Conv1D(16, 5, ReLU) → BatchNorm
Conv1D( 8, 3, ReLU) → BatchNorm
GlobalAveragePooling1D
Dense(16, ReLU)
Dense(3, softmax)
```

## Firmware integration

The firmware must reproduce the preprocessing chain exactly:
1. Sample at ≥50 Hz, decimate to **40 Hz**
2. Detrend (remove linear trend from the 60-second window)
3. Divide by sensitivity (counts → Pa) — sensitivity is in `data/processed/scaler.json` per station; for a fixed ELVH sensor use the measured sensitivity constant
4. Bandpass 0.01–20 Hz (implement as IIR on device, or apply offline)
5. Compute Welch PSD with `nperseg=256, noverlap=128` → log10
6. Subtract `scaler.json["mean"]`, divide by `scaler.json["std"]` (element-wise, per frequency bin)
7. Run int8 inference with `results/model_data.cc` via TFLite Micro

## Known limitations

1. **Domain shift — sensor mismatch**: TA broadband sensors have a different frequency response and noise floor than an ELVH-based microbarometer. The accuracy figures from this dataset are a **pretrained baseline only**. The model must be re-validated and fine-tuned on locally collected recordings from your own sensor before field deployment.

2. **Small, imbalanced event set**: Only 16 explosion-class and 23 rocket/bolide-class events from TAIRED. Expect high uncertainty in per-class metrics, especially for the explosion class. Report per-class counts alongside any accuracy figure.

3. **Geographic bias**: TAIRED events are concentrated in North America (2011–2015 USArray TA footprint). The rocket/bolide class is heavily weighted toward Kennedy Space Center launches. Performance on events from other geographies or source types not in training is unknown.

4. **Thunder not present**: No thunder class. If your deployment area has frequent lightning, add a thunder class from local recordings.

## Dependencies

```
obspy>=1.5
tensorflow>=2.22
scikit-learn>=1.9
scipy>=1.18
numpy>=2.0
pandas>=2.0
matplotlib>=3.11
```

Install: `python -m venv .venv && .venv/bin/pip install obspy tensorflow scikit-learn scipy matplotlib pandas`

## Citation

TAIRED dataset:
> EarthScope DS (2012). TA Infrasound Reference Event Database (TAIRED). https://doi.org/10.17611/DP/IS.1
