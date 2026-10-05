"""
Download waveforms from IRIS FDSN for TAIRED labeled events and background windows.

Classes (decided after inspecting TAIRED source-type counts):
  explosion    = SourceType in {Explosion, Mining}                    → 16 events
  rocket_bolide = SourceType in {Rocket launch, Rocket re-entry, Bolide} → 23 events
  background   = quiet TA BDF windows, no catalogued event ±2 h       → scraped

Usage:
    python src/download_data.py [--dry-run] [--bg-count 200]
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException

# ── Config ──────────────────────────────────────────────────────────────────
CATALOG_PATH  = Path("data/raw/taired_events.json")
WAVEFORM_DIR  = Path("data/raw/waveforms")
STATION_CACHE = Path("data/raw/ta_bdf_stations.json")
SAMPLE_RATE   = 40.0          # Hz — TA BDF target rate
PRE_EVENT     = 60.0          # s before event (provides pre-event context windows)
POST_EVENT    = 150.0         # s after event (capture full coda at 15 s window size)
BG_DURATION   = 210.0         # s of background per window (matches event window total length)
SEARCH_RADIUS_DEG = 10.0      # ~1100 km — max distance from event
MAX_STATIONS  = 15            # stations per event — more windows per labeled event
BG_BLACKOUT_H = 2.0           # hours to exclude around any TAIRED event
BG_START      = "2011-01-01"
BG_END        = "2015-12-31"
RANDOM_SEED   = 42

CLASS_MAP = {
    "Explosion":     "explosion",
    "Mining":        "explosion",
    "Rocket launch": "rocket_bolide",
    "Rocket re-entry": "rocket_bolide",
    "Bolide":        "rocket_bolide",
}


def load_catalog() -> list[dict]:
    events = json.loads(CATALOG_PATH.read_text())
    print(f"Loaded {len(events)} total TAIRED events")
    return events


def labeled_events(catalog: list[dict]) -> list[dict]:
    labeled = [e for e in catalog if e["SourceType"] in CLASS_MAP]
    for e in labeled:
        e["label"] = CLASS_MAP[e["SourceType"]]
    by_label = {}
    for e in labeled:
        by_label.setdefault(e["label"], 0)
        by_label[e["label"]] += 1
    print(f"Labeled events: {by_label}")
    return labeled


def all_event_times(catalog: list[dict]) -> list[UTCDateTime]:
    return [UTCDateTime(e["Time"]) for e in catalog]


def get_or_cache_stations(client: Client) -> list[dict]:
    if STATION_CACHE.exists():
        return json.loads(STATION_CACHE.read_text())

    print("Querying all TA BDF stations (may take a minute)...")
    inv = client.get_stations(
        network="TA",
        channel="BDF",
        level="channel",
        starttime=UTCDateTime(BG_START),
        endtime=UTCDateTime(BG_END),
    )
    stations = []
    for net in inv:
        for sta in net:
            for cha in sta:
                stations.append({
                    "network": net.code,
                    "station": sta.code,
                    "location": cha.location_code,
                    "channel": cha.code,
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "start": str(cha.start_date),
                    "end": str(cha.end_date) if cha.end_date else None,
                })
    STATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    STATION_CACHE.write_text(json.dumps(stations, indent=2))
    print(f"Cached {len(stations)} TA BDF channels")
    return stations


def stations_active_at(stations: list[dict], t: UTCDateTime) -> list[dict]:
    active = []
    for s in stations:
        start = UTCDateTime(s["start"])
        end   = UTCDateTime(s["end"]) if s["end"] else UTCDateTime(BG_END) + 86400
        if start <= t <= end:
            active.append(s)
    return active


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_stations(stations, lat, lon, n=MAX_STATIONS, max_radius_km=1200):
    with_dist = []
    for s in stations:
        d = haversine_km(lat, lon, s["latitude"], s["longitude"])
        if d <= max_radius_km:
            with_dist.append((d, s))
    with_dist.sort(key=lambda x: x[0])
    return [s for _, s in with_dist[:n]]


def download_waveform(client, s, t_start, t_end, out_path, dry_run=False):
    if out_path.exists():
        return "cached"
    if dry_run:
        print(f"    [DRY] {s['network']}.{s['station']}.{s['channel']}  {t_start}")
        return "dry"
    try:
        st = client.get_waveforms(
            network=s["network"], station=s["station"],
            location=s["location"], channel=s["channel"],
            starttime=t_start, endtime=t_end,
        )
        if len(st) == 0:
            return "empty"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        st.write(str(out_path), format="MSEED")
        return "ok"
    except FDSNNoDataException:
        return "nodata"
    except FDSNException as e:
        return f"err:{e}"
    except Exception as e:
        return f"err:{e}"


def download_events(client, labeled, stations, dry_run=False):
    ok, skip, fail = 0, 0, 0
    for ev in labeled:
        t = UTCDateTime(ev["Time"])
        t_start = t - PRE_EVENT
        t_end   = t + POST_EVENT
        ev_stations = nearest_stations(
            stations_active_at(stations, t),
            ev["Latitude"], ev["Longitude"],
        )
        if not ev_stations:
            print(f"  [WARN] No TA BDF stations near {ev['Description'][:50]}")
            continue
        ev_id = ev["id"]
        for s in ev_stations:
            out_path = WAVEFORM_DIR / ev["label"] / f"{ev_id}_{s['station']}.mseed"
            status = download_waveform(client, s, t_start, t_end, out_path, dry_run)
            if status in ("ok", "cached", "dry"):
                ok += 1
            elif status == "nodata":
                skip += 1
            else:
                fail += 1
                print(f"    {s['station']}: {status}")
        time.sleep(0.1)  # polite rate limiting
    print(f"Events: ok={ok}, skip(nodata)={skip}, fail={fail}")


def is_near_any_event(t: UTCDateTime, all_times, blackout_s: float) -> bool:
    for et in all_times:
        if abs(t - et) < blackout_s:
            return True
    return False


def download_background(client, all_times, stations, n_windows: int, dry_run=False):
    rng = random.Random(RANDOM_SEED)
    blackout_s = BG_BLACKOUT_H * 3600
    start_ts = UTCDateTime(BG_START).timestamp
    end_ts   = UTCDateTime(BG_END).timestamp

    ok, skip, fail, attempts = 0, 0, 0, 0
    while ok < n_windows and attempts < n_windows * 20:
        attempts += 1
        # Pick random time
        ts = rng.uniform(start_ts, end_ts - BG_DURATION)
        t = UTCDateTime(ts)
        if is_near_any_event(t, all_times, blackout_s):
            skip += 1
            continue
        # Pick a random active station
        active = stations_active_at(stations, t)
        if not active:
            continue
        s = rng.choice(active)
        t_end = t + BG_DURATION
        out_path = WAVEFORM_DIR / "background" / f"bg_{int(ts)}_{s['station']}.mseed"
        status = download_waveform(client, s, t, t_end, out_path, dry_run)
        if status in ("ok", "cached", "dry"):
            ok += 1
            if ok % 20 == 0:
                print(f"  background: {ok}/{n_windows}")
        elif status == "nodata":
            pass
        else:
            fail += 1
        time.sleep(0.1)
    print(f"Background: ok={ok}, attempts={attempts}, fail={fail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without fetching")
    parser.add_argument("--bg-count", type=int, default=500, help="Background windows to download")
    args = parser.parse_args()

    WAVEFORM_DIR.mkdir(parents=True, exist_ok=True)
    client = Client("IRIS")

    catalog  = load_catalog()
    labeled  = labeled_events(catalog)
    all_times = all_event_times(catalog)

    print("\nFetching TA BDF station inventory...")
    stations = get_or_cache_stations(client)
    print(f"Total TA BDF channels: {len(stations)}")

    print("\n--- Downloading labeled event waveforms ---")
    download_events(client, labeled, stations, dry_run=args.dry_run)

    print(f"\n--- Downloading {args.bg_count} background windows ---")
    download_background(client, all_times, stations, args.bg_count, dry_run=args.dry_run)

    print("\nDone. Check data/raw/waveforms/")


if __name__ == "__main__":
    main()
