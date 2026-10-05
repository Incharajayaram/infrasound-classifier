"""
Step 0: Download TAIRED event catalog and show source-type counts.
STOP after printing — do not proceed until the user confirms class definitions.
"""

import json
import requests
import pandas as pd
from pathlib import Path

CATALOG_URL = "https://data.earthscope.org/catalog/infrasound/events.json"
OUT_PATH = Path("data/raw/taired_events.json")
OUT_TSV  = Path("data/raw/taired_events.tsv")

def main():
    print("Downloading TAIRED catalog...")
    r = requests.get(CATALOG_URL, timeout=120,
                     headers={"User-Agent": "infrasound-classifier/0.1 (research)"})
    r.raise_for_status()
    events = r.json()
    print(f"Total records: {len(events)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(events, indent=2))
    print(f"Saved raw JSON to {OUT_PATH}")

    df = pd.DataFrame(events)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Saved TSV to {OUT_TSV}")

    print("\nColumns:", list(df.columns))
    print(f"\nDate range: {df['Time'].min()} → {df['Time'].max()}")

    print("\n" + "=" * 60)
    print("EVENT COUNTS BY SourceType")
    print("=" * 60)
    counts = df["SourceType"].value_counts(dropna=False)
    for stype, n in counts.items():
        print(f"  {str(stype):35s}  {n:5d}")
    print(f"  {'TOTAL':35s}  {len(df):5d}")

    print("\n" + "=" * 60)
    print("METEOROLOGICAL sub-descriptions (first 30)")
    print("=" * 60)
    met = df[df["SourceType"] == "Meteorological"]
    for desc in sorted(met["Description"].unique())[:30]:
        print(f"  {desc}")

    print("\n" + "=" * 60)
    print("SAMPLE ROWS PER SourceType (first 2 each)")
    print("=" * 60)
    for stype in df["SourceType"].unique():
        sub = df[df["SourceType"] == stype].head(2)
        print(f"\n  [{stype}]")
        for _, row in sub.iterrows():
            print(f"    {row['Time']}  {row['Description'][:70]}")

    print("\n" + "=" * 60)
    print("STOP: Review counts and confirm class definitions before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
