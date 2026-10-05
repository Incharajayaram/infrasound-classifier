"""
Step 0: Download TAIRED event table and inspect source types.
Stops after printing counts — do not proceed until confirmed.
"""

import requests
import pandas as pd
import io
import sys

SPUD_URL = "https://ds.iris.edu/spud/infrasoundevent"

# SPUD's text export endpoint — returns CSV/tab-separated event list
TEXT_EXPORT_URL = "https://ds.iris.edu/spud/infrasoundevent?format=text"

# Try alternate export formats SPUD supports
EXPORT_URLS = [
    ("text/plain", "https://ds.iris.edu/spud/infrasoundevent?format=text"),
    ("CSV",        "https://ds.iris.edu/spud/infrasoundevent?output=text"),
]

def fetch_event_table() -> pd.DataFrame:
    headers = {"User-Agent": "infrasound-classifier/0.1 (research)"}

    for label, url in EXPORT_URLS:
        print(f"Trying {label}: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            content = r.text
            print(f"  Status {r.status_code}, content length {len(content)} bytes")
            print(f"  First 500 chars:\n{content[:500]}\n")

            # Try to parse as delimited text
            for sep in ["\t", "|", ","]:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep, comment="#",
                                     skipinitialspace=True, on_bad_lines="skip")
                    if len(df.columns) >= 3:
                        print(f"  Parsed with sep={repr(sep)}: {df.shape[0]} rows, {df.shape[1]} cols")
                        print(f"  Columns: {list(df.columns)}")
                        return df
                except Exception as e:
                    continue
        except requests.RequestException as e:
            print(f"  Failed: {e}")

    return None


def main():
    print("=" * 60)
    print("TAIRED event table inspection")
    print("=" * 60)

    df = fetch_event_table()

    if df is None or df.empty:
        print("\nCould not parse event table from SPUD web service.")
        print("Falling back: trying SPUD HTML page to find download link...")
        r = requests.get(SPUD_URL, timeout=30,
                         headers={"User-Agent": "infrasound-classifier/0.1"})
        print(f"HTML page status: {r.status_code}")
        # Print first 3000 chars to find the right download endpoint
        print(r.text[:3000])
        sys.exit(1)

    # Find source-type column (varies by SPUD version)
    src_col = None
    for candidate in ["Source Type", "source_type", "SourceType", "type", "Type",
                       "Event Type", "event_type", "Category"]:
        if candidate in df.columns:
            src_col = candidate
            break
    if src_col is None:
        # Show all columns and first rows so we can identify manually
        print("\nAll columns:", list(df.columns))
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nCould not identify source-type column automatically.")
        sys.exit(1)

    print(f"\nSource-type column: '{src_col}'")
    print("\nEvent counts per source type:")
    counts = df[src_col].value_counts(dropna=False)
    print(counts.to_string())
    print(f"\nTotal events: {len(df)}")

    # Save raw table
    out = "/home/achinthyaj/infrasound-classifier/data/raw/taired_events.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"\nSaved raw table to {out}")

    print("\n" + "=" * 60)
    print("STOP: Review counts above before proceeding to build_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
