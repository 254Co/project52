import requests
import json
import logging
import re

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def normalize_key(key: str) -> str:
    """
    Normalize a key by:
    - Converting to lowercase,
    - Stripping whitespace,
    - Replacing spaces with underscores,
    - Removing non-alphanumeric characters (except underscores).
    """
    key = key.strip().lower().replace(" ", "_")
    return re.sub(r'[^a-z0-9_]', '', key)

def normalize_value(value):
    """
    Normalize a value:
    - Trim strings,
    - Convert dicts/lists to JSON strings,
    - Otherwise pass through.
    """
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    else:
        return value

def flatten_json(data, parent_key: str = '', sep: str = '_') -> dict:
    """
    Recursively flatten nested dicts/lists into a single-level dict,
    normalizing keys and values along the way.
    """
    items = {}
    if isinstance(data, dict):
        for k, v in data.items():
            norm_key = normalize_key(str(k))
            new_key = f"{parent_key}{sep}{norm_key}" if parent_key else norm_key
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        items[parent_key] = normalize_value(data)
    return items

def extract_indicators(data: dict) -> list[dict]:
    """
    Given the top-level JSON, pull out the list of indicator records,
    each tagged with its 'indicator_code'.
    """
    raw = data.get("indicators", {})
    records = []

    if isinstance(raw, dict):
        for code, details in raw.items():
            # skip empty-code records with all-null values
            if code == "" and all(v is None for v in getattr(details, 'values', lambda: [])()):
                continue
            rec = {"indicator_code": code}
            if isinstance(details, dict):
                rec.update(details)
            else:
                rec["details"] = details
            records.append(rec)
    elif isinstance(raw, list):
        records = raw
    else:
        records = [raw]

    return records

def fetch_imf_indicators(url: str = "https://www.imf.org/external/datamapper/api/v1/indicators") -> pd.DataFrame:
    """
    Fetch and return a flattened DataFrame of IMF indicators.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.error(f"Failed to fetch or parse data: {e}")
        return pd.DataFrame()

    raw_records = extract_indicators(data)
    if not raw_records:
        logging.error("No indicator records found in response.")
        return pd.DataFrame()

    flattened = []
    cols = set()
    for rec in raw_records:
        flat = flatten_json(rec)
        flattened.append(flat)
        cols.update(flat.keys())

    # Build DataFrame with consistent column ordering
    columns = sorted(cols)
    df = pd.DataFrame(flattened, columns=columns)
    logging.info(f"Built DataFrame with {len(df)} records and {len(columns)} columns.")
    return df

if __name__ == "__main__":
    df = fetch_imf_indicators()
    if not df.empty:
        #print(df)
        df.to_csv("imf_output.csv")
        # or do whatever downstream processing you need
