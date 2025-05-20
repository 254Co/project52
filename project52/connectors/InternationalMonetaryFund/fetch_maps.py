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

def _fetch_and_flatten(url: str, root_key: str, code_field: str) -> pd.DataFrame:
    """
    Generic helper to fetch a top-level map under `root_key`, tag each entry
    with `code_field`, flatten and collect into a DataFrame.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.error(f"Failed to fetch or parse data from {url!r}: {e}")
        return pd.DataFrame()

    raw = data.get(root_key, {})
    if not raw:
        logging.error(f"No records found under key '{root_key}'.")
        return pd.DataFrame()

    records = []
    if isinstance(raw, dict):
        for code, details in raw.items():
            rec = {code_field: code}
            if isinstance(details, dict):
                rec.update(details)
            else:
                rec["details"] = details
            records.append(rec)
    elif isinstance(raw, list):
        records = raw
    else:
        records = [raw]

    flattened = []
    cols = set()
    for rec in records:
        flat = flatten_json(rec)
        flattened.append(flat)
        cols.update(flat.keys())

    df = pd.DataFrame(flattened, columns=sorted(cols))
    if "description" in df.columns:
        df = df.drop(columns="description")
    return df

def fetch_imf_indicators_map(
    url: str = "https://www.imf.org/external/datamapper/api/v1/indicators"
) -> pd.DataFrame:
    """
    Fetch and return a flattened DataFrame of IMF indicators.
    """
    raw_records = extract_indicators(requests.get(url).json())
    if not raw_records:
        logging.error("No indicator records found in response.")
        return pd.DataFrame()

    flattened, cols = [], set()
    for rec in raw_records:
        flat = flatten_json(rec)
        flattened.append(flat)
        cols.update(flat.keys())

    df = pd.DataFrame(flattened, columns=sorted(cols))
    if "description" in df.columns:
        df = df.drop(columns="description")
    return df

def fetch_imf_countries_map(
    url: str = "https://www.imf.org/external/datamapper/api/v1/countries"
) -> pd.DataFrame:
    """
    Fetch and return a flattened DataFrame of IMF countries.
    """
    return _fetch_and_flatten(url, root_key="countries", code_field="country_code")

def fetch_imf_regions_map(
    url: str = "https://www.imf.org/external/datamapper/api/v1/regions"
) -> pd.DataFrame:
    """
    Fetch and return a flattened DataFrame of IMF regions.
    """
    return _fetch_and_flatten(url, root_key="regions", code_field="region_code")

def fetch_imf_groups_map(
    url: str = "https://www.imf.org/external/datamapper/api/v1/groups"
) -> pd.DataFrame:
    """
    Fetch and return a flattened DataFrame of IMF groups.
    """
    return _fetch_and_flatten(url, root_key="groups", code_field="group_code")

if __name__ == "__main__":
    for fn in (
        fetch_imf_indicators_map,
        fetch_imf_countries_map,
        fetch_imf_regions_map,
        fetch_imf_groups_map,
    ):
        df = fn()
        logging.info(f"{fn.__name__} -> {len(df)} rows, {len(df.columns)} cols")
        print(df.head())
