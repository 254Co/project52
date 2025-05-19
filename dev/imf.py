import requests
import csv
import json
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def normalize_key(key):
    """
    Normalize a key by:
    - Converting to lowercase,
    - Stripping whitespace,
    - Replacing spaces with underscores,
    - Removing non-alphanumeric characters (except underscores).
    """
    key = key.strip().lower().replace(" ", "_")
    key = re.sub(r'[^a-z0-9_]', '', key)
    return key

def normalize_value(value):
    """
    Normalize a value:
    - Trim strings,
    - Convert dictionaries or lists to JSON strings,
    - Otherwise, return the value unchanged.
    """
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    else:
        return value

def flatten_json(data, parent_key='', sep='_'):
    """
    Recursively flattens nested dictionaries and lists while normalizing keys and values.
    """
    items = {}
    if isinstance(data, dict):
        for k, v in data.items():
            norm_key = normalize_key(str(k))
            new_key = f"{parent_key}{sep}{norm_key}" if parent_key else norm_key
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(item, new_key, sep=sep))
    else:
        items[parent_key] = normalize_value(data)
    return items

def extract_indicators(data):
    """
    Extract indicators from the API data.
    The JSON structure is:
      { "indicators": { indicator_code: { ... }, ... }, "api": { ... } }
    This function returns a list of records with an added field 'indicator_code'.
    """
    indicators = data.get("indicators", {})
    records = []
    
    if isinstance(indicators, dict):
        for code, details in indicators.items():
            # Skip if code is an empty string and details are all null
            if code == "" and all(v is None for v in details.values()):
                continue
            record = {"indicator_code": code}
            if isinstance(details, dict):
                record.update(details)
            else:
                record["details"] = details
            records.append(record)
    elif isinstance(indicators, list):
        # Fallback if it's a list, just use it as is.
        records = indicators
    else:
        # If it's a single record, wrap it in a list.
        records = [indicators]
    
    return records

def main():
    url = "https://www.imf.org/external/datamapper/api/v1/indicators"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return

    logging.debug("Fetched JSON data:")
    logging.debug(json.dumps(data, indent=2))

    # Extract the list of indicator records.
    raw_records = extract_indicators(data)
    if not raw_records:
        logging.error("No indicators data found!")
        return

    # Flatten and normalize each record.
    flattened_data = []
    all_keys = set()
    for record in raw_records:
        flat_record = flatten_json(record)
        flattened_data.append(flat_record)
        all_keys.update(flat_record.keys())

    # Sort the columns for consistent ordering.
    columns = sorted(list(all_keys))

    output_filename = 'imf_indicators.csv'
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for rec in flattened_data:
                writer.writerow(rec)
        logging.info(f"CSV file '{output_filename}' created successfully.")
    except Exception as e:
        logging.error(f"Error writing CSV: {e}")

if __name__ == '__main__':
    main()
