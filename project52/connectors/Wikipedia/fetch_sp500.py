import requests
import pandas as pd
import logging
from io import StringIO

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500_companies() -> pd.DataFrame:
    """
    Fetch the latest S&P 500 constituents table from Wikipedia
    and return it as a pandas DataFrame, with CIK as a zero-padded
    10-digit string.
    """
    logging.info(f"Fetching S&P 500 list from {_WIKI_URL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    resp = session.get(_WIKI_URL)
    resp.raise_for_status()

    # Wrap the HTML in StringIO to avoid the FutureWarning
    html_buf = StringIO(resp.text)
    tables = pd.read_html(html_buf)
    if not tables:
        raise RuntimeError("No tables found on the Wikipedia page")

    df = tables[0].copy()
    # clean up column names
    df.columns = [c.strip().replace("\n", " ") for c in df.columns]

    # ensure CIK is a 10-digit string with leading zeros
    if "CIK" in df.columns:
        df["CIK"] = df["CIK"].astype(str).str.zfill(10)

    df = df.drop(columns=['Security', 'GICS Sub-Industry', 'GICS Sector', 'Headquarters Location', 'Founded'])
    
    order = ["CIK", "Symbol", "Date added"]
    df = df[order]
    logging.info(f"Retrieved {len(df)} S&P 500 constituents")
    return df


if __name__ == "__main__":
    df = fetch_sp500_companies()
    print(df)