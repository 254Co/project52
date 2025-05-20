import requests
import pandas as pd

def fetch_ticker_cik_map() -> pd.DataFrame:
    """
    Fetches all ticker symbols and their CIKs from the SEC’s company_tickers.json
    directory and returns them as a DataFrame.

    Parameters
    ----------
    user_agent : str, optional
        A descriptive User-Agent (e.g. "Your Name your_email@example.com").
        SEC requires one to avoid request throttling. If None, a generic
        agent is used (but you should supply your own).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['ticker', 'cik'], where 'cik' is a zero-padded
        10-digit string.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "Alice Example alice@example.com" or "python-requests/2.x",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    records = [
        {
            "ticker": entry["ticker"],
            "cik": str(entry["cik_str"]).zfill(10),
            "name": entry["title"]
        }
        for entry in data.values()
    ]

    df = pd.DataFrame.from_records(records)

    return df 

# Example usage:
if __name__ == "__main__":
    df = fetch_ticker_cik_map()
    print(df)
