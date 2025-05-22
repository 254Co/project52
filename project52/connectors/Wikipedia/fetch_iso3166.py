import requests
import pandas as pd
import logging
from io import StringIO

_WIKI_URL = "https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2#Officially_assigned_code_elements"

def fetch_iso3166_codes() -> pd.DataFrame:
    """
    Fetch the latest ISO 3166-1 alpha-2 codes table from Wikipedia
    and return it as a pandas DataFrame.
    """
    logging.info(f"Fetching ISO 3166-1 alpha-2 codes table from {_WIKI_URL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    resp = session.get(_WIKI_URL)
    resp.raise_for_status()

    # Wrap the HTML in StringIO to avoid the FutureWarning
    html_buf = StringIO(resp.text)
    tables = pd.read_html(html_buf)

    if not tables:
        raise RuntimeError("No tables found on the Wikipedia page")

    df = tables[4].copy()
    df = df.drop(columns=['Year', 'ccTLD', 'Notes'])
    
    # rename multiple columns
    df.rename(columns={
        "Code": "code",
        "Country name (using title case)": "name"
    }, inplace=True)

    logging.info(f"Retrieved {len(df)} ISO 3166-1 alpha-2 codes")
    return df


if __name__ == "__main__":
    df = fetch_iso3166_codes()
    print(df)


"""
    OUTPUT:

            code                  name
        0     AD               Andorra
        1     AE  United Arab Emirates
        2     AF           Afghanistan
        3     AG   Antigua and Barbuda
        4     AI              Anguilla
        ..   ...                   ...
        244   YE                 Yemen
        245   YT               Mayotte
        246   ZA          South Africa
        247   ZM                Zambia
        248   ZW              Zimbabwe

        [249 rows x 2 columns]

"""