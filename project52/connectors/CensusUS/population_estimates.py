import requests
import pandas as pd
from typing import List, Optional

def fetch_population_estimates(
    api_key: str,
    vintage: int = 2023,
    variables: List[str] = None,
    for_clause: str = "us:1",
    in_clause: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch annual population estimates from the U.S. Census API.

    Parameters
    ----------
    api_key : str
        Your Census API key.
    vintage : int
        Vintage year of the estimates (e.g. 2023).
    variables : List[str], optional
        List of variables to retrieve (default ['NAME', f'POP_{vintage}']).
    for_clause : str
        Geography filter in “for” syntax (e.g. 'us:1', 'state:06', 'county:*').
    in_clause : str, optional
        Optional “in” clause for nested geographies (e.g. 'state:06').

    Returns
    -------
    pd.DataFrame
        Columns = requested variables + geography; rows = observations.
    """
    if variables is None:
        variables = ["NAME", f"POP_{vintage}"]

    base_url = f"https://api.census.gov/data/{vintage}/pep/population"
    params = {
        "get": ",".join(variables),
        "for": for_clause,
        "key": api_key
    }
    if in_clause:
        params["in"] = in_clause

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json()
    # first row are column names
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


if __name__ == "__main__":
    df_est = fetch_population_estimates(
        api_key="7db8a4234d704d7475dfd0d7ab4c12f5530092fd",
        vintage=2023,
        variables=["NAME", "POP_2023"],
        for_clause=""
    )
