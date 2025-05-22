import requests
import pandas as pd
from typing import List, Optional

def fetch_population_projections(
    api_key: str,
    vintage: int = 2017,
    endpoint: str = "pop",
    variables: List[str] = None,
    for_clause: str = "us:1",
    in_clause: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch population projections from the U.S. Census API.

    Parameters
    ----------
    api_key : str
        Your Census API key.
    vintage : int
        Vintage year of the projections (2017, 2014, or 2012).
    endpoint : str
        Projection dataset (e.g. 'pop', 'agegroups', 'nat', 'births', 'deaths', 'nim').
    variables : List[str], optional
        List of variables to retrieve (default ['POP','YEAR']).
    for_clause : str
        Geography filter in “for” syntax (e.g. 'us:1', 'state:06').
    in_clause : str, optional
        Optional “in” clause for nested geographies.

    Returns
    -------
    pd.DataFrame
        Columns = requested variables + geography; rows = projections.
    """
    if variables is None:
        variables = ["POP", "YEAR"]

    base_url = f"https://api.census.gov/data/{vintage}/popproj/{endpoint}"
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
    df = pd.DataFrame(data[1:], columns=data[0])
    return df


if __name__ == "__main__":
    df_proj = fetch_population_projections(
        api_key="7db8a4234d704d7475dfd0d7ab4c12f5530092fd",
        vintage=2017,
        endpoint="pop",
        variables=["POP", "YEAR"],
        for_clause="us:1"
    )
    
    print(df_proj)
