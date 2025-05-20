import requests
import orjson
import pandas as pd
from typing import Optional, Tuple


def fetch_polygon_news_page(
    api_key: str,
    limit: int = 1000,
    sort: str = "published_utc",
    order: str = "desc",
    cursor: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch one page of news from Polygon.io’s v2 reference/news endpoint.

    Parameters
    ----------
    api_key : str
        Your Polygon.io API key.
    limit : int
        Number of articles to fetch (max 100).
    sort : str
        Field to sort by (e.g. 'published_utc').
    order : str
        'asc' or 'desc'.
    cursor : Optional[str]
        Pagination cursor from prior response.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[str]]
        - DataFrame of this page’s articles.
        - Next-page cursor or None if no more pages.
    """
    session = requests.Session()
    url = "https://api.polygon.io/v2/reference/news"
    params = {"apiKey": api_key, "limit": limit, "sort": sort, "order": order}
    if cursor:
        params["cursor"] = cursor

    resp = session.get(url, params=params)
    resp.raise_for_status()
    payload = orjson.loads(resp.content)

    results = payload.get("results")
    if results is None:
        raise RuntimeError(f"Unexpected response: {payload!r}")

    df = pd.json_normalize(results)
    next_cursor = payload.get("cursor")
    return df, next_cursor


def fetch_all_polygon_news(
    api_key: str,
    limit: int = 1000,
    sort: str = "published_utc",
    order: str = "desc",
    max_pages: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch all pages of news, concatenating into one DataFrame.

    Parameters
    ----------
    api_key : str
        Your Polygon.io API key.
    limit : int
        Articles per page (max 100).
    sort : str
        Field to sort by.
    order : str
        'asc' or 'desc'.
    max_pages : Optional[int]
        Stop after this many pages (None → fetch all).

    Returns
    -------
    pd.DataFrame
        All fetched articles.
    """
    all_dfs = []
    cursor = None
    page = 0

    while True:
        df_page, cursor = fetch_polygon_news_page(api_key, limit, sort, order, cursor)
        all_dfs.append(df_page)
        page += 1
        if cursor is None:
            break
        if max_pages is not None and page >= max_pages:
            break

    return pd.concat(all_dfs, ignore_index=True)


# ── Example usage ──
if __name__ == "__main__":
    API_KEY = "cqWpEROd6Kq0Q0zihGGYEosjAi4IPd_w"
    # Fetch first 10 articles only:
    df1, cur = fetch_polygon_news_page(API_KEY, limit=1000)
    print(df1.shape, "next-cursor:", cur)
    print(df1.columns)

    # Fetch ALL available articles, 100 per page:
    df_all = fetch_all_polygon_news(API_KEY, limit=1000)
    print("Total articles fetched:", len(df_all))
    print(df_all)

