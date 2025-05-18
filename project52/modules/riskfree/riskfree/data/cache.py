# riskfree/data/cache.py
from functools import lru_cache
import requests

@lru_cache(maxsize=60)
def get_month_xml(ym: str) -> bytes:
    url = ("https://home.treasury.gov/resource-center/data-chart-center/"
           "interest-rates/pages/xml")
    params = {"data": "daily_treasury_yield_curve",
              "field_tdr_date_value_month": ym}
    r = requests.get(url, params=params, timeout=(5, 30))
    r.raise_for_status()
    return r.content
