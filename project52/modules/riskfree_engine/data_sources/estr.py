# -------------------------------------------------
# data_sources/estr.py (ECB SDW)
# -------------------------------------------------
import requests
import pandas as pd
from xml.etree import ElementTree as ET
from .base import BaseDataSource

class ESTRDataSource(BaseDataSource):
    # One‑shot query returning an SDMX‑XML with daily rates
    BASE_URL = (
        "https://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=ECB.D.ESTR.DAILY.ESTR.RATE"
    )

    def _parse_xml(self, xml_text: str) -> pd.DataFrame:
        ns = {"generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}
        root = ET.fromstring(xml_text)
        series = root.find(".//generic:Series", ns)
        if series is None:
            return pd.DataFrame()
        obs = []
        for obs_node in series.findall("generic:Obs", ns):
            date_str = obs_node.find("generic:ObsDimension", ns).attrib["value"]
            val = obs_node.find("generic:ObsValue", ns).attrib["value"]
            obs.append((pd.to_datetime(date_str), float(val) / 100))
        return pd.DataFrame(obs, columns=["date", "yield"])

    def fetch_data(self, start_date, end_date):
        cache_name = f"estr_{start_date}_{end_date}"
        if (c := self.try_cache(cache_name)) is not None:
            return c
        r = requests.get(self.BASE_URL, timeout=60)
        r.raise_for_status()
        df = self._parse_xml(r.text)
        df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
        df["maturity"] = 1 / 365
        self.validate_data(df)
        self.cache_data(df, cache_name)
        return df[["date", "maturity", "yield"]]