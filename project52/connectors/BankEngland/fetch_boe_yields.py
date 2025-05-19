#!/usr/bin/env python3
"""
fetch_boe_yields.py

Fetch the latest daily UK government (gilt) yield curve data from the
Bank of England “latest yield curve data” ZIP, ignoring any “info”
tab in the Excel workbook.
"""
import requests
import zipfile
import io
import pandas as pd

# Browser-style headers to avoid 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
    ),
    "Referer": "https://www.bankofengland.co.uk/statistics/yield-curves"
}

ZIP_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/"
    "statistics/yield-curves/latest-yield-curve-data.zip"
)


def fetch_daily_yield_curve() -> pd.DataFrame:
    # 1) Download ZIP
    resp = requests.get(ZIP_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    # 2) Open in-memory ZIP
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    # look for CSV or Excel inside
    csvs = [f for f in z.namelist() if f.lower().endswith(".csv")]
    xls  = [f for f in z.namelist() if f.lower().endswith((".xls", ".xlsx"))]

    if csvs:
        with z.open(csvs[0]) as f:
            return _parse_csv(f)

    elif xls:
        # pick the “nominal” file if present, else first
        fn = next((f for f in xls if "nominal" in f.lower()), xls[0])
        # read its bytes once
        data = z.read(fn)
        # build an ExcelFile and ignore any sheet named “info”
        xfile = pd.ExcelFile(io.BytesIO(data))
        sheets = [s for s in xfile.sheet_names if "info" not in s.lower()]
        if not sheets:
            raise FileNotFoundError(f"All sheets are ‘info’: {xfile.sheet_names!r}")

        # try each remaining sheet until one parses
        last_err = None
        for sheet in sheets:
            try:
                return _parse_weird_excel(io.BytesIO(data), sheet)
            except Exception as e:
                last_err = e
        # if none worked, bubble up the last error
        raise last_err

    else:
        raise FileNotFoundError(f"No CSV/Excel in ZIP – contents: {z.namelist()!r}")


def _parse_csv(fobj) -> pd.DataFrame:
    df = pd.read_csv(fobj)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    return df.apply(pd.to_numeric, errors="coerce")


def _parse_weird_excel(buffer, sheet_name) -> pd.DataFrame:
    # read raw with no header
    raw = pd.read_excel(buffer, sheet_name=sheet_name, header=None)

    # find the “years:” row for maturities
    mask = raw.astype(str).applymap(
        lambda x: "years:" in x.lower() if isinstance(x, str) else False
    )
    if not mask.any(axis=None):
        raise ValueError(f"No ‘years:’ header row in sheet {sheet_name!r}")
    header_row = mask.any(axis=1).idxmax()

    # extract maturities (skip first cell)
    mat_vals = raw.iloc[header_row, 1:].tolist()
    maturities = pd.to_numeric(mat_vals, errors="coerce").tolist()

    # find first date row below
    data_start = None
    for i in range(header_row + 1, len(raw)):
        try:
            if pd.notna(pd.to_datetime(raw.iat[i, 0], dayfirst=True)):
                data_start = i
                break
        except Exception:
            continue
    if data_start is None:
        raise ValueError(f"No date row found in sheet {sheet_name!r}")

    # slice out data block
    block = raw.iloc[data_start:, : len(maturities) + 1]
    cols  = ["Date"] + maturities
    block.columns = cols

    # parse and clean
    block["Date"] = pd.to_datetime(block["Date"], dayfirst=True, errors="coerce")
    df = block.set_index("Date").apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)

    # convert year‐based columns to <months>MONTH, e.g. 0.5→6MONTH, 1.0→12MONTH, …
    df.rename(columns={
        col: f"{int(float(col) * 12)}MONTH"
        for col in df.columns
    }, inplace=True)

    return df


if __name__ == "__main__":
    df = fetch_daily_yield_curve()
    print(df.head())
