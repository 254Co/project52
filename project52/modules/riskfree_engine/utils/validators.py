# File: utils/validators.py
import pandas as pd


def validate_non_empty(df: pd.DataFrame, name: str):
    assert not df.empty, f"DataFrame {name} is empty."

