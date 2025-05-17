# File: chen3/operations/serialization.py
"""
Utilities for serializing simulation outputs.
"""
import pandas as pd


def to_parquet(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to Parquet."""
    df.to_parquet(filepath, index=False)


def to_feather(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to Feather."""
    df.to_feather(filepath)


def read_parquet(filepath: str) -> pd.DataFrame:
    """Read DataFrame from Parquet."""
    return pd.read_parquet(filepath)


def read_feather(filepath: str) -> pd.DataFrame:
    """Read DataFrame from Feather."""
    return pd.read_feather(filepath)
