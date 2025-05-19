# File: caching/cache.py
import sqlite3
import pandas as pd

class CacheManager:
    """Simple SQLite-based cache manager."""
    def __init__(self, db_path: str = 'cache/cache.db'):
        self.conn = sqlite3.connect(db_path)

    def to_cache(self, table: str, df: pd.DataFrame):
        df.to_sql(table, self.conn, if_exists='replace', index=False)

    def from_cache(self, table: str) -> pd.DataFrame:
        return pd.read_sql(f"SELECT * FROM {table}", self.conn)

