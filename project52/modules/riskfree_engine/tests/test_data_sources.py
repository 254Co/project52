# File: tests/test_data_sources.py
import pytest
from data_sources.treasury import TreasuryDataSource

def test_treasury_fetch():
    src = TreasuryDataSource()
    df = src.fetch_data('2025-01-01', '2025-05-01')
    assert 'yield' in df.columns
