# File: adjustments/anomaly_detection.py
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Identify outlier yield observations."""
    clf = IsolationForest(random_state=42)
    df = df.dropna().reset_index(drop=True)
    clf.fit(df[['yield']])
    df['anomaly'] = clf.predict(df[['yield']])
    return df
