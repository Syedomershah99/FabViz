"""Data loading and preprocessing for UCI SECOM fab dataset."""

import os
import requests
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

DATA_DIR = os.path.join(os.path.dirname(__file__))
SECOM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


def download_file(url: str, filepath: str) -> None:
    """Download a file if it doesn't already exist."""
    if not os.path.exists(filepath):
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)


def load_raw_data() -> tuple[pd.DataFrame, pd.Series]:
    """Download (if needed) and load raw SECOM data and labels."""
    secom_path = os.path.join(DATA_DIR, "secom.data")
    labels_path = os.path.join(DATA_DIR, "secom_labels.data")

    download_file(SECOM_URL, secom_path)
    download_file(LABELS_URL, labels_path)

    features = pd.read_csv(secom_path, sep=r"\s+", header=None, na_values="NaN")
    features.columns = [f"Sensor_{i}" for i in range(features.shape[1])]

    labels_df = pd.read_csv(labels_path, sep=r"\s+", header=None)
    labels = labels_df.iloc[:, 0].map({-1: 0, 1: 1})  # 0=Pass, 1=Fail

    return features, labels


def preprocess(features: pd.DataFrame, labels: pd.Series,
               variance_threshold: float = 0.01,
               correlation_threshold: float = 0.95
               ) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Clean, impute, select features, and scale."""
    # Drop columns that are >50% missing
    missing_frac = features.isnull().mean()
    features = features.loc[:, missing_frac < 0.5]

    # Impute remaining NaNs with median
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    # Remove near-zero-variance features
    vt = VarianceThreshold(threshold=variance_threshold)
    vt.fit(imputed)
    kept_cols = imputed.columns[vt.get_support()]
    imputed = imputed[kept_cols]

    # Remove highly correlated features
    corr_matrix = imputed.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    imputed = imputed.drop(columns=drop_cols)

    # Scale
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(imputed),
        columns=imputed.columns,
        index=imputed.index,
    )

    # Drop rows where label is NaN
    valid = labels.notna()
    scaled = scaled.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)

    return scaled, labels, scaler


def get_dataset(variance_threshold: float = 0.01,
                correlation_threshold: float = 0.95):
    """Full pipeline: download, load, preprocess. Returns (X, y, scaler, raw_features)."""
    raw_features, labels = load_raw_data()
    X, y, scaler = preprocess(raw_features, labels, variance_threshold, correlation_threshold)
    return X, y, scaler, raw_features
