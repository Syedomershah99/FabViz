"""Yield classification models: Random Forest and XGBoost."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)
from xgboost import XGBClassifier


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str = "rf",
                       n_splits: int = 5, handle_imbalance: str = "class_weight"):
    """Train model with stratified k-fold CV. Returns model, metrics, predictions."""
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced" if handle_imbalance == "class_weight" else None,
            random_state=42,
            n_jobs=-1,
        )
    else:
        scale = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale if handle_imbalance == "class_weight" else 1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Manual CV to avoid sklearn/xgboost compatibility issues
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        y_prob[test_idx] = model.predict_proba(X_test)[:, 1]

    # Fit final model on all data for feature importance
    model.fit(X, y)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_prob)
    report = classification_report(y, y_pred, target_names=["Pass", "Fail"])

    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "model_name": "Random Forest" if model_name == "rf" else "XGBoost",
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "classification_report": report,
        "feature_importance": feature_importance,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
