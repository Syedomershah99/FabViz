"""End-to-end tests for FabViz."""

import warnings
import ast
import sys

warnings.filterwarnings("ignore")


def test_data_loading():
    from data.loader import load_raw_data, get_dataset

    raw_features, raw_labels = load_raw_data()
    assert raw_features.shape[0] == 1567, f"Expected 1567 wafers, got {raw_features.shape[0]}"
    assert raw_features.shape[1] >= 500, f"Expected 500+ sensors, got {raw_features.shape[1]}"

    X, y, scaler, _ = get_dataset()
    assert len(X) == len(y), "X and y length mismatch"
    assert X.shape[1] > 50, f"Too few features after selection: {X.shape[1]}"
    assert set(y.unique()).issubset({0, 1}), "Labels should be binary (0/1)"
    assert (y == 0).sum() > (y == 1).sum(), "Pass count should exceed fail count"
    print(f"[PASS] Data loading: {X.shape[0]} wafers, {X.shape[1]} features")


def test_random_forest():
    from data.loader import get_dataset
    from models.classifier import train_and_evaluate

    X, y, _, _ = get_dataset()
    results = train_and_evaluate(X, y, model_name="rf")

    assert results["model"] is not None
    assert results["model_name"] == "Random Forest"
    assert 0 < results["metrics"]["accuracy"] <= 1
    assert 0 < results["metrics"]["roc_auc"] <= 1
    assert results["confusion_matrix"].shape == (2, 2)
    assert len(results["feature_importance"]) == X.shape[1]
    assert len(results["y_pred"]) == len(y)
    assert len(results["y_prob"]) == len(y)

    fpr, tpr = results["roc_curve"]
    assert len(fpr) > 2
    assert len(tpr) == len(fpr)
    print(f"[PASS] Random Forest: acc={results['metrics']['accuracy']:.3f}, auc={results['metrics']['roc_auc']:.3f}")


def test_xgboost():
    from data.loader import get_dataset
    from models.classifier import train_and_evaluate

    X, y, _, _ = get_dataset()
    results = train_and_evaluate(X, y, model_name="xgb")

    assert results["model"] is not None
    assert results["model_name"] == "XGBoost"
    assert 0 < results["metrics"]["accuracy"] <= 1
    assert 0 < results["metrics"]["roc_auc"] <= 1
    assert results["confusion_matrix"].shape == (2, 2)
    print(f"[PASS] XGBoost: acc={results['metrics']['accuracy']:.3f}, auc={results['metrics']['roc_auc']:.3f}")


def test_charts():
    import numpy as np
    import pandas as pd
    from data.loader import get_dataset
    from models.classifier import train_and_evaluate
    from viz.charts import (
        plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
        plot_spc_chart, plot_drift_detection, plot_correlation_heatmap,
        plot_class_distribution,
    )

    X, y, _, _ = get_dataset()
    results = train_and_evaluate(X, y, "rf")

    fig = plot_confusion_matrix(results["confusion_matrix"])
    assert fig is not None

    fpr, tpr = results["roc_curve"]
    fig = plot_roc_curve(fpr, tpr, results["metrics"]["roc_auc"], "RF")
    assert fig is not None

    fig = plot_feature_importance(results["feature_importance"], top_n=20)
    assert fig is not None

    sensor = X.columns[0]
    fig = plot_spc_chart(X[sensor], sensor, window=20)
    assert fig is not None

    fig = plot_drift_detection(X[sensor], sensor, window=50)
    assert fig is not None

    fig = plot_correlation_heatmap(X, top_n=15)
    assert fig is not None

    fig = plot_class_distribution(y)
    assert fig is not None

    print("[PASS] All 7 chart types render without errors")


def test_app_syntax():
    with open("app.py") as f:
        ast.parse(f.read())
    print("[PASS] app.py syntax valid")


if __name__ == "__main__":
    tests = [test_data_loading, test_random_forest, test_xgboost, test_charts, test_app_syntax]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    if failed == 0:
        print(f"ALL {len(tests)} TESTS PASSED")
    else:
        print(f"{failed}/{len(tests)} TESTS FAILED")
        sys.exit(1)
