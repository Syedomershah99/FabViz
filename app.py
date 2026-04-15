"""FabViz: Semiconductor Yield Analysis Dashboard.

Streamlit app for exploring the UCI SECOM fab dataset (1,567 wafers, 591 sensors).
Features yield classification (RF/XGBoost), SPC control charts, feature importance,
and process drift detection.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="FabViz - Semiconductor Yield Analysis",
    page_icon="🔬",
    layout="wide",
)

# --- Data Loading (cached) ---

@st.cache_data(show_spinner="Downloading and preprocessing SECOM data...")
def load_data():
    from data.loader import get_dataset, load_raw_data
    raw_features, raw_labels = load_raw_data()
    X, y, scaler, _ = get_dataset()
    return X, y, raw_features, raw_labels


@st.cache_resource(show_spinner="Training models...")
def train_models(model_name: str):
    from models.classifier import train_and_evaluate
    X, y, _, _ = load_data()
    return train_and_evaluate(X, y, model_name=model_name)


# --- Sidebar ---

st.sidebar.title("FabViz")
st.sidebar.caption("Semiconductor Yield Analysis")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Yield Classification", "SPC Control Charts",
     "Feature Importance", "Process Drift"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)  \n"
    "1,567 wafers | 591 sensors"
)

# --- Pages ---

if page == "Dashboard":
    st.title("Semiconductor Yield Analysis Dashboard")
    st.markdown("Explore the UCI SECOM fab dataset: wafer yield classification, "
                "statistical process control, and sensor drift detection.")

    X, y, raw_features, raw_labels = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Wafers", f"{len(raw_features):,}")
    col2.metric("Raw Sensors", f"{raw_features.shape[1]}")
    col3.metric("Features (after selection)", f"{X.shape[1]}")

    pass_count = (y == 0).sum()
    fail_count = (y == 1).sum()
    yield_pct = pass_count / len(y) * 100
    col4.metric("Yield Rate", f"{yield_pct:.1f}%")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        from viz.charts import plot_class_distribution
        st.plotly_chart(plot_class_distribution(y), use_container_width=True)

    with right:
        st.subheader("Dataset Summary")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Wafers | {len(raw_features):,} |
        | Original Sensors | {raw_features.shape[1]} |
        | After Preprocessing | {X.shape[1]} |
        | Pass | {pass_count:,} ({yield_pct:.1f}%) |
        | Fail | {fail_count:,} ({100 - yield_pct:.1f}%) |
        | Imbalance Ratio | {pass_count // max(fail_count, 1)}:1 |
        """)

        st.subheader("Preprocessing Steps")
        st.markdown("""
        1. Dropped columns with >50% missing values
        2. Median imputation for remaining NaNs
        3. Removed near-zero-variance features
        4. Removed highly correlated features (r > 0.95)
        5. Standard scaling (zero mean, unit variance)
        """)


elif page == "Yield Classification":
    st.title("Yield Classification")

    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
    model_key = "rf" if model_choice == "Random Forest" else "xgb"

    results = train_models(model_key)
    metrics = results["metrics"]

    st.subheader("Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("F1 Score", f"{metrics['f1']:.3f}")
    c5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("*5-fold stratified cross-validation with class-weight balancing.*")
    st.markdown("---")

    from viz.charts import plot_confusion_matrix, plot_roc_curve

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_confusion_matrix(results["confusion_matrix"],
                                  title=f"Confusion Matrix - {model_choice}"),
            use_container_width=True,
        )
    with right:
        fpr, tpr = results["roc_curve"]
        st.plotly_chart(
            plot_roc_curve(fpr, tpr, metrics["roc_auc"], model_choice),
            use_container_width=True,
        )

    with st.expander("Classification Report"):
        st.code(results["classification_report"])


elif page == "SPC Control Charts":
    st.title("Statistical Process Control (SPC)")
    st.markdown("X-bar control charts with upper/lower control limits (UCL/LCL) at 3 sigma. "
                "Red markers indicate out-of-control measurements.")

    X, y, raw_features, _ = load_data()

    sensor_list = X.columns.tolist()
    selected_sensor = st.selectbox("Select Sensor", sensor_list, index=0)
    window = st.slider("Moving Average Window", min_value=5, max_value=100, value=20)

    from viz.charts import plot_spc_chart
    fig = plot_spc_chart(X[selected_sensor], selected_sensor, window=window)
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    series = X[selected_sensor]
    ooc_count = ((series > series.mean() + 3 * series.std()) |
                 (series < series.mean() - 3 * series.std())).sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{series.mean():.4f}")
    c2.metric("Std Dev", f"{series.std():.4f}")
    c3.metric("Out of Control", f"{ooc_count}")
    c4.metric("OOC Rate", f"{ooc_count / len(series) * 100:.1f}%")


elif page == "Feature Importance":
    st.title("Feature Importance Analysis")

    model_choice = st.selectbox("Model for Importance", ["Random Forest", "XGBoost"])
    model_key = "rf" if model_choice == "Random Forest" else "xgb"
    top_n = st.slider("Top N Features", min_value=10, max_value=50, value=20)

    results = train_models(model_key)

    from viz.charts import plot_feature_importance, plot_correlation_heatmap

    st.plotly_chart(
        plot_feature_importance(results["feature_importance"], top_n=top_n),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Sensor Correlation Heatmap")

    X, y, _, _ = load_data()
    heatmap_n = st.slider("Heatmap Sensors", min_value=10, max_value=30, value=15)
    st.plotly_chart(
        plot_correlation_heatmap(X, top_n=heatmap_n),
        use_container_width=True,
    )


elif page == "Process Drift":
    st.title("Process Drift Detection")
    st.markdown("Monitors sensor readings for systematic shifts using rolling z-scores. "
                "A drift score above 2 sigma flags potential process instability.")

    X, y, _, _ = load_data()

    sensor_list = X.columns.tolist()
    selected_sensor = st.selectbox("Select Sensor", sensor_list, index=0)
    window = st.slider("Rolling Window Size", min_value=20, max_value=200, value=50)

    from viz.charts import plot_drift_detection
    fig = plot_drift_detection(X[selected_sensor], selected_sensor, window=window)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table: top drifting sensors
    st.markdown("---")
    st.subheader("Top Drifting Sensors")
    st.markdown("Sensors ranked by maximum rolling drift score (higher = more drift).")

    drift_scores = {}
    for col in X.columns:
        rolling_mean = X[col].rolling(window=window, min_periods=1).mean()
        overall_mean = X[col].mean()
        overall_std = X[col].std()
        if overall_std > 0:
            max_drift = ((rolling_mean - overall_mean) / overall_std).abs().max()
            drift_scores[col] = max_drift

    drift_df = pd.DataFrame(
        sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)[:20],
        columns=["Sensor", "Max Drift Score"],
    )
    drift_df.index = range(1, len(drift_df) + 1)
    st.dataframe(drift_df, use_container_width=True)
