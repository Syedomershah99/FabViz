"""Plotly chart builders for FabViz."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> go.Figure:
    labels = ["Pass", "Fail"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18},
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=450,
        height=400,
    )
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                   model_name: str = "Model") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC={auc:.3f})",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=550,
        height=450,
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_feature_importance(fi: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = fi.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        title=f"Top {top_n} Important Sensors",
        xaxis_title="Importance",
        yaxis_title="Sensor",
        height=max(400, top_n * 25),
        margin=dict(l=120),
    )
    return fig


def plot_spc_chart(series: pd.Series, sensor_name: str,
                   window: int = 20) -> go.Figure:
    """X-bar SPC control chart with UCL/LCL."""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    overall_mean = series.mean()
    overall_std = series.std()
    ucl = overall_mean + 3 * overall_std
    lcl = overall_mean - 3 * overall_std

    ooc_mask = (series > ucl) | (series < lcl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(series))),
        y=series,
        mode="lines",
        name="Measurement",
        line=dict(color="#1f77b4", width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(rolling_mean))),
        y=rolling_mean,
        mode="lines",
        name=f"Moving Avg (w={window})",
        line=dict(color="#2ca02c", width=2),
    ))
    # Control limits
    fig.add_hline(y=overall_mean, line_dash="solid", line_color="black",
                  annotation_text="CL", annotation_position="top right")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text="UCL (+3σ)", annotation_position="top right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text="LCL (-3σ)", annotation_position="bottom right")

    # Out-of-control points
    if ooc_mask.any():
        fig.add_trace(go.Scatter(
            x=np.where(ooc_mask)[0].tolist(),
            y=series[ooc_mask].tolist(),
            mode="markers",
            name="Out of Control",
            marker=dict(color="red", size=6, symbol="x"),
        ))

    fig.update_layout(
        title=f"SPC Control Chart: {sensor_name}",
        xaxis_title="Wafer Index",
        yaxis_title="Sensor Value",
        height=400,
    )
    return fig


def plot_drift_detection(series: pd.Series, sensor_name: str,
                         window: int = 50) -> go.Figure:
    """Detect process drift using CUSUM-style moving average shift."""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    overall_mean = series.mean()
    overall_std = series.std()

    # Drift score: how many stds the rolling mean is from overall mean
    drift_score = ((rolling_mean - overall_mean) / overall_std).abs()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{sensor_name} - Rolling Mean",
                                        "Drift Score (|z|)"],
                        vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=list(range(len(rolling_mean))),
        y=rolling_mean,
        mode="lines",
        name="Rolling Mean",
        line=dict(color="#1f77b4"),
    ), row=1, col=1)
    fig.add_hline(y=overall_mean, line_dash="dash", line_color="gray",
                  row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(drift_score))),
        y=drift_score,
        mode="lines",
        name="Drift Score",
        line=dict(color="#ff7f0e"),
    ), row=2, col=1)
    fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                  annotation_text="Drift Threshold (2σ)",
                  row=2, col=1)

    fig.update_layout(height=550, showlegend=True)
    fig.update_xaxes(title_text="Wafer Index", row=2, col=1)
    return fig


def plot_correlation_heatmap(X: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Correlation heatmap for top-N features by variance."""
    top_cols = X.var().nlargest(top_n).index.tolist()
    corr = X[top_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
    ))
    fig.update_layout(
        title=f"Correlation Heatmap (Top {top_n} Sensors by Variance)",
        width=700,
        height=650,
    )
    return fig


def plot_class_distribution(y: pd.Series) -> go.Figure:
    counts = y.value_counts().sort_index()
    labels = ["Pass (0)", "Fail (1)"]
    colors = ["#2ca02c", "#d62728"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition="auto",
    ))
    fig.update_layout(
        title="Yield Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=350,
        width=400,
    )
    return fig
