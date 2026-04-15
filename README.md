# FabViz - Semiconductor Yield Analysis

Interactive Streamlit dashboard for exploring the UCI SECOM semiconductor fabrication dataset (1,567 wafers, 590 sensor features). Built for yield classification, statistical process control, and process drift detection.

**Live Demo:** [fabviz.streamlit.app](https://fabviz.streamlit.app)

## Features

- **Dashboard** -- Overview of dataset statistics, yield distribution, and preprocessing summary
- **Yield Classification** -- Random Forest and XGBoost classifiers with ROC curves, confusion matrices, and classification reports (5-fold stratified CV with class-weight balancing)
- **SPC Control Charts** -- X-bar charts with UCL/LCL at 3 sigma, out-of-control point detection
- **Feature Importance** -- Sensor ranking by model importance, correlation heatmaps
- **Process Drift Detection** -- Rolling z-score drift monitoring, top drifting sensor ranking

## Setup

```bash
pip install -r requirements.txt

# macOS: XGBoost requires OpenMP
brew install libomp

streamlit run app.py
```

The app automatically downloads the UCI SECOM dataset on first run.

## Documentation

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for a detailed walkthrough of every component, all technical concepts explained in plain language, and the full data pipeline from raw sensor readings to interactive visualizations.

## Tech Stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, XGBoost, Plotly, imbalanced-learn

## Dataset

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) -- 1,567 wafers, 590 process sensor measurements, binary pass/fail labels (~93% pass, ~7% fail).
