# FabViz: Complete Technical Documentation

## Table of Contents

1. [What is FabViz?](#1-what-is-fabviz)
2. [The Problem it Solves](#2-the-problem-it-solves)
3. [The Dataset: UCI SECOM](#3-the-dataset-uci-secom)
4. [Project Architecture](#4-project-architecture)
5. [Data Pipeline (data/loader.py)](#5-data-pipeline)
6. [Machine Learning Models (models/classifier.py)](#6-machine-learning-models)
7. [Visualization Engine (viz/charts.py)](#7-visualization-engine)
8. [The Streamlit App (app.py)](#8-the-streamlit-app)
9. [Key Technical Concepts Explained](#9-key-technical-concepts-explained)
10. [How to Run](#10-how-to-run)
11. [Deployment](#11-deployment)

---

## 1. What is FabViz?

FabViz is an interactive web dashboard for analyzing semiconductor manufacturing (fab) data. It takes raw sensor readings from a chip fabrication line, cleans and processes them, trains machine learning models to predict which wafers will pass or fail quality checks, and visualizes everything in real time.

Think of it as a quality control dashboard that a fab engineer could use to:
- Predict wafer yield (pass/fail) before final testing
- Monitor individual sensors for abnormal behavior
- Identify which sensors matter most for yield
- Detect when a manufacturing process is drifting out of spec

---

## 2. The Problem it Solves

In semiconductor manufacturing, a single wafer goes through hundreds of processing steps (deposition, etching, lithography, etc.). At each step, sensors record measurements like temperature, pressure, gas flow, and chemical concentrations. By the time a wafer reaches final electrical testing, it has accumulated 500+ sensor readings.

The challenge: most wafers pass (~93%), but the ~7% that fail are expensive. Each failed wafer wastes hours of processing time and thousands of dollars in materials. If you could predict failure earlier using sensor data, you could:
- Stop processing a bad wafer before wasting more resources
- Identify which process step is causing failures
- Catch equipment drift before it produces a batch of bad wafers

FabViz addresses all three of these by combining classification (predict pass/fail), feature importance (which sensors matter), and process monitoring (SPC charts and drift detection).

---

## 3. The Dataset: UCI SECOM

**Source:** [UCI Machine Learning Repository - SECOM](https://archive.ics.uci.edu/dataset/179/secom)

**What it contains:**
- **1,567 wafers** (rows) - each row is one wafer that went through the fab
- **590 sensors** (columns) - each column is a measurement from a different process sensor
- **Labels** - each wafer is labeled as Pass (-1 in raw data, mapped to 0) or Fail (1)

**Key characteristics:**
- **Heavy class imbalance:** ~93% Pass, ~7% Fail (14:1 ratio). This is realistic because most fabs operate at high yield.
- **Lots of missing data:** Many sensor columns have NaN values (sensor was offline, reading was corrupted, etc.)
- **High dimensionality:** 590 features is a lot. Many are redundant (correlated with each other) or useless (near-zero variance).
- **No feature names:** Sensors are anonymized (Sensor_0, Sensor_1, ...) for confidentiality. In a real fab, these would have names like "Chamber_3_Pressure" or "Etch_Rate_Zone_2".

The dataset is automatically downloaded on first run. No manual setup needed.

---

## 4. Project Architecture

```
FabViz/
├── app.py                  # Main Streamlit app (UI and page routing)
├── data/
│   ├── __init__.py
│   └── loader.py           # Data download, cleaning, feature selection, scaling
├── models/
│   ├── __init__.py
│   └── classifier.py       # Random Forest and XGBoost training and evaluation
├── viz/
│   ├── __init__.py
│   └── charts.py           # All Plotly chart builders (7 chart types)
├── tests.py                # End-to-end test suite
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

**Data flow:**

```
UCI SECOM (web) --> loader.py (download + clean) --> classifier.py (train models) --> charts.py (visualize) --> app.py (display)
```

Each module has a single responsibility:
- `loader.py` handles everything data-related (download, impute, select features, scale)
- `classifier.py` handles everything model-related (train, evaluate, feature importance)
- `charts.py` handles everything visual (7 different chart types)
- `app.py` ties them together with the Streamlit UI

---

## 5. Data Pipeline

**File:** `data/loader.py`

### Step 1: Download

The raw data is two files from UCI:
- `secom.data` - whitespace-separated matrix of sensor readings (1,567 x 590)
- `secom_labels.data` - pass/fail labels with timestamps

The `download_file()` function downloads these only if they don't already exist locally. This avoids re-downloading on every app restart.

### Step 2: Load

`load_raw_data()` reads both files into pandas DataFrames. The raw labels use -1 for Pass and 1 for Fail. We remap these to 0 (Pass) and 1 (Fail) because scikit-learn expects this convention.

Sensor columns are named `Sensor_0` through `Sensor_589` since the original data has no headers.

### Step 3: Preprocess

`preprocess()` runs 5 cleaning steps in order:

**a) Drop columns with >50% missing values**

If a sensor column is more than half NaN, it's unreliable. We remove it entirely. This typically drops around 100 columns.

**b) Median imputation**

For remaining NaN values, we fill them with the median of that column. Why median instead of mean? Because sensor data often has outliers (a spike to 9999 when a sensor glitches), and the median is robust to outliers while the mean gets pulled toward them.

We use scikit-learn's `SimpleImputer` for this.

**c) Near-zero-variance removal**

Some sensors barely change across wafers. If a sensor reads 3.0001, 3.0002, 3.0001, ... with variance below 0.01, it carries almost no useful information. `VarianceThreshold` removes these.

**d) High-correlation removal**

If Sensor_12 and Sensor_47 have a correlation of 0.98, they're measuring nearly the same thing. Keeping both wastes compute and can confuse models. We compute the full correlation matrix, find pairs with |r| > 0.95, and drop one from each pair.

The implementation uses a triangular matrix trick: we mask the lower triangle and diagonal of the correlation matrix (to avoid counting each pair twice), then flag any column that has a correlation above the threshold with any other column.

**e) Standard scaling**

Finally, we scale all features to have zero mean and unit variance using `StandardScaler`. This is important because:
- Different sensors operate at wildly different scales (one might read 0.001-0.002, another might read 500-1500)
- Many ML algorithms (and especially distance-based metrics) are sensitive to scale
- SPC control charts need standardized data for meaningful sigma-based limits

**Result:** 590 raw features are reduced to ~195 clean, scaled features.

---

## 6. Machine Learning Models

**File:** `models/classifier.py`

### Why Two Models?

We train both Random Forest (RF) and XGBoost (XGB) to give users a comparison:

- **Random Forest** builds many independent decision trees and averages their predictions. It's robust, hard to overtrain, and gives reliable feature importance scores.
- **XGBoost** builds trees sequentially, where each new tree corrects the mistakes of the previous ones (boosting). It often achieves higher accuracy but can overfit if not tuned carefully.

### Handling Class Imbalance

With a 14:1 pass-to-fail ratio, a model that always predicts "Pass" gets 93% accuracy but catches zero failures. This is useless in a fab.

We handle this with **class-weight balancing**:
- **Random Forest:** `class_weight="balanced"` tells scikit-learn to weight the Fail class 14x higher during training, so misclassifying a Fail wafer costs 14x more than misclassifying a Pass wafer.
- **XGBoost:** `scale_pos_weight` does the same thing. We compute it as `count(Pass) / count(Fail)`.

### Cross-Validation

We use **5-fold stratified cross-validation** to evaluate models honestly:

1. Split the data into 5 equal parts (folds)
2. Train on 4 folds, test on the remaining 1
3. Rotate which fold is the test set
4. Every wafer gets exactly one prediction (when it was in the test fold)

**Stratified** means each fold maintains the same pass/fail ratio as the full dataset. Without stratification, a fold might accidentally get zero fail samples.

We implement this manually (loop over `StratifiedKFold.split()`) rather than using scikit-learn's `cross_val_predict` to avoid compatibility issues between scikit-learn and XGBoost versions.

### Metrics Computed

| Metric | What it Measures |
|---|---|
| **Accuracy** | Overall % of correct predictions |
| **Precision** | Of wafers predicted as Fail, what % actually failed? (avoid false alarms) |
| **Recall** | Of wafers that actually failed, what % did we catch? (avoid missed defects) |
| **F1 Score** | Harmonic mean of precision and recall (balances both) |
| **ROC AUC** | Area under the ROC curve (overall ranking quality, threshold-independent) |

In fab contexts, **recall** is usually the most important metric. Missing a defective wafer (false negative) is more expensive than a false alarm (false positive), because false alarms just trigger an extra inspection while missed defects ship bad product.

### Feature Importance

After cross-validation, we retrain the final model on all data and extract `feature_importances_`. For tree-based models, importance is measured by how much each feature reduces impurity (Gini impurity for RF, gain for XGBoost) across all trees and splits.

---

## 7. Visualization Engine

**File:** `viz/charts.py`

All charts use Plotly for interactivity (hover, zoom, pan). Seven chart types:

### a) Confusion Matrix (`plot_confusion_matrix`)
A 2x2 heatmap showing:
- True Positives (correctly predicted Fail)
- True Negatives (correctly predicted Pass)
- False Positives (Pass wafer wrongly called Fail)
- False Negatives (Fail wafer wrongly called Pass)

### b) ROC Curve (`plot_roc_curve`)
Plots True Positive Rate vs False Positive Rate at every possible classification threshold. The diagonal gray line represents a random classifier (AUC = 0.5). The further the curve bows toward the top-left corner, the better the model.

AUC (Area Under the Curve) summarizes this in one number: 1.0 is perfect, 0.5 is random.

### c) Feature Importance (`plot_feature_importance`)
Horizontal bar chart of the top-N most important sensors. Helps fab engineers focus their attention on the sensors that actually matter for yield.

### d) SPC Control Chart (`plot_spc_chart`)
**Statistical Process Control (SPC)** is a standard manufacturing quality tool. The chart shows:
- Raw sensor measurements (light blue line)
- Moving average (green line, smoothed trend)
- Center Line (CL) = overall mean
- Upper Control Limit (UCL) = mean + 3 standard deviations
- Lower Control Limit (LCL) = mean - 3 standard deviations
- Out-of-control points (red X markers) = measurements beyond UCL or LCL

The "3 sigma" limits come from the empirical rule: ~99.7% of normally distributed data falls within 3 standard deviations of the mean. Points outside this range are statistically unusual and warrant investigation.

### e) Process Drift Detection (`plot_drift_detection`)
A two-panel chart:
- **Top panel:** Rolling mean of the sensor over a sliding window. If the process is stable, this should stay flat near the overall mean.
- **Bottom panel:** Drift score = |rolling_mean - overall_mean| / overall_std. This is a z-score that measures how many standard deviations the local mean has shifted from the global mean.

A drift score above 2.0 (the red threshold line) means the process has shifted by more than 2 sigma. This is a sign that equipment may be degrading, a chemical is running low, or some other systematic change is happening.

### f) Correlation Heatmap (`plot_correlation_heatmap`)
Shows pairwise Pearson correlations between the top-N highest-variance sensors. Red = positive correlation (sensors move together), blue = negative correlation (sensors move opposite), white = no correlation.

Helps identify redundant sensors and understand process relationships (e.g., if temperature and pressure are highly correlated, they might be driven by the same process step).

### g) Class Distribution (`plot_class_distribution`)
Simple bar chart showing the count of Pass vs Fail wafers. Visually communicates the class imbalance.

---

## 8. The Streamlit App

**File:** `app.py`

### Caching

Two decorators are used to avoid recomputing expensive operations on every page interaction:
- `@st.cache_data` for data loading: runs once, then reuses the result for all subsequent calls
- `@st.cache_resource` for model training: trains each model type once and caches it

### Pages

The sidebar radio button selects between 5 pages:

**1. Dashboard**
- 4 metric cards: Total Wafers, Raw Sensors, Features (after selection), Yield Rate
- Class distribution bar chart
- Dataset summary table
- Preprocessing steps list

**2. Yield Classification**
- Dropdown to select RF or XGBoost
- 5 metric cards (accuracy, precision, recall, F1, AUC)
- Confusion matrix and ROC curve side by side
- Expandable classification report

**3. SPC Control Charts**
- Dropdown to select any sensor
- Slider for moving average window size
- Interactive SPC chart with control limits
- 4 summary metrics (mean, std dev, OOC count, OOC rate)

**4. Feature Importance**
- Dropdown to select model type
- Slider for top-N features
- Horizontal bar chart of feature importances
- Correlation heatmap for top sensors

**5. Process Drift**
- Dropdown to select sensor
- Slider for rolling window size
- Two-panel drift detection chart
- Table of top 20 drifting sensors ranked by maximum drift score

---

## 9. Key Technical Concepts Explained

### Wafer
A thin disc of silicon (usually 300mm diameter) on which hundreds of individual chips are fabricated simultaneously. After processing, the wafer is diced into individual chips.

### Yield
The percentage of wafers (or chips) that pass final quality testing. A fab running at 93% yield means 93 out of every 100 wafers are good.

### SPC (Statistical Process Control)
A set of methods for monitoring a manufacturing process using statistics. The core idea: if a process is "in control" (running normally), measurements will stay within predictable limits. Points outside those limits signal something has changed.

### Control Limits (UCL/LCL)
Upper and Lower Control Limits. Set at +/- 3 standard deviations from the mean. These are NOT the same as specification limits (which define acceptable product). Control limits define the natural variation of the process itself. A point outside control limits means the process has changed, even if the product might still be in spec.

### Feature Importance (Gini/Gain)
For tree-based models, each time a feature is used to split data at a node, it reduces "impurity" (how mixed the pass/fail labels are in each resulting group). Features that produce the best splits across many trees rank higher in importance.

### Cross-Validation
A technique for estimating how well a model will perform on new, unseen data. Instead of a single train/test split (which can be lucky or unlucky), we average performance across multiple splits. 5-fold CV means every data point is tested exactly once.

### Stratified Sampling
When splitting data for cross-validation, maintaining the same class ratio in each fold. Critical for imbalanced datasets: without stratification, some folds might have zero fail samples.

### Class Imbalance
When one class vastly outnumbers the other. The SECOM dataset has 14:1 pass-to-fail ratio. Naive models learn to always predict the majority class. Solutions include class weights, oversampling (SMOTE), undersampling, or specialized loss functions.

### ROC AUC
Receiver Operating Characteristic - Area Under the Curve. Measures a model's ability to rank positive examples higher than negative examples, across all possible thresholds. Insensitive to class imbalance, making it a better metric than accuracy for datasets like SECOM.

### Process Drift
A gradual, systematic change in a process over time. Unlike a sudden equipment failure (which shows up as a spike), drift is slow and can go unnoticed until yield drops. Examples: a chemical bath slowly depleting, a heater element degrading, a vacuum seal slowly leaking.

### Standard Scaling (z-score normalization)
Transforms each feature to have mean=0 and standard deviation=1. Formula: z = (x - mean) / std. Makes features comparable regardless of their original scale.

---

## 10. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# macOS users: XGBoost requires OpenMP
brew install libomp

# Run the app
streamlit run app.py

# Run tests
python tests.py
```

The app automatically downloads the SECOM dataset on first launch. All subsequent launches use the cached local copy.

---

## 11. Deployment

The app is deployed on Streamlit Community Cloud at:

**https://fabviz.streamlit.app**

To deploy your own instance:
1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repo, branch (main), and main file (app.py)
5. Click Deploy

Streamlit Cloud automatically installs `requirements.txt` and runs the app. No Docker or server configuration needed.
