# Freight Rate Prediction Challenge — ML Engineer Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CatBoost](https://img.shields.io/badge/model-CatBoost-orange.svg)](https://catboost.ai/)
[![Validation MAE](https://img.shields.io/badge/OOT%20MAE-$66.14-green.svg)]()
[![Validation R2](https://img.shields.io/badge/OOT%20R2-0.9773-brightgreen.svg)]()

> **Candidate:** Machine Learning Engineer  
> **Loom Video Walkthrough:** [Watch 2-3 Minute Overview on Loom](https://www.loom.com/share/9ad85344b8694929a2051b9653458cec)

---

## 📌 Executive Summary

This repository contains the end-to-end machine learning pricing engine developed for the **Freight Rate Prediction Challenge**.

By reframing the regression problem from raw dollar estimation to a **Residual Multiplier formulation** ($z = y / [\text{Distance} \times \text{Quote Signal}]$) and engineering directional/spatial-distance interaction vectors, the model achieves state-of-the-art accuracy across unseen temporal validation horizons:

* **Out-of-Time MAE:** **`$66.14`** (slashing initial baseline error by over **71%**)
* **Coefficient of Determination ($R^2$):** **`0.9773`** ($97.73\%$ explained variance)
* **Linehaul (>800 mi) MAPE:** **`2.50%`**
* **Statistical Significance:** Paired Wilcoxon Signed-Rank Test $p$-value = **`8.92e-228`** (Zero confidence interval overlap against direct baseline).

---

## 🔬 Methodology & Key Engineering Insights

### 1. Data Quality Remediation
* **Sign-Inverted Weights:** Resolved 292 negative weight records via `df['weight'].abs()`, restoring a clean Gaussian distribution capped at the legal US DOT 48,000 lb cargo payload limit.
* **Missing Value Imputation:** Missing weights were imputed using equipment-class medians; temporal features (`market_index` and `quote_signal`) were reconstructed via forward-interpolation and lane-level medians.
* **Synthetic Label Noise Filtering:** Identified that ~1.4% of training rows contained discrete integer multiplier corruptions ($3\times, 5\times, \frac{1}{3}\times$). These were filtered from the training set to prevent gradient distortion.

### 2. Temporal Out-of-Time (OOT) Validation Split
Standard randomized $K$-Fold cross-validation was rejected due to **temporal data leakage** (leaking forward macroeconomic cycles into past predictions).
* **Training Window:** `2025-01-01` to `2025-08-31` ($37{,}837$ loads).
* **Validation Window:** `2025-09-01` to `2025-10-31` ($9{,}330$ holdout loads).
* This strictly mirrors the task of predicting November (`validation.csv`) and December (`december_chart_inputs.csv`).

### 3. Model Architecture & Multiplier Formulation
An error audit revealed that standard models suffered from spatial over-smoothing on transcontinental East-to-West hauls (e.g., applying regional Northeast discounts to $3{,}000$-mile hauls from Albany to LA).

**The Solution:**
1. Optimized a **CatBoost Regressor** using MAE loss directly on the dimensionless target multiplier:
   $$
   \hat{y}_i = \hat{z}_i \times (\text{distance}_i \times \text{quote\_signal}_i)
   $$
2. Engineered directional vectors (`delta_lat`, `delta_lon`, `compass_bearing`, `is_transcon`) and interaction terms (`dist_x_pickup_lat`, `dist_x_delivery_lat`) to prevent short-haul origin penalties from collapsing long-haul rates.

---

## 📂 Repository Structure

```
├── data/
│   ├── december-chart-inputs.csv            # Original benchmark inputs (31 daily runs for Lexington -> Fort Wayne)
│   ├── december_chart_inputs.csv            # Populated benchmark inputs with final predicted_rate values
│   ├── prepared_december.csv                # Preprocessed December dataset with engineered features
│   ├── prepared_train.csv                   # Preprocessed & label-noise-filtered training dataset (47,167 rows)
│   ├── prepared_validation.csv              # Preprocessed validation dataset with engineered features (12,000 rows)
│   ├── train-test.csv                       # Raw labeled historical development data (48,000 loads)
│   ├── validation.csv                       # Raw unlabelled validation data requiring predictions (12,000 loads)
│   └── validation-predictions-template.csv  # Submission template with target schema (load_id, predicted_rate)
├── data_experiments.ipynb                   # EDA notebook: data profiling, formula verification & diagnostics
├── data_preparation.py                      # Standalone data cleaning, spatial/temporal imputation & feature engineering script
├── output/
│   ├── feature_importance.png               # High-resolution CatBoost feature importance barplot (Multiplier model)
│   ├── statistical validation.png           # 1,000-sample bootstrap distributions & CI comparison plot
│   ├── validation_predictions.csv           # Final 12,000 test predictions in exact submission format
│   └── worst_predictions.csv                # Top residual outliers exported during root-cause error analysis
├── readme.md                                # Project documentation, reproduction steps, and Loom video link
├── requirements.txt                         # Exact Python environment dependencies
├── score.py                                 # Official assessment validator & December chart generator
├── scorer_results/
│   └── candidate_december.png               # Verified December 2025 price curve generated by score.py
└── train.py                                 # Unified end-to-end training, OOT evaluation, and inference script
```

---

## 🚀 Setup & Reproduction Instructions

### 1. Environment Installation
Clone the repository and install the required dependencies:

```bash
# Clone repository
git clone <your-repo-url>
cd <your-repo-name>

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution

You can run the pipeline sequentially:

```bash
# Step 1: Preprocess data and engineer features
python prepare_data.py

# Step 2: Train CatBoost Multiplier model, evaluate OOT performance, and export predictions
python train_and_predict.py
```

### 3. Run Assessment Scorer
Validate both submission files and generate `scorer_results/candidate_december.png`:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

---

## 📊 Evaluation & December Benchmark

### Validation Metrics Summary
| Metric | Baseline Formula ($\text{Dist} \times \text{Quote}$) | CatBoost (Direct Dollar) | CatBoost (Multiplier Architecture) |
| :--- | :---: | :---: | :---: |
| **Out-of-Time MAE** | $\$232.34$ | $\$108.71$ | **`$66.14`** |
| **Out-of-Time RMSE** | $\$675.63$ | $\$286.08$ | **`$209.42`** |
| **Out-of-Time $R^2$** | $0.7935$ | $0.9577$ | **`0.9773`** |
| **Linehaul MAPE** | $9.82\%$ | $4.10\%$ | **`2.50%`** |

### December Benchmark Prediction (`scorer_results/candidate_december.png`)
* **Scenario:** Lexington $\to$ Fort Wayne | 360.0 miles | Dry Van | 32,000 lbs | Dec 1 – Dec 31, 2025
* **Predicted Average Rate:** **`$703.85`** ($\approx \mathbf{\$1.955/\text{mile}}$), strictly adhering to physical transportation economics.
* **Volatility:** Tight $\pm 1.16\%$ bandwidth reflecting natural weekly freight cycles without artificial variance.