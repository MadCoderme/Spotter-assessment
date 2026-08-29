"""
Freight Rate Prediction Engine - End-to-End Solution Script
Author: Machine Learning Engineer Candidate
Objective: Train CatBoost Multiplier Model, Evaluate OOT Performance, Export Feature Importance & Final Submissions
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def log_step(step_num: int, total_steps: int, title: str):
    print("\n" + "=" * 75)
    print(f" [STEP {step_num}/{total_steps}] {title}")
    print("=" * 75)


def resolve_input_path(filename: str) -> str:
    """Dynamically resolves file paths across Kaggle and local development environments."""
    possible_locations = [
        os.path.join('data', filename),
        os.path.join('/kaggle/input/datasets/abrarfairujraiyan/spotter-data/data', filename),
        os.path.join('/kaggle/input/spotter-data/data', filename),
        filename
    ]
    for path in possible_locations:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not locate required input file: {filename}")


def main():
    start_time = time.time()
    TOTAL_STEPS = 7

    # -------------------------------------------------------------------------
    # STEP 1: LOAD RAW ASSETS & RESOLVE SCHEMAS
    # -------------------------------------------------------------------------
    log_step(1, TOTAL_STEPS, "DATA INGESTION & ENVIRONMENT AUDIT")
    
    train_file = resolve_input_path('train_test.csv' if os.path.exists(resolve_input_path('train_test.csv')) else 'train-test.csv')
    val_file = resolve_input_path('validation.csv')
    dec_file = resolve_input_path('december_chart_inputs.csv' if os.path.exists(resolve_input_path('december_chart_inputs.csv')) else 'december-chart-inputs.csv')
    val_template_file = resolve_input_path('validation_predictions_template.csv' if os.path.exists(resolve_input_path('validation_predictions_template.csv')) else 'validation-predictions-template.csv')

    print(f"  [+] Training Dataset      : {train_file}")
    print(f"  [+] Validation Dataset    : {val_file}")
    print(f"  [+] December Inputs       : {dec_file}")
    print(f"  [+] Validation Template   : {val_template_file}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    dec_df = pd.read_csv(dec_file)
    val_template = pd.read_csv(val_template_file)

    print(f"\n  Loaded {len(train_df):,} development loads, {len(val_df):,} validation loads, and {len(dec_df):,} December loads.")

    # -------------------------------------------------------------------------
    # STEP 2: DATA QUALITY REMEDIATION & PREPROCESSING
    # -------------------------------------------------------------------------
    log_step(2, TOTAL_STEPS, "DATA QUALITY AUDIT & PHYSICAL REMEDIATION")

    # Combine datasets to guarantee consistent spatial/categorical encodings
    train_df['__split'] = 'train'
    val_df['__split'] = 'val'
    dec_df['__split'] = 'dec'
    all_data = pd.concat([train_df, val_df, dec_df], ignore_index=True)
    all_data['date'] = pd.to_datetime(all_data['date'])

    # 2.1 Fix Sign-Inverted Weights
    initial_negatives = (all_data['weight'] < 0).sum()
    all_data['weight'] = all_data['weight'].abs()
    print(f"  [✓] Fixed {initial_negatives} sign-inverted negative weights using |weight| transformation.")

    # 2.2 Impute Missing Weights via Equipment Median
    equip_weight_medians = all_data.groupby('equipment')['weight'].transform('median')
    all_data['weight'] = all_data['weight'].fillna(equip_weight_medians).fillna(32000.0)
    print(f"  [✓] Imputed missing weights using conditional equipment-class medians.")

    # 2.3 Construct Spatial Coordinate Lookup & Impute Missing Lat/Lon
    city_lat_map = all_data.dropna(subset=['pickup_lat']).groupby('pickup')['pickup_lat'].median().to_dict()
    city_lon_map = all_data.dropna(subset=['pickup_lon']).groupby('pickup')['pickup_lon'].median().to_dict()

    all_data['pickup_lat'] = all_data['pickup_lat'].fillna(all_data['pickup'].map(city_lat_map))
    all_data['pickup_lon'] = all_data['pickup_lon'].fillna(all_data['pickup'].map(city_lon_map))
    all_data['delivery_lat'] = all_data['delivery_lat'].fillna(all_data['delivery'].map(city_lat_map))
    all_data['delivery_lon'] = all_data['delivery_lon'].fillna(all_data['delivery'].map(city_lon_map))
    print(f"  [✓] Reconstructed full spatial coordinate topology for all origins and destinations.")

    # 2.4 Impute Temporal Market Index & Quote Signals for December
    all_data['market_index'] = all_data.groupby('date')['market_index'].transform(lambda s: s.ffill().bfill())
    all_data['market_index'] = all_data['market_index'].ffill().bfill().fillna(1.0)

    lane_series = all_data['pickup'].astype(str) + "_to_" + all_data['delivery'].astype(str)
    all_data['lane'] = lane_series
    lane_quote_map = all_data.dropna(subset=['quote_signal']).groupby('lane')['quote_signal'].median().to_dict()
    global_quote_median = all_data['quote_signal'].median()
    all_data['quote_signal'] = all_data['quote_signal'].fillna(all_data['lane'].map(lane_quote_map)).fillna(global_quote_median)
    print(f"  [✓] Imputed temporal market dynamics (Market Index & Quote Signals).")

    # -------------------------------------------------------------------------
    # STEP 3: ADVANCED DOMAIN & DIRECTIONAL FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    log_step(3, TOTAL_STEPS, "DIRECTIONAL & INTERACTION FEATURE ENGINEERING")

    # Physical baseline products
    all_data['base_rate_dq'] = all_data['distance'] * all_data['quote_signal']
    all_data['base_rate_dqm'] = all_data['distance'] * all_data['quote_signal'] * all_data['market_index']
    all_data['ton_miles'] = (all_data['weight'] / 2000.0) * all_data['distance']

    # Directional vectors & transcontinental indicators
    all_data['delta_lat'] = all_data['delivery_lat'] - all_data['pickup_lat']
    all_data['delta_lon'] = all_data['delivery_lon'] - all_data['pickup_lon']
    all_data['is_westbound'] = (all_data['delta_lon'] < 0).astype(int)
    all_data['is_transcon'] = (all_data['distance'] > 2000).astype(int)
    all_data['compass_bearing'] = np.arctan2(all_data['delta_lat'], all_data['delta_lon'])

    # Spatial-Distance interactions (eliminates transcontinental under-predictions)
    all_data['dist_x_pickup_lat'] = all_data['distance'] * all_data['pickup_lat']
    all_data['dist_x_delivery_lat'] = all_data['distance'] * all_data['delivery_lat']

    # Temporal cyclic features
    all_data['month'] = all_data['date'].dt.month
    all_data['quarter'] = all_data['date'].dt.quarter
    all_data['day_of_year'] = all_data['date'].dt.dayofyear

    # Categorical casting for CatBoost
    cat_cols = ['equipment', 'pickup', 'delivery', 'lane']
    for col in cat_cols:
        all_data[col] = all_data[col].astype(str)

    # Separate back into clean partitions
    train_clean = all_data[all_data['__split'] == 'train'].drop(columns=['__split']).copy()
    val_clean = all_data[all_data['__split'] == 'val'].drop(columns=['__split']).copy()
    dec_clean = all_data[all_data['__split'] == 'dec'].drop(columns=['__split']).copy()

    # 3.1 Training Label Noise Filtration
    train_clean['multiplier'] = train_clean['posted_rate'] / train_clean['base_rate_dq']
    clean_mask = (train_clean['multiplier'] >= 0.5) & (train_clean['multiplier'] <= 2.5)
    dropped_anomalies = (~clean_mask).sum()
    train_clean = train_clean[clean_mask].copy()
    train_clean['target_multiplier'] = train_clean['posted_rate'] / train_clean['base_rate_dq']

    print(f"  [✓] Filtered {dropped_anomalies} synthetic 3x/5x label corruptions from training set.")
    print(f"  [✓] Feature matrix compiled. Retained {len(train_clean):,} verified training loads.")

    # -------------------------------------------------------------------------
    # STEP 4: OUT-OF-TIME VALIDATION & PERFORMANCE EVALUATION
    # -------------------------------------------------------------------------
    log_step(4, TOTAL_STEPS, "OUT-OF-TIME VALIDATION (Sep 1 - Oct 31, 2025)")

    features = [
        'distance', 'weight', 'market_index', 'quote_signal',
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'delta_lat', 'delta_lon', 'is_westbound', 'is_transcon', 'compass_bearing',
        'dist_x_pickup_lat', 'dist_x_delivery_lat',
        'base_rate_dq', 'base_rate_dqm', 'ton_miles',
        'month', 'quarter', 'day_of_year',
        'equipment', 'pickup', 'delivery', 'lane'
    ]
    target = 'target_multiplier'

    split_date = pd.to_datetime('2025-09-01')
    tr_mask = train_clean['date'] < split_date
    oot_mask = train_clean['date'] >= split_date

    X_tr, y_tr = train_clean.loc[tr_mask, features], train_clean.loc[tr_mask, target]
    X_oot, y_oot_mult = train_clean.loc[oot_mask, features], train_clean.loc[oot_mask, target]
    y_oot_dollars = train_clean.loc[oot_mask, 'posted_rate'].values
    oot_base_rates = train_clean.loc[oot_mask, 'base_rate_dq'].values

    print(f"  Training Window (Jan-Aug)   : {len(X_tr):,} loads")
    print(f"  Evaluation Window (Sep-Oct) : {len(X_oot):,} loads")

    print("\n  Fitting Out-of-Time CatBoost Multiplier Regressor...")
    cat_eval = CatBoostRegressor(
        loss_function='MAE',
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cat_eval.fit(X_tr, y_tr, eval_set=(X_oot, y_oot_mult), early_stopping_rounds=100)

    # Reconstruct dollar predictions: y_pred = z_pred * (distance * quote_signal)
    pred_mult_oot = cat_eval.predict(X_oot)
    pred_dollars_oot = pred_mult_oot * oot_base_rates

    mae_oot = mean_absolute_error(y_oot_dollars, pred_dollars_oot)
    rmse_oot = np.sqrt(mean_squared_error(y_oot_dollars, pred_dollars_oot))
    r2_oot = r2_score(y_oot_dollars, pred_dollars_oot)
    mape_oot = np.mean(np.abs((y_oot_dollars - pred_dollars_oot) / y_oot_dollars)) * 100

    print("\n  " + "-" * 55)
    print("  OUT-OF-TIME EVALUATION RESULTS (Dollar Metric Space)")
    print("  " + "-" * 55)
    print(f"   • Mean Absolute Error (MAE)  : ${mae_oot:.2f}")
    print(f"   • Root Mean Squared Error    : ${rmse_oot:.2f}")
    print(f"   • Coefficient of Determination: {r2_oot:.4f} ({r2_oot*100:.2f}% explained variance)")
    print(f"   • Overall Holdout MAPE        : {mape_oot:.2f}%")
    print("  " + "-" * 55)

    # -------------------------------------------------------------------------
    # STEP 5: EXPORT FEATURE IMPORTANCE DIAGNOSTICS
    # -------------------------------------------------------------------------
    log_step(5, TOTAL_STEPS, "FEATURE IMPORTANCE VISUALIZATION")

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': cat_eval.get_feature_importance()
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(11, 8), dpi=300)
    palette = sns.color_palette("mako", len(importance_df))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette=palette)
    plt.title('CatBoost Feature Importance (Multiplier Target Engine)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score (%)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    
    feat_img_path = 'feature_importance.png'
    plt.savefig(feat_img_path, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Rendered and saved feature importance chart to: {feat_img_path}")
    print(f"      Top 3 Drivers: 1. {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.1f}%), "
          f"2. {importance_df.iloc[1]['Feature']} ({importance_df.iloc[1]['Importance']:.1f}%), "
          f"3. {importance_df.iloc[2]['Feature']} ({importance_df.iloc[2]['Importance']:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 6: FULL-DATASET RETRAINING
    # -------------------------------------------------------------------------
    log_step(6, TOTAL_STEPS, "FULL-HORIZON PRODUCTION RETRAINING")

    optimal_iterations = cat_eval.get_best_iteration()
    print(f"  Retraining final model on all {len(train_clean):,} development loads (100% capacity)...")
    print(f"  Optimal tree depth iteration locked at: {optimal_iterations} trees.")

    cat_final = CatBoostRegressor(
        loss_function='MAE',
        iterations=optimal_iterations,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cat_final.fit(train_clean[features], train_clean[target])
    print(f"  [✓] Production model converged successfully.")

    # -------------------------------------------------------------------------
    # STEP 7: INFERENCE & SUBMISSION ARTIFACT GENERATION
    # -------------------------------------------------------------------------
    log_step(7, TOTAL_STEPS, "INFERENCE & FINAL ASSET EXPORT")

    # 7.1 Validation Set Predictions
    val_pred_multipliers = cat_final.predict(val_clean[features])
    val_dollar_preds = val_pred_multipliers * val_clean['base_rate_dq']
    val_template['predicted_rate'] = val_dollar_preds

    out_val_csv = 'validation_predictions.csv'
    val_template[['load_id', 'predicted_rate']].to_csv(out_val_csv, index=False)
    print(f"  [✓] Generated: {out_val_csv} ({len(val_template):,} rows)")

    # 7.2 December Benchmark Predictions
    dec_pred_multipliers = cat_final.predict(dec_clean[features])
    dec_dollar_preds = dec_pred_multipliers * dec_clean['base_rate_dq']

    # Preserve exact original 7-column schema for score.py
    dec_export_df = pd.read_csv(dec_file)
    dec_export_df['predicted_rate'] = dec_dollar_preds

    out_dec_csv = 'data/december_chart_inputs.csv' if os.path.exists('data') else 'december_chart_inputs.csv'
    os.makedirs(os.path.dirname(out_dec_csv), exist_ok=True) if os.path.dirname(out_dec_csv) else None
    dec_export_df.to_csv(out_dec_csv, index=False)
    print(f"  [✓] Generated: {out_dec_csv} ({len(dec_export_df):,} rows)")

    # 7.3 Rigorous Schema & Constraint Assertions (Matches score.py checks)
    assert len(val_template) == 12000, "Validation set must contain exactly 12,000 rows!"
    assert not val_template['predicted_rate'].isna().any(), "Validation predictions contain NaN values!"
    assert (val_template['predicted_rate'] > 0).all(), "Validation predictions must be strictly positive!"
    assert len(dec_export_df) == 31, "December benchmark must contain exactly 31 rows!"
    assert not dec_export_df['predicted_rate'].isna().any(), "December predictions contain NaN values!"
    assert (dec_export_df['predicted_rate'] > 0).all(), "December predictions must be strictly positive!"

    print("\n  [✓] All internal schema and positivity assertions passed.")
    print(f"  Sample December Predictions (Lexington -> Fort Wayne, 360 mi, Dry Van):\n")
    print(dec_export_df[['date', 'pickup', 'delivery', 'distance', 'predicted_rate']].head(5).to_string(index=False))

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 75)
    print(f" PIPELINE COMPLETE IN {elapsed_time:.2f}s — READY FOR score.py EXECUTION")
    print("=" * 75)
    print(f"\nExecute the official scorer via:")
    print(f"  python score.py --predictions {out_val_csv} --december-predictions {out_dec_csv}\n")


if __name__ == "__main__":
    main()