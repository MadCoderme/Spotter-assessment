"""
Data Preparation & Feature Engineering Pipeline for Freight Rate Prediction
Authors: Machine Learning Engineering Candidate
Description: Cleans raw data, imputes missing coordinates/signals for December, 
             engineers directional vectors & interaction terms, filters training label noise,
             and exports prepared artifacts for model training.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def resolve_path(filename: str) -> str:
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
    raise FileNotFoundError(f"Required input file '{filename}' was not found.")


def engineer_features(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, imputes, and engineers physical, directional, and temporal features across combined data.
    """
    df = all_data.copy()
    df['date'] = pd.to_datetime(df['date'])

    # -------------------------------------------------------------------------
    # 1. PHYSICAL CLEANING & IMPUTATION
    # -------------------------------------------------------------------------
    # 1.1 Fix Negative Weights (Sign-inversion bug -> restores DOT 48,000 lb legal cap)
    df['weight'] = df['weight'].abs()
    
    # 1.2 Impute Missing Weights via Equipment-Class Median
    equip_medians = df.groupby('equipment')['weight'].transform('median')
    df['weight'] = df['weight'].fillna(equip_medians).fillna(32000.0)

    # 1.3 Construct Spatial Coordinate Lookup (Fills missing December Lat/Lon)
    city_lat_map = df.dropna(subset=['pickup_lat']).groupby('pickup')['pickup_lat'].median().to_dict()
    city_lon_map = df.dropna(subset=['pickup_lon']).groupby('pickup')['pickup_lon'].median().to_dict()

    df['pickup_lat'] = df['pickup_lat'].fillna(df['pickup'].map(city_lat_map))
    df['pickup_lon'] = df['pickup_lon'].fillna(df['pickup'].map(city_lon_map))
    df['delivery_lat'] = df['delivery_lat'].fillna(df['delivery'].map(city_lat_map))
    df['delivery_lon'] = df['delivery_lon'].fillna(df['delivery'].map(city_lon_map))

    # 1.4 Impute Temporal Market Index & Quote Signals (For missing December inputs)
    df['market_index'] = df.groupby('date')['market_index'].transform(lambda s: s.ffill().bfill())
    df['market_index'] = df['market_index'].ffill().bfill().fillna(1.0)

    df['lane'] = df['pickup'].astype(str) + "_to_" + df['delivery'].astype(str)
    lane_quote_map = df.dropna(subset=['quote_signal']).groupby('lane')['quote_signal'].median().to_dict()
    global_quote_median = df['quote_signal'].median()
    df['quote_signal'] = df['quote_signal'].fillna(df['lane'].map(lane_quote_map)).fillna(global_quote_median)

    # -------------------------------------------------------------------------
    # 2. CORE DOMAIN & DIRECTIONAL FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    # 2.1 Physical Baseline Products
    df['base_rate_dq'] = df['distance'] * df['quote_signal']
    df['base_rate_dqm'] = df['distance'] * df['quote_signal'] * df['market_index']
    df['ton_miles'] = (df['weight'] / 2000.0) * df['distance']

    # 2.2 Directional & Transcontinental Vectors
    df['delta_lat'] = df['delivery_lat'] - df['pickup_lat']
    df['delta_lon'] = df['delivery_lon'] - df['pickup_lon']
    df['is_westbound'] = (df['delta_lon'] < 0).astype(int)
    df['is_transcon'] = (df['distance'] > 2000).astype(int)
    df['compass_bearing'] = np.arctan2(df['delta_lat'], df['delta_lon'])

    # 2.3 Spatial-Distance Interaction Terms (Eliminates transcontinental under-predictions)
    df['dist_x_pickup_lat'] = df['distance'] * df['pickup_lat']
    df['dist_x_delivery_lat'] = df['distance'] * df['delivery_lat']

    # 2.4 Temporal Cyclic Features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear

    # 2.5 Categorical Formats for Tree Ensembles
    cat_cols = ['equipment', 'pickup', 'delivery', 'lane']
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df


def run_data_preparation():
    print("=" * 65)
    print("STARTING DATA PREPARATION & FEATURE PIPELINE")
    print("=" * 65)

    # 1. Ingest Raw Datasets
    train_file = resolve_path('train_test.csv' if os.path.exists(resolve_path('train_test.csv')) else 'train-test.csv')
    val_file = resolve_path('validation.csv')
    dec_file = resolve_path('december_chart_inputs.csv' if os.path.exists(resolve_path('december_chart_inputs.csv')) else 'december-chart-inputs.csv')

    print(f"Loading Raw Datasets:\n  • Train: {train_file}\n  • Val  : {val_file}\n  • Dec  : {dec_file}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    dec_df = pd.read_csv(dec_file)

    # 2. Tag splits and combine
    train_df['__split'] = 'train'
    val_df['__split'] = 'val'
    dec_df['__split'] = 'dec'
    all_data = pd.concat([train_df, val_df, dec_df], ignore_index=True)

    # 3. Apply Feature Engineering
    print("\nEngineering domain, directional, and spatial features...")
    processed_df = engineer_features(all_data)

    # 4. Partition back into splits
    train_clean = processed_df[processed_df['__split'] == 'train'].drop(columns=['__split']).copy()
    val_clean = processed_df[processed_df['__split'] == 'val'].drop(columns=['__split']).copy()
    dec_clean = processed_df[processed_df['__split'] == 'dec'].drop(columns=['__split']).copy()

    # 5. Filter Injected Synthetic Label Noise (TRAINING SET ONLY)
    print("\nAuditing and filtering training label anomalies...")
    train_clean['multiplier'] = train_clean['posted_rate'] / train_clean['base_rate_dq']
    valid_mask = (train_clean['multiplier'] >= 0.5) & (train_clean['multiplier'] <= 2.5)
    dropped_count = (~valid_mask).sum()

    print(f"  [✓] Filtered {dropped_count} synthetic 3x/5x label anomalies from training data.")
    train_clean = train_clean[valid_mask].drop(columns=['multiplier'])

    # 6. Sanity Checks & Assertions
    assert not train_clean['base_rate_dq'].isna().any(), "Train base_rate_dq contains NaNs!"
    assert not val_clean['base_rate_dq'].isna().any(), "Validation base_rate_dq contains NaNs!"
    assert not dec_clean['base_rate_dq'].isna().any(), "December base_rate_dq contains NaNs!"
    assert (dec_clean['base_rate_dq'] > 0).all(), "December base_rate_dq must be strictly positive!"

    # 7. Export Processed Artifacts
    out_dir = 'data' if os.path.exists('data') else '.'
    os.makedirs(out_dir, exist_ok=True)

    out_train_path = os.path.join(out_dir, 'prepared_train.csv')
    out_val_path = os.path.join(out_dir, 'prepared_validation.csv')
    out_dec_path = os.path.join(out_dir, 'prepared_december.csv')

    print("\nExporting prepared datasets:")
    train_clean.to_csv(out_train_path, index=False)
    val_clean.to_csv(out_val_path, index=False)
    dec_clean.to_csv(out_dec_path, index=False)

    print(f"  [✓] Saved: {out_train_path} {train_clean.shape}")
    print(f"  [✓] Saved: {out_val_path}   {val_clean.shape}")
    print(f"  [✓] Saved: {out_dec_path}   {dec_clean.shape}")
    print("=" * 65)
    print("DATA PREPARATION COMPLETE & VERIFIED")
    print("=" * 65)


if __name__ == "__main__":
    run_data_preparation()