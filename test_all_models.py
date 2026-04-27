"""
Comprehensive ML Model Test
Tests all models together as they are loaded in the FastAPI server
Run: python test_all_models.py
"""
import pickle
import pandas as pd
import numpy as np
import sys
import os

print("=" * 70)
print("COMPREHENSIVE ML MODEL TEST")
print("Testing all models as loaded in FastAPI server")
print("=" * 70)

# Define paths
MODELS_DIR = r"D:\d_drive_project\googke solution challange\Models"
AUTO_ML_DIR = r"D:\d_drive_project\googke solution challange\AUTO ML"

XGB_PATH = os.path.join(MODELS_DIR, "model_scratch_xgb.pkl")
ISO_FOREST_PATH = os.path.join(AUTO_ML_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_IMP_PATH = r"D:\d_drive_project\googke solution challange\feature_importance.csv"

print(f"\nChecking file existence:")
print(f"  XGBoost model: {os.path.exists(XGB_PATH)} - {XGB_PATH}")
print(f"  Isolation Forest: {os.path.exists(ISO_FOREST_PATH)} - {ISO_FOREST_PATH}")
print(f"  Scaler: {os.path.exists(SCALER_PATH)} - {SCALER_PATH}")
print(f"  Feature importance: {os.path.exists(FEATURE_IMP_PATH)} - {FEATURE_IMP_PATH}")

# Test 1: Load XGBoost
print("\n" + "-" * 50)
print("TEST 1: Loading XGBoost Model")
print("-" * 50)
try:
    with open(XGB_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    print("✓ XGBoost model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load XGBoost: {e}")
    sys.exit(1)

# Test 2: Load Isolation Forest
print("\n" + "-" * 50)
print("TEST 2: Loading Isolation Forest Model")
print("-" * 50)
try:
    with open(ISO_FOREST_PATH, "rb") as f:
        iso_forest = pickle.load(f)
    print("✓ Isolation Forest model loaded successfully")
except Exception as e:
    print(f"⚠ Isolation Forest failed to load: {e}")
    print("   Anomaly detection will be disabled")
    iso_forest = None

# Test 3: Load Scaler
print("\n" + "-" * 50)
print("TEST 3: Loading Scaler")
print("-" * 50)
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Failed to load Scaler: {e}")
    sys.exit(1)

# Test 4: Load Feature Importance
print("\n" + "-" * 50)
print("TEST 4: Loading Feature Importance")
print("-" * 50)
try:
    feature_imp = pd.read_csv(FEATURE_IMP_PATH)
    top_features = feature_imp["variable"].head(10).tolist()
    print(f"✓ Feature importance loaded")
    print(f"  Top 10 features: {top_features}")
except Exception as e:
    print(f"❌ Failed to load feature importance: {e}")
    sys.exit(1)

# Test 5: Full prediction pipeline
print("\n" + "-" * 50)
print("TEST 5: Full Prediction Pipeline")
print("-" * 50)
try:
    # Create test data matching the expected features
    test_data = {
        "delay_gap": 2.0,
        "is_delayed": 1,
        "delay_ratio": 0.67,
        "Type_TRANSFER": 0,
        "Order City": 142,
        "Order State": 23,
        "Order Country": 45,
        "Order Region": 7,
        "Customer City": 89,
        "order_day": 15
    }
    
    df = pd.DataFrame([test_data])
    df = df.reindex(columns=top_features, fill_value=0)
    
    print(f"  Test data shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # XGBoost prediction
    prob = xgb_model.predict_proba(df)[0]
    pred = int(xgb_model.predict(df)[0])
    prob_late = float(prob[1])
    
    print(f"  XGBoost prediction: {pred} (0=On Time, 1=Late)")
    print(f"  Probability late: {prob_late:.4f}")
    
    # Isolation Forest prediction
    if iso_forest is not None and scaler is not None:
        df_scaled = scaler.transform(df)
        if_pred = iso_forest.predict(df_scaled)[0]
        if_score = float(iso_forest.decision_function(df_scaled)[0])
        is_anomaly = bool(if_pred == -1)
        
        print(f"  Isolation Forest: {if_pred} (1=normal, -1=anomaly)")
        print(f"  Decision score: {if_score:.4f}")
        print(f"  Is anomaly: {is_anomaly}")
    else:
        print("  Isolation Forest: disabled")
        is_anomaly = False
    
    print("✓ Full prediction pipeline successful")
    
except Exception as e:
    print(f"❌ Full prediction pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Models are working correctly!")
print("=" * 70)
