"""
Test Isolation Forest Model Loading
Run: python test_isolation_forest.py
"""
import pickle
import pandas as pd
import numpy as np
import sys

print("=" * 60)
print("Testing Isolation Forest Model Loading")
print("=" * 60)

try:
    model_path = r"D:\d_drive_project\googke solution challange\AUTO ML\isolation_forest.pkl"
    print(f"\nLoading model from: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print("✓ Isolation Forest model loaded successfully!")
    print(f"  Model type: {type(model)}")
    
    # Test prediction with dummy data
    print("\nTesting anomaly detection with dummy data...")
    dummy_data = pd.DataFrame({
        "delay_gap": [2.0],
        "is_delayed": [1],
        "delay_ratio": [0.67],
        "Type_TRANSFER": [0],
        "Order City": [142],
        "Order State": [23],
        "Order Country": [45],
        "Order Region": [7],
        "Customer City": [89],
        "order_day": [15]
    })
    
    # Need to scale data first
    scaler_path = r"D:\d_drive_project\googke solution challange\Models\scaler.pkl"
    print(f"\nLoading scaler from: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✓ Scaler loaded successfully!")
    
    # Scale the data
    scaled_data = scaler.transform(dummy_data)
    print(f"✓ Data scaled successfully!")
    
    # Test anomaly detection
    prediction = model.predict(scaled_data)
    decision_score = model.decision_function(scaled_data)
    
    print(f"✓ Anomaly detection successful!")
    print(f"  Prediction: {prediction[0]} (1=normal, -1=anomaly)")
    print(f"  Decision score: {decision_score[0]:.4f}")
    
    print("\n✅ Isolation Forest model test PASSED")
    
except Exception as e:
    print(f"\n❌ Isolation Forest model test FAILED")
    print(f"   Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
