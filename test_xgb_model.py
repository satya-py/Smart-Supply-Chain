"""
Test XGBoost Model Loading
Run: python test_xgb_model.py
"""
import pickle
import pandas as pd
import sys

print("=" * 60)
print("Testing XGBoost Model Loading")
print("=" * 60)

try:
    model_path = r"D:\d_drive_project\googke solution challange\Models\model_scratch_xgb.pkl"
    print(f"\nLoading model from: {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print("✓ XGBoost model loaded successfully!")
    print(f"  Model type: {type(model)}")
    print(f"  Model attributes: {dir(model)[:5]}...")
    
    # Test prediction with dummy data
    print("\nTesting prediction with dummy data...")
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
    
    prediction = model.predict(dummy_data)
    probability = model.predict_proba(dummy_data)
    
    print(f"✓ Prediction successful!")
    print(f"  Predicted class: {prediction[0]}")
    print(f"  Probabilities: {probability[0]}")
    
    print("\n✅ XGBoost model test PASSED")
    
except Exception as e:
    print(f"\n❌ XGBoost model test FAILED")
    print(f"   Error: {str(e)}")
    sys.exit(1)
