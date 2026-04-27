"""
Test Scaler Loading
Run: python test_scaler.py
"""
import pickle
import pandas as pd
import sys

print("=" * 60)
print("Testing Scaler Loading")
print("=" * 60)

try:
    scaler_path = r"D:\d_drive_project\googke solution challange\Models\scaler.pkl"
    print(f"\nLoading scaler from: {scaler_path}")
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✓ Scaler loaded successfully!")
    print(f"  Scaler type: {type(scaler)}")
    
    # Test transformation with dummy data
    print("\nTesting transformation with dummy data...")
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
    
    print(f"  Original data shape: {dummy_data.shape}")
    print(f"  Original data:\n{dummy_data.head()}")
    
    transformed = scaler.transform(dummy_data)
    print(f"\n✓ Transformation successful!")
    print(f"  Transformed shape: {transformed.shape}")
    print(f"  Transformed data:\n{transformed}")
    
    print("\n✅ Scaler test PASSED")
    
except Exception as e:
    print(f"\n❌ Scaler test FAILED")
    print(f"   Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
