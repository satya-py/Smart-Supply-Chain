"""
Test FastAPI ML Endpoint
Run: python test_fastapi_endpoint.py
"""
import requests
import json

print("=" * 60)
print("Testing FastAPI ML Endpoint")
print("=" * 60)

# Test health endpoint first
print("\n1. Testing /health endpoint...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    exit(1)

# Test predict endpoint
print("\n2. Testing /predict endpoint...")
payload = {
    "delay_gap": 2.0,
    "is_delayed": 1,
    "delay_ratio": 0.67,
    "Type_TRANSFER": 0,
    "Order_City": 142,
    "Order_State": 23,
    "Order_Country": 45,
    "Order_Region": 7,
    "Customer_City": 89,
    "order_day": 15
}

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=payload,
        timeout=5
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Prediction successful!")
        print(f"   Prediction: {result['prediction_label']}")
        print(f"   Probability Late: {result['probability_late']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Risk Score: {result['risk_score']}")
        print(f"   Recommended Action: {result['recommended_action']}")
        print(f"   Is Anomaly: {result['is_anomaly']}")
        print(f"   Confidence: {result['confidence']}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 60)
print("✅ FastAPI endpoint test complete!")
print("=" * 60)
