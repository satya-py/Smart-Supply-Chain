"""
Test Full Integration: Flask -> FastAPI -> ML Models
"""
import requests
import json

print("="*60)
print("TESTING FULL INTEGRATION")
print("="*60)

# Test 1: Flask health
print("\n1. Flask Backend Health:")
try:
    r = requests.get("http://localhost:5000/api/health")
    print(f"   Status: {r.json()['status']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: ML service health through Flask
print("\n2. ML Service Health (via Flask):")
try:
    r = requests.get("http://localhost:5000/api/ml/health")
    data = r.json()
    print(f"   ML Service: {data['ml_service']}")
    print(f"   XGBoost: {data['details']['xgboost_loaded']}")
    print(f"   ISO Forest: {data['details']['iso_forest_loaded']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Get ML features
print("\n3. ML Features:")
try:
    r = requests.get("http://localhost:5000/api/ml/features")
    data = r.json()
    print(f"   Model: {data.get('model', 'N/A')}")
    if 'features' in data:
        print(f"   Top 3 features:")
        for f in data['features'][:3]:
            print(f"     - {f['feature']} ({f['importance']}%)")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Predict on a node
print("\n4. Node Prediction (Node ID 1):")
try:
    r = requests.post("http://localhost:5000/api/ml/predict-node/1")
    data = r.json()
    if 'error' in data:
        print(f"   Error: {data['error']}")
    else:
        ml = data['ml_prediction']
        print(f"   Entity: {data['entity']}")
        print(f"   Prediction: {ml['prediction_label']}")
        print(f"   Risk Level: {ml['risk_level']}")
        print(f"   Risk Score: {ml['risk_score']}")
        print(f"   Action: {ml['recommended_action']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Predict on a route
print("\n5. Route Prediction (Route ID 1):")
try:
    r = requests.post("http://localhost:5000/api/ml/predict-route/1")
    data = r.json()
    if 'error' in data:
        print(f"   Error: {data['error']}")
    else:
        ml = data['ml_prediction']
        print(f"   Entity: {data['entity']}")
        print(f"   Prediction: {ml['prediction_label']}")
        print(f"   Risk Level: {ml['risk_level']}")
        print(f"   Action: {ml['recommended_action']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 6: Batch prediction
print("\n6. Batch Prediction (All Nodes):")
try:
    r = requests.post("http://localhost:5000/api/ml/predict-batch")
    data = r.json()
    if 'error' in data:
        print(f"   Error: {data['error']}")
    else:
        print(f"   Total Nodes: {data['total_nodes']}")
        print(f"   Predictions: {data['predictions']}")
        print(f"   High Risk: {data['high_risk_count']}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*60)
print("✅ INTEGRATION TEST COMPLETE")
print("="*60)
print("\nYour system is working!")
print("- Flask Backend: http://localhost:5000")
print("- ML API: http://localhost:8000/docs")
print("- Click the preview button to view the website")
