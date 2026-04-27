# ML Model Testing & Integration Guide

## 📋 Overview
This document explains how to test the ML models individually and run the complete system.

## 🧪 Test Files Created

### 1. Individual Model Tests
- **`test_xgb_model.py`** - Tests XGBoost model loading and prediction
- **`test_isolation_forest.py`** - Tests Isolation Forest anomaly detection
- **`test_scaler.py`** - Tests StandardScaler transformation

### 2. Comprehensive Tests
- **`test_all_models.py`** - Tests all models together (recommended)
- **`test_fastapi_endpoint.py`** - Tests the FastAPI server endpoints

## 🚀 How to Run

### Step 1: Activate Virtual Environment
```powershell
cd "d:\d_drive_project\googke solution challange"
.\env\Scripts\activate
```

### Step 2: Install Dependencies (First Time Only)
```powershell
pip install fastapi uvicorn pandas numpy scikit-learn xgboost requests
```

### Step 3: Run Individual Tests
```powershell
# Test all models together (recommended)
python test_all_models.py

# Test individual models
python test_xgb_model.py
python test_isolation_forest.py
python test_scaler.py

# Test FastAPI endpoints (server must be running)
python test_fastapi_endpoint.py
```

## 🖥️ Running the Complete System

### Option 1: Using the Batch File (Easiest)
```powershell
.\start_servers.bat
```
This will:
- Activate the virtual environment
- Install required packages
- Start both ML FastAPI server (port 8000) and Flask backend (port 5000)
- Open two terminal windows

### Option 2: Manual Startup

**Terminal 1 - ML FastAPI Server:**
```powershell
cd "d:\d_drive_project\googke solution challange\NOTE BOOKS"
uvicorn fast_api_server:app --reload --port 8000
```

**Terminal 2 - Flask Backend:**
```powershell
cd "d:\d_drive_project\googke solution challange\Full_project\supply"
python run.py
```

## 🔗 Access Points

- **Frontend Application**: http://localhost:5000
- **ML API Documentation**: http://localhost:8000/docs
- **ML Health Check**: http://localhost:8000/health

## ✅ Verification Checklist

After starting the servers, verify everything works:

1. **Check ML Server Health:**
   ```powershell
   python test_fastapi_endpoint.py
   ```

2. **Expected Output:**
   ```
   ✓ XGBoost loaded
   ✓ Isolation Forest loaded
   ✓ Scaler loaded
   ✓ Models loaded successfully (ISO Forest: enabled)
   ```

3. **Test Prediction:**
   - Visit http://localhost:8000/docs
   - Try the `/predict` endpoint with sample data

## 📊 Model Files Location

- **XGBoost Model**: `Models\model_scratch_xgb.pkl`
- **Isolation Forest**: `AUTO ML\isolation_forest.pkl`
- **Scaler**: `Models\scaler.pkl`
- **Feature Importance**: `feature_importance.csv`

## 🔧 Troubleshooting

### Models Not Loading
1. Check file paths in `fast_api_server.py` (lines 65-84)
2. Run `python test_all_models.py` to verify models load correctly
3. Check if virtual environment is activated

### Port Already in Use
- **Port 8000**: Task Manager → End Python processes running uvicorn
- **Port 5000**: Task Manager → End Python processes running Flask

### Missing Dependencies
```powershell
.\env\Scripts\activate
pip install fastapi uvicorn pandas numpy scikit-learn xgboost requests flask flask-socketio
```

## 📝 Notes

- The FastAPI server loads models at startup (lines 63-89 in `fast_api_server.py`)
- Models are loaded once and stay in memory for fast predictions
- The Flask backend proxies ML requests to FastAPI via HTTP (see `backend/routes/api_ml.py`)
- Both servers must be running for full ML integration

## 🎯 Quick Start
```powershell
# One command to test everything
.\env\Scripts\activate
python test_all_models.py

# Start the complete system
.\start_servers.bat
```
