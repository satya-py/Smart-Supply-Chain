"""
================================================================
Supply Chain Disruption Detection & Route Optimization API
================================================================
Team: AI/ML → Backend handoff

Required files in same folder:
    model_scratch_xgb.pkl     ← XGBoost classifier
    isolation_forest.pkl      ← Isolation Forest anomaly detector
    scaler.pkl                ← StandardScaler for IF features
    feature_importance.csv    ← top 10 feature names

Install:
    pip install fastapi uvicorn pandas numpy scikit-learn xgboost

Run:
    uvicorn main:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs
================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Supply Chain Disruption Detection API",
    description = (
        "Detects transit disruptions and recommends route optimizations "
        "using XGBoost (AUC=0.9922) + Isolation Forest anomaly detection. "
        "Built on the DataCo Smart Supply Chain dataset."
    ),
    version     = "1.0.0",
)

# Allow frontend / dashboard to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ──────────────────────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# Why: loading models once at startup is much faster than
#      loading on every request. Models stay in memory.
# ──────────────────────────────────────────────────────────────
print("Loading models...")

with open(r"D:\d_drive_project\googke solution challange\Models\model_scratch_xgb.pkl", "rb") as f:
    XGB_MODEL = pickle.load(f)

print("✓ XGBoost loaded")

try:
    with open(r"D:\d_drive_project\googke solution challange\AUTO ML\isolation_forest.pkl", "rb") as f:
        ISO_FOREST = pickle.load(f)
    print("✓ Isolation Forest loaded")
except Exception as e:
    print(f"⚠ Isolation Forest failed to load: {e}")
    print("   Anomaly detection will be disabled")
    ISO_FOREST = None

with open(r"D:\d_drive_project\googke solution challange\Models\scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

print("✓ Scaler loaded")

FEATURE_IMP = pd.read_csv(r"D:\d_drive_project\googke solution challange\feature_importance.csv")
TOP_FEATURES = FEATURE_IMP["variable"].head(10).tolist()

print(f"✓ XGBoost loaded")
print(f"✓ Top features: {TOP_FEATURES}")
print(f"✓ Models loaded successfully (ISO Forest: {'enabled' if ISO_FOREST else 'disabled'})")


# ──────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# Why: Pydantic models validate input automatically —
#      if a required field is missing or wrong type,
#      FastAPI returns a clear 422 error before your code runs
# ──────────────────────────────────────────────────────────────

class OrderInput(BaseModel):
    """
    Single order input for disruption prediction.
    All fields match the top 10 features from feature_importance.csv
    """
    # Core disruption signals (engineered features)
    delay_gap: float = Field(
        ...,
        description="actual_days - scheduled_days. Positive = late, Negative = early",
        example=2.0
    )
    is_delayed: int = Field(
        ...,
        description="1 if delay_gap > 0 else 0",
        example=1
    )
    delay_ratio: float = Field(
        ...,
        description="delay_gap / scheduled_days (normalized severity)",
        example=0.67
    )

    # Shipping features
    Type_TRANSFER: int = Field(
        ...,
        description="1 if payment type is bank transfer, else 0",
        example=0
    )

    # Geographic features
    Order_City: int = Field(
        ...,
        description="Label-encoded Order City",
        example=142
    )
    Order_State: int = Field(
        ...,
        description="Label-encoded Order State",
        example=23
    )
    Order_Country: int = Field(
        ...,
        description="Label-encoded Order Country",
        example=45
    )
    Order_Region: int = Field(
        ...,
        description="Label-encoded Order Region",
        example=7
    )
    Customer_City: int = Field(
        ...,
        description="Label-encoded Customer City",
        example=89
    )

    # Temporal features
    order_day: int = Field(
        ...,
        description="Day of month the order was placed (1-31)",
        example=15
    )

    class Config:
        json_schema_extra = {
            "example": {
                "delay_gap"     : 2.0,
                "is_delayed"    : 1,
                "delay_ratio"   : 0.67,
                "Type_TRANSFER" : 1,
                "Order_City"    : 142,
                "Order_State"   : 23,
                "Order_Country" : 45,
                "Order_Region"  : 7,
                "Customer_City" : 89,
                "order_day"     : 15,
            }
        }


class BatchInput(BaseModel):
    """Multiple orders for batch prediction."""
    orders: List[OrderInput] = Field(
        ...,
        description="List of orders to score",
        min_items=1,
        max_items=1000
    )


class PredictionResult(BaseModel):
    """Prediction result for a single order."""
    prediction          : int           # 0 = On Time, 1 = Late
    prediction_label    : str           # "ON TIME" or "LATE"
    probability_late    : float         # 0.0 to 1.0
    probability_ontime  : float         # 0.0 to 1.0
    is_anomaly          : bool          # True if Isolation Forest flagged it
    risk_score          : float         # 0-100 composite risk score
    risk_level          : str           # HIGH / MEDIUM / LOW
    recommended_action  : str           # specific route recommendation
    reason              : str           # why this action was recommended
    confidence          : str           # HIGH / MEDIUM / LOW confidence in prediction


class BatchResult(BaseModel):
    """Batch prediction results."""
    total_orders        : int
    late_count          : int
    ontime_count        : int
    high_risk_count     : int
    medium_risk_count   : int
    low_risk_count      : int
    anomaly_count       : int
    predictions         : List[PredictionResult]


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def order_to_dataframe(order: OrderInput) -> pd.DataFrame:
    """
    Converts an OrderInput object to a pandas DataFrame
    with column names matching what the model was trained on.
    """
    # Map Pydantic field names back to original feature names
    feature_map = {
        "delay_gap"     : "delay_gap",
        "is_delayed"    : "is_delayed",
        "delay_ratio"   : "delay_ratio",
        "Type_TRANSFER" : "Type_TRANSFER",
        "Order_City"    : "Order City",
        "Order_State"   : "Order State",
        "Order_Country" : "Order Country",
        "Order_Region"  : "Order Region",
        "Customer_City" : "Customer City",
        "order_day"     : "order_day",
    }

    data = {}
    for pydantic_field, model_feature in feature_map.items():
        data[model_feature] = getattr(order, pydantic_field)

    # Ensure column order matches TOP_FEATURES exactly
    df = pd.DataFrame([data])
    df = df.reindex(columns=TOP_FEATURES, fill_value=0)
    return df


def compute_risk_score(prob_late: float, if_score: float,
                       delay_gap: float, is_transfer: int) -> float:
    """
    Composite risk score (0-100) combining 4 signals:
      1. XGBoost probability  → max 40 pts
      2. Isolation Forest     → max 30 pts
      3. Delay gap magnitude  → max 20 pts
      4. Transfer payment     → max 10 pts
    """
    score  = prob_late * 40
    score += min(max(0, -if_score * 100), 30)
    score += min(max(0, delay_gap) * 5, 20)
    if is_transfer == 1:
        score += 10
    return round(min(score, 100), 2)


def get_risk_level(score: float) -> str:
    """Convert numeric score to HIGH/MEDIUM/LOW tier."""
    if score >= 55: return "HIGH"
    if score >= 30: return "MEDIUM"
    return "LOW"


def get_recommendation(delay_gap: float, is_transfer: int,
                       order_region: int, order_day: int,
                       prob_late: float, risk_level: str) -> tuple:
    """
    Generate a specific route recommendation based on
    the dominant risk signal for this order.
    Returns (action, reason) tuple.
    """
    # Transfer payment with high late probability
    if is_transfer == 1 and prob_late > 0.7:
        return (
            "Verify payment clearance before dispatch",
            "High-risk bank transfer — confirm funds before releasing shipment"
        )

    # Severe delay (2+ days over schedule)
    if delay_gap >= 2:
        return (
            "URGENT: Switch to Same Day or Air freight immediately",
            f"Shipment is {delay_gap:.0f} days behind schedule — immediate escalation needed"
        )

    # Moderate delay (1 day over schedule)
    if delay_gap >= 1:
        return (
            "Switch to Second Class or expedited shipping",
            f"Shipment is {delay_gap:.0f} day behind schedule — upgrade shipping tier"
        )

    # High risk but no delay yet — preemptive action
    if risk_level == "HIGH" and delay_gap < 1:
        return (
            "Pre-stage inventory at nearest distribution centre",
            "High risk score despite low current delay — preemptive positioning recommended"
        )

    # Weekend order pattern
    if order_day in [6, 7, 13, 14, 20, 21, 27, 28]:
        return (
            "Pre-stage inventory — weekend order detected",
            "Weekend orders show higher delay rates — advance preparation reduces risk"
        )

    # Medium risk — monitor
    if risk_level == "MEDIUM":
        return (
            "Monitor order — flag for next logistics review",
            "Moderate risk detected — check carrier status within 24 hours"
        )

    # Low risk — no action
    return (
        "No action required — order on track",
        f"Risk score below threshold — standard shipping process"
    )


def get_confidence(prob_late: float) -> str:
    """Model confidence level based on prediction probability."""
    p = max(prob_late, 1 - prob_late)  # distance from 0.5
    if p >= 0.90: return "HIGH"
    if p >= 0.70: return "MEDIUM"
    return "LOW"


def score_order(order: OrderInput) -> PredictionResult:
    """
    Full scoring pipeline for a single order:
    1. Convert to DataFrame
    2. XGBoost prediction
    3. Isolation Forest anomaly score
    4. Composite risk score
    5. Route recommendation
    """
    df = order_to_dataframe(order)

    # XGBoost prediction
    prob      = XGB_MODEL.predict_proba(df)[0]
    pred      = int(XGB_MODEL.predict(df)[0])
    prob_late = float(prob[1])

    # Isolation Forest anomaly detection
    if ISO_FOREST is not None and SCALER is not None:
        df_scaled  = SCALER.transform(df)
        if_pred    = ISO_FOREST.predict(df_scaled)[0]
        if_score   = float(ISO_FOREST.decision_function(df_scaled)[0])
        is_anomaly = bool(if_pred == -1)
    else:
        # Fallback: no anomaly detection
        if_score = 0.0
        is_anomaly = False

    # Composite risk score
    risk_score = compute_risk_score(
        prob_late, if_score, order.delay_gap, order.Type_TRANSFER
    )
    risk_level = get_risk_level(risk_score)

    # Route recommendation
    action, reason = get_recommendation(
        order.delay_gap, order.Type_TRANSFER,
        order.Order_Region, order.order_day,
        prob_late, risk_level
    )

    return PredictionResult(
        prediction          = pred,
        prediction_label    = "LATE" if pred == 1 else "ON TIME",
        probability_late    = round(prob_late, 4),
        probability_ontime  = round(float(prob[0]), 4),
        is_anomaly          = is_anomaly,
        risk_score          = risk_score,
        risk_level          = risk_level,
        recommended_action  = action,
        reason              = reason,
        confidence          = get_confidence(prob_late),
    )


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms API is running."""
    return {
        "status"     : "running",
        "api"        : "Supply Chain Disruption Detection",
        "version"    : "1.0.0",
        "model_auc"  : 0.9922,
        "endpoints"  : ["/predict", "/predict/batch", "/health", "/features", "/docs"],
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check — confirms all models are loaded."""
    return {
        "status"           : "healthy",
        "xgboost_loaded"   : XGB_MODEL is not None,
        "iso_forest_loaded": ISO_FOREST is not None,
        "scaler_loaded"    : SCALER is not None,
        "top_features"     : TOP_FEATURES,
        "n_features"       : len(TOP_FEATURES),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(order: OrderInput):
    """
    Predict disruption risk for a single shipment order.

    Returns:
    - prediction (0=OnTime, 1=Late)
    - probability of being late
    - anomaly detection result
    - composite risk score (0-100)
    - risk level (HIGH / MEDIUM / LOW)
    - specific route recommendation
    - reason for recommendation
    - confidence level
    """
    try:
        return score_order(order)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict disruption risk for multiple orders (up to 1000 at a time).

    Returns aggregated summary + individual predictions for each order.
    Useful for nightly batch scoring of all pending shipments.
    """
    try:
        results = [score_order(order) for order in batch.orders]

        late_count   = sum(1 for r in results if r.prediction == 1)
        high_count   = sum(1 for r in results if r.risk_level == "HIGH")
        medium_count = sum(1 for r in results if r.risk_level == "MEDIUM")
        low_count    = sum(1 for r in results if r.risk_level == "LOW")
        anom_count   = sum(1 for r in results if r.is_anomaly)

        return BatchResult(
            total_orders      = len(results),
            late_count        = late_count,
            ontime_count      = len(results) - late_count,
            high_risk_count   = high_count,
            medium_risk_count = medium_count,
            low_risk_count    = low_count,
            anomaly_count     = anom_count,
            predictions       = results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/features", tags=["Model Info"])
def get_features():
    """
    Returns the top 10 features the model uses,
    with their importance percentages from H2O AutoML.
    Useful for frontend to know what fields to send.
    """
    features_list = []
    for _, row in FEATURE_IMP.head(10).iterrows():
        features_list.append({
            "rank"       : int(_ + 1),
            "feature"    : row["variable"],
            "importance" : round(float(row["percentage"]) * 100, 2),
        })
    return {
        "model"    : "XGBoost (AUC=0.9922)",
        "n_features": len(features_list),
        "features" : features_list,
    }


@app.get("/model/info", tags=["Model Info"])
def model_info():
    """Returns full model performance metrics and system summary."""
    return {
        "model_name"         : "XGBoost Classifier",
        "automl_benchmark"   : {"model": "H2O GBM", "auc": 0.9926},
        "scratch_model"      : {
            "auc"            : 0.9922,
            "f1_score"       : 0.9716,
            "accuracy"       : 0.9682,
            "recall_late"    : 0.9922,
            "precision_late" : 0.9517,
            "gap_to_automl"  : 0.0004,
            "features_used"  : 10,
        },
        "anomaly_detector"   : {
            "model"          : "Isolation Forest",
            "contamination"  : 0.05,
            "extra_catches"  : 17,
            "combined_recall": 0.9931,
        },
        "training_data"      : {
            "rows"           : 180519,
            "train_rows"     : 144415,
            "test_rows"      : 36104,
            "dataset"        : "DataCo Smart Supply Chain",
        },
        "risk_thresholds"    : {
            "HIGH"   : "score >= 55",
            "MEDIUM" : "score >= 30",
            "LOW"    : "score < 30",
        },
    }