import os
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


class FraudNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 30,
        hidden_size: int = 224,
        n_layers: int = 3,
        dropout_rate: float = 0.4394849794542452,
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODEL_DIR = os.getenv("MODEL_DIR", "model")

with open(os.path.join(MODEL_DIR, "scaler_final.pkl"), "rb") as f:
    scaler = pickle.load(f)

rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))

device = torch.device("cpu")
dnn = FraudNet().to(device)
dnn.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "dnn_model_final.pt"),
        map_location=device
    )
)
dnn.eval()

with open(os.path.join(MODEL_DIR, "threshold_final.pkl"), "rb") as f:
    THRESHOLD = pickle.load(f)

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)
SCORE_HIST = Histogram(
    "fraud_api_predicted_score",
    "Predicted fraud probability distribution"
)

app = FastAPI(title="Fraud Detection Ensemble API")


@app.get("/metrics")
def metrics() -> Response:
    """
    Expose Prometheus metrics.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    return df.drop(["Time"], axis=1).values


def ensemble_score(X_raw: np.ndarray) -> np.ndarray:
    X = scaler.transform(X_raw)
    p_rf = rf.predict_proba(X)[:, 1]
    with torch.no_grad():
        logits = dnn(torch.tensor(X, dtype=torch.float32).to(device))
        p_dnn = torch.sigmoid(logits).cpu().numpy().squeeze()
    return 0.5 * p_rf + 0.5 * p_dnn


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float


class TxBatch(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=1)


@app.post("/predict")
async def predict(batch: TxBatch, request: Request) -> dict:
    start = time.time()
    df = pd.DataFrame([t.dict() for t in batch.transactions])
    X_raw = preprocess(df)
    probs = ensemble_score(X_raw)
    preds = (probs > THRESHOLD).tolist()
    for p in probs:
        SCORE_HIST.observe(p)
    latency = time.time() - start
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)
    REQUEST_COUNT.labels(request.method, request.url.path, "200").inc()
    return {"predictions": preds, "probabilities": probs.tolist(), "threshold": THRESHOLD}
