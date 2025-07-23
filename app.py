# app.py

import os
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# ==== 1. Define your DNN architecture ====

class FraudNet(nn.Module):
    def __init__(self, input_dim=30, hidden_size=224, n_layers=3, dropout_rate=0.4394849794542452):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_size
            layers += [
                nn.Linear(in_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==== 2. Load artifacts ====

MODEL_DIR = os.getenv("MODEL_DIR", "model")

# 2a) Scaler
with open(os.path.join(MODEL_DIR, "scaler_final.pkl"), "rb") as f:
    scaler = pickle.load(f)

# 2b) RandomForest
rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))

# 2c) DNN
device = torch.device("cpu")
dnn = FraudNet(input_dim=30, hidden_size=224, n_layers=3, dropout_rate=0.4394849794542452).to(device)
dnn.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "dnn_model_final.pt"),
    map_location=device
))
dnn.eval()

# 2d) Threshold
with open(os.path.join(MODEL_DIR, "threshold_final.pkl"), "rb") as f:
    THRESHOLD = pickle.load(f)

# ==== 3. Preprocessing function ====

def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    # drop only the Time column; keep V1–V28, Amount, LogAmount
    return df.drop(["Time"], axis=1).values

# ==== 4. Ensemble scoring helper ====

def ensemble_score(X_raw: np.ndarray) -> np.ndarray:
    X = scaler.transform(X_raw)
    # RF probability
    p_rf = rf.predict_proba(X)[:, 1]
    # DNN probability
    with torch.no_grad():
        logits = dnn(torch.tensor(X, dtype=torch.float32).to(device))
        p_dnn = torch.sigmoid(logits).cpu().numpy().squeeze()
    # 50/50 blend
    return 0.5 * p_rf + 0.5 * p_dnn

# ==== 5. Pydantic models ====

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float

class TxBatch(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=1)

# ==== 6. FastAPI setup ====

app = FastAPI(title="Fraud Detection Ensemble API")

@app.post("/predict")
def predict(batch: TxBatch):
    # Build DataFrame from incoming transactions
    df = pd.DataFrame([t.dict() for t in batch.transactions])
    # Preprocess & score
    X_raw = preprocess(df)
    probs = ensemble_score(X_raw)
    preds = (probs > THRESHOLD).tolist()
    return {
        "predictions": preds,
        "probabilities": probs.tolist(),
        "threshold": THRESHOLD
    }
