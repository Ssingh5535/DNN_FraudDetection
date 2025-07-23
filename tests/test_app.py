# tests/test_app.py

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from fastapi.testclient import TestClient  # noqa: E402
from app import app  # noqa: E402


client = TestClient(app)


def test_docs_available():
    """API docs should be up at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_smoke():
    """
    POST /predict with a minimal valid payload returns 200
    and correct JSON schema.
    """
    sample = {
        "transactions": [
            {
                "V1": 0.0,
                "V2": 0.0,
                "V3": 0.0,
                "V4": 0.0,
                "V5": 0.0,
                "V6": 0.0,
                "V7": 0.0,
                "V8": 0.0,
                "V9": 0.0,
                "V10": 0.0,
                "V11": 0.0,
                "V12": 0.0,
                "V13": 0.0,
                "V14": 0.0,
                "V15": 0.0,
                "V16": 0.0,
                "V17": 0.0,
                "V18": 0.0,
                "V19": 0.0,
                "V20": 0.0,
                "V21": 0.0,
                "V22": 0.0,
                "V23": 0.0,
                "V24": 0.0,
                "V25": 0.0,
                "V26": 0.0,
                "V27": 0.0,
                "V28": 0.0,
                "Amount": 0.0,
                "Time": 0.0,
            }
        ]
    }

    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200

    data = resp.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert "threshold" in data
