import math
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import main


VALID_INPUT = {
    "panel_area_m2": 20.0,
    "sunshine_hours": 7.0,
    "temp_c": 6.0,
    "dust_days": 50.0,
    "panel_age_yr": 2.0,
    "cloud_cover_pct": 25.0,
    "roof_type": "flat",
    "area": "Old City",
}


@pytest.fixture
def service(monkeypatch):
    model = Mock()
    model.predict.return_value = [12.5]
    load = Mock(return_value=model)
    monkeypatch.setattr(main.joblib, "load", load)
    with TestClient(main.app) as client:
        yield client, model
        load.assert_called_once_with(main.MODEL_PATH)


def test_health(service):
    client, model = service
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    model.predict.assert_not_called()


def test_predict_accepts_boundary_values(service):
    client, model = service
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert response.json() == {"prediction": 12.5}
    model.predict.assert_called_once()
    features = model.predict.call_args.args[0]
    assert list(features.columns) == list(VALID_INPUT)
    assert features.to_dict("records") == [VALID_INPUT]


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("dust_days", 50.1, "less_than_equal"),
        ("temp_c", 5.9, "greater_than_equal"),
        ("record_id", "R001", "extra_forbidden"),
        ("area", None, "missing"),
    ],
)
def test_invalid_input_never_reaches_model(service, field, value, error_type):
    client, model = service
    payload = VALID_INPUT.copy()
    if error_type == "missing":
        del payload[field]
    else:
        payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", field]
    assert error["type"] == error_type
    model.predict.assert_not_called()


def test_predict_with_committed_model():
    with TestClient(main.app) as client:
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert math.isfinite(prediction)
    assert prediction >= 0
