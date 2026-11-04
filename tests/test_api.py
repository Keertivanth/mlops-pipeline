
# Import TestClient to send simulated HTTP requests
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

# Reusable test client for all test functions
client = TestClient(app)


def test_health_endpoint():
    """
    Verify that the application responds to a health-check request.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
    """
    Verify that the API returns a valid Iris prediction.
    """
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    # Send a simulated POST request with JSON input
    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    # Match the response keys returned by the current API
    assert "predicted_class" in result
    assert "class_id" in result

    assert result["predicted_class"] in [
        "setosa", "versicolor", "virginica"
    ]
    assert result["class_id"] in [0, 1, 2]
