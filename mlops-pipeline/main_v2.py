import os
from contextlib import asynccontextmanager
from typing import Callable

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"
)
MODEL_URI = "models:/iris-classifier@champion"


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


def create_app(model_loader: Callable = load_mlflow_model):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load MLflow champion model.\n{ex}"
            ) from ex

        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        champion = client.get_model_version_by_alias(
            name="iris-classifier",
            alias="champion",
        )
        app.state.model_version = champion.version
        print(f"Loaded iris-classifier version {app.state.model_version}")
        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLflow",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def root():
        return {"message": "API is ready to serve..."}

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": "iris-classifier",
            "model_version": app.state.model_version,
        }

    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]]
        prediction = int(request.app.state.model.predict(features)[0])

        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction],
        }

    return app


app = create_app()
