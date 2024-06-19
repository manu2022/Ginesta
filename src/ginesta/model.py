from river import compose
from river import preprocessing
from river import optim
from typing import Literal
from river import linear_model
from river import metrics
from river import evaluate
import pickle

allowed_features = Literal["clouds", "humidity", "pressure", "temperature", "wind"]


def model_pipeline(features_selection: list[allowed_features], model_config: dict):
    model = compose.Pipeline(
        compose.Select(*features_selection), preprocessing.StandardScaler()
    )

    if model_config.get("optimizer") is True:
        model |= linear_model.LinearRegression(
            optimizer=optim.SGD(model_config.get("optimizer_value"))
        )
    else:
        model |= linear_model.LinearRegression()

    return model


def evaluate_model(model, dataset, evaluate_sample=20_000):
    return evaluate.progressive_val_score(
        dataset, model, metrics.MAE(), print_every=evaluate_sample
    )


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
