from .model import model_pipeline
from datetime import datetime

def predict(model, input_features):
    return model.predict_one(input_features)

