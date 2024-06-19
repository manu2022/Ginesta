# def hello() -> str:
#     return "Hello from ginesta!"

from pathlib import Path
from datetime import datetime
from model import model_pipeline, evaluate_model, save_model, load_model
from predict import forecast
from river import datasets

model_config = {
    "optimizer": True,
    "optimizer_value": 0.001,
}


class BikeRentalPredictor:
    def __init__(self):
        if Path("model.pkl").exists():
            print("Model found! Loading model...")
            self.model = load_model("model.pkl")
        else:
            print("Model not found. Running pipeline")
            self.model = model_pipeline(
                features_selection=[
                    "clouds",
                    "humidity",
                    "pressure",
                    "temperature",
                    "wind",
                ],
                model_config=model_config,
            )
            evaluate_model(self.model, dataset=datasets.Bikes(), evaluate_sample=20_000)
            save_model(self.model, "model.pkl")

        self.predict_input = {
            "clouds": 60,
            "humidity": 75,
            "pressure": 1020.0,
            "temperature": 8.5,
            "wind": 5.2,
            "station": "metro-canal-du-midi",
            "moment": datetime(2016, 10, 5, 10, 57, 18),
        }

        prediction = forecast(self.model, self.predict_input)
        print(f"Forecasted value: {prediction}")


BikeRentalPredictor()
