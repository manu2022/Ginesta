import json
from river import datasets


def test_dataset_shape():
    dataset = datasets.Bikes()
    shape = (dataset.n_samples, dataset.n_features)

    assert shape == (182470, 8)


def test_input_features_type_for_prediction():
    predict_object_path = "../assets/predict_object.json"

    with open(predict_object_path, "r") as file:
        predict_object = json.load(file)

    expected_types = {
        "clouds": int,
        "humidity": (int, float),
        "pressure": (int, float),
        "temperature": (int, float),
        "wind": (int, float),
        "station": str,
        "moment": str,
    }

    for feature, expected_type in expected_types.items():
        assert isinstance(predict_object[feature], expected_type)
