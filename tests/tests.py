import pytest
from src.ginesta.model import *
from ..src.predict import predict
from river import datasets


@pytest.fixture
def model_config():
    return {
        'optimizer': True,
        'optimizer_value': 0.001,
    }

@pytest.fixture
def input_features():
    return {
        'clouds': 60,
        'humidity': 75,
        'pressure': 1020.0,
        'temperature': 8.5,
        'wind': 5.2,
    }
    

def test_dataset_length():
    dataset = datasets.Bikes()
    dataset_length = sum(1 for _ in dataset)
    assert dataset_length == 182479