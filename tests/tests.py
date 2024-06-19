# from src.ginesta.model import model_pipeline, load_model, save_model, evaluate_model
from river.datasets.base import Dataset
from river.compose.pipeline import Pipeline


def dataset_shape(dataset: Dataset):
    shape = (dataset.n_samples, dataset.n_features)

    assert shape == (182470, 8)


def pipeline_is_valid(pipeline: Pipeline):
    assert type(pipeline) == Pipeline
