from river import compose
from river import linear_model
from river import preprocessing
from river import optim
from river import feature_extraction
from river import stats
from typing import Union



def model_pipeline(features_selection: list[Union[
    'moment', 
    'station', 
    'clouds', 
    'humidity', 
    'pressure', 
    'temperature']],
    model_config: dict):
    
    model = compose.Pipeline(
        compose.Select(*features_selection),
        preprocessing.StandardScaler()
    )
    
    if model_config.get('optimizer') is True:
        model |= linear_model.LinearRegression(optimizer=optim.SGD(model_config.get('optimizer_value')))
    else:
        model |= linear_model.LinearRegression()
    
    return model


# def preprocess_model_data(model):
    
#     model |= preprocessing.StandardScaler()
    
#     return
#     model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))
    
#     return model
