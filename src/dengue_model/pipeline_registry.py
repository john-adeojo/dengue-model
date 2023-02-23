"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from dengue_model.pipelines import data_pre_processing as dp
from dengue_model.pipelines import feature_engineering as fe
from dengue_model.pipelines import model_predict_xgboost as xgboost_predict
from dengue_model.pipelines import model_training as training

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_pre_processing_pipeline = dp.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    model_predict_xgboost_pipeline = xgboost_predict.create_pipeline()
    model_training_pipeline = training.create_pipeline()
    
    return {
    
    "__default__":data_pre_processing_pipeline + feature_engineering_pipeline + model_training_pipeline,
        "data_pre_processing":data_pre_processing_pipeline + feature_engineering_pipeline,
        "model_traing_pipeline":model_training_pipeline
        
    
    }
    

