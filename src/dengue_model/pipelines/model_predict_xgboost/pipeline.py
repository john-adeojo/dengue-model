"""
This is a boilerplate pipeline 'model_predict_xgboost'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import xgboost_predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=xgboost_predict,
                inputs= ["test_features_df", "test_engineered_df", "xgboost_random_search"],
                outputs="submission",
                name="xgboost_predict"
            )
        ]
    )
