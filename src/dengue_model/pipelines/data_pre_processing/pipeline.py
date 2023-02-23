"""
This is a boilerplate pipeline 'data_pre_processing'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import read_wrangle


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=read_wrangle,
                inputs=["dengue_features_train", "dengue_features_test", "dengue_labels_train"],
                outputs=["df", "test_features_df"],
                name="read_wrangle"
            )
        ]
    )
