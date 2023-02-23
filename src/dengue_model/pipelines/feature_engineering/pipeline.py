"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_engineering,
                inputs=["df", "test_features_df"],
                outputs=["test_engineered_df", "train_engineered_df"],
                name="feature_engineering"
            )
        ]
    )
