"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.5
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import initialise, ridge_regressor_train, xgboost_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=initialise,
                inputs=None,
                outputs=["generate_uniform_distribution", "plot_joint_distribution", "plot_time_series", "model_features"],
                name="initialise"
            ), 
            node(
                func=ridge_regressor_train, 
                inputs=["generate_uniform_distribution", "plot_joint_distribution", "plot_time_series", "model_features", "train_engineered_df", "df"],
                outputs=["iq_timeseries_ridge_reg", "sj_timeseries_ridge_reg", "ridge_mae_plot"],
                name="ridge_regressor_train"
            ), 
            node(
                func=xgboost_train,
                inputs=["generate_uniform_distribution", "plot_joint_distribution", "plot_time_series", "model_features", "train_engineered_df", "df"],
                outputs=["xgboost_mae_plot", "iq_timeseries_xgboost", "sj_timeseries_xgboost"],
                name="xgboost_train"
            )
        ]
    )
