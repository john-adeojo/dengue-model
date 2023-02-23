"""
This is a boilerplate pipeline 'model_predict_xgboost'
generated using Kedro 0.18.5
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor



def xgboost_predict(test_features_df: pd.DataFrame(), test_engineered_df: pd.DataFrame(), xgboost_random_search):
    
    # Helper function to remove unwanted features
    def model_features(train_df, remove=[]):
        X = train_df.drop(columns=remove)
        return X
    
    remove = ["week_start_date", "year", "weekofyear_binned"]
    X_test = model_features(train_df=test_engineered_df, remove=remove)
    y_hat = np.round(np.maximum(xgboost_random_search.predict(X_test),0)).astype(int)
    submission = test_features_df[["city", "year", "weekofyear"]].copy()
    submission["total_cases"] = y_hat
    
    return submission