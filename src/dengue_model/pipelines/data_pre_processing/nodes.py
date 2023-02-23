"""
This is a boilerplate pipeline 'data_pre_processing'
generated using Kedro 0.18.5
"""

import pandas as pd

def read_wrangle(dengue_features_train: pd.DataFrame, dengue_features_test: pd.DataFrame, dengue_labels_train: pd.DataFrame):
    
    train_features_df = dengue_features_train
    test_features_df = dengue_features_test
    train_labels_df = dengue_labels_train
    
    df = train_features_df.merge(right=train_labels_df, left_on=["city", "year", "weekofyear"], right_on=["city", "year", "weekofyear"], how="left")
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    test_features_df['week_start_date'] = pd.to_datetime(test_features_df['week_start_date'])
    
    
    return df, test_features_df 