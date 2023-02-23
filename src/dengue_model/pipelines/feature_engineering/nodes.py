"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.5
"""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame(), test_features_df: pd.DataFrame()):

    class FeatureEngineering:

        def __init__(self, data):
            self.data = data.copy()

        def replace_na(self, exclude=[]):
            num_cols = self.data.select_dtypes(include=['int', 'float']).columns.difference(exclude)
            means = self.data[num_cols].mean()
            self.data[num_cols] = self.data[num_cols].fillna(means)

        def binning(self, variable, num_buckets):
            labels = [i for i in range(num_buckets)]
            self.data[variable+'_binned'] = pd.cut(self.data[variable], num_buckets, labels=labels)

        def normalise(self, exclude=[]):
            num_cols = self.data.select_dtypes(include=['int', 'float']).columns.difference(exclude)
            self.data[num_cols] = (self.data[num_cols] - self.data[num_cols].min()) / (self.data[num_cols].max() - self.data[num_cols].min())

        def standardise(self, exclude=[]):
            num_cols = self.data.select_dtypes(include=['int', 'float']).columns.difference(exclude)
            scaler = StandardScaler()
            self.data[num_cols] = scaler.fit_transform(self.data[num_cols])

        def one_hot_encode(self, exclude=[]):
            obj_cols = self.data.select_dtypes(include=['object']).columns.difference(exclude)
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoded = pd.DataFrame(encoder.fit_transform(self.data[obj_cols]).toarray(), columns=encoder.get_feature_names_out(obj_cols))
            self.data = pd.concat([self.data.drop(obj_cols, axis=1), encoded], axis=1)

        def return_data(self):
            return self.data


    # Feature engineering on train
    exclude = ["year", "total_cases", "week_start_date"]
    fe = FeatureEngineering(data=df)
    fe.replace_na(exclude=exclude)
    fe.binning(variable="weekofyear", num_buckets=4)
    fe.normalise(exclude=exclude)
    fe.one_hot_encode(exclude=exclude)
    train_engineered_df = fe.return_data()

    # Feature engineering on test
    exclude = ["year", "week_start_date"]
    fe = FeatureEngineering(data=test_features_df)
    fe.replace_na(exclude=exclude)
    fe.binning(variable="weekofyear", num_buckets=4)
    fe.normalise(exclude=exclude)
    fe.one_hot_encode(exclude=exclude)
    test_engineered_df = fe.return_data()

    train_engineered_df.sort_values(by="week_start_date", ascending=True, inplace=True)
    remove = ["total_cases", "week_start_date", "year", "weekofyear_binned"]

    
    return test_engineered_df, train_engineered_df