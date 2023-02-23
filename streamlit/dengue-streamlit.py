import sys
import os
import inspect
import numpy as np
import pandas as pd 
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import plotly.express as px
from xgboost import XGBRegressor
import time

# # Helper functions 
def generate_uniform_distribution(min_val, max_val, size=1, seed=19):
    # create a uniform distribution object
    dist = uniform(loc=min_val, scale=max_val-min_val)
    return dist

# def plot_joint_distribution(data, var1, var2, title):
#     df = pd.DataFrame({var1: data[var1], var2: data[var2] *-1})
#     fig = px.scatter(df, x=var1, y=var2, marginal_x="histogram", marginal_y="histogram", title=title)
#     fig.show()
    

# def plot_time_series(y, y_hat, week_start_date, title):
#     df = pd.DataFrame({'y': y, 'y_hat': y_hat, 'week_start_date': week_start_date})
#     fig = px.line(df, x='week_start_date', y=['y', 'y_hat'], title=title,
#                   labels={'value': 'Value', 'week_of_year': 'Week of Year'})
#     fig.show()


# Plotting functions
# def generate_uniform_distribution(min_val, max_val, size=1, seed=19):
#     # create a uniform distribution object
#     dist = uniform(loc=min_val, scale=max_val-min_val)
#     return dist.rvs(size=size, random_state=seed)

def plot_joint_distribution(data, var1, var2, title):
    df = pd.DataFrame({var1: data[var1], var2: data[var2] *-1})
    fig = px.scatter(df, x=var1, y=var2, marginal_x="histogram", marginal_y="histogram", title=title)
    return fig

def plot_time_series(y, y_hat, week_start_date, title):
    df = pd.DataFrame({'Actual Dengue': y, 'Predicted Dengue': y_hat, 'week_start_date': week_start_date})
    fig = px.line(df, x='week_start_date', y=['Actual Dengue', 'Predicted Dengue'], title=title,
                  labels={'value': 'Value', 'week_of_year': 'Week of Year'})
    return fig



# Import data
url = 'https://raw.githubusercontent.com/john-adeojo/dengue-model/master/data/01_raw/dengue_features_train.csv'
train_features_df = pd.read_csv(url)

url = 'https://raw.githubusercontent.com/john-adeojo/dengue-model/master/data/01_raw/dengue_features_test.csv'
test_features_df = pd.read_csv(url)

url = 'https://raw.githubusercontent.com/john-adeojo/dengue-model/master/data/01_raw/dengue_labels_train.csv'
train_labels_df = pd.read_csv(url)

# Data pre-processing
df = train_features_df.merge(right=train_labels_df, left_on=["city", "year", "weekofyear"], right_on=["city", "year", "weekofyear"], how="left")
df['week_start_date'] = pd.to_datetime(df['week_start_date'])
test_features_df['week_start_date'] = pd.to_datetime(test_features_df['week_start_date'])

# Feature Engineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Feature engineering pipeline
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

def model_features(train_df, remove=[]):
    X = train_df.drop(columns=remove)
    return X
    

X = model_features(train_df=train_engineered_df, remove=remove)
y = train_engineered_df["total_cases"]



# Streamlit actions below -----------------------------------------------------

st.title("Predicting Dengue Fever Cases: Iquitos & San Juan")



# Model Training: Ridge Regressor
Ridge_tab, XGB_tab = st.tabs(["Ridge Regressor", "XGBoost Regressor"])

with Ridge_tab:
    st.header("Ridge Regressor")
    
    st.text("Set the minimum and maximum values for Alpha")
    min_val = st.number_input("Minimum Alpha", value=1)
    max_val = st.number_input("Maximum Alpha", value=10)

    args={
        "n_splits":5,
        "test_size":250,
        "gap": 50
        # "max_train_size":350
    }

    random_search_kwargs = {
        "estimator": Ridge(),
        "param_distributions": {"alpha": generate_uniform_distribution(min_val=min_val, max_val=max_val)},
        "n_iter": 500,
        "scoring": "neg_mean_absolute_error",
        "cv": TimeSeriesSplit(**args)
    }

    train_model = RandomizedSearchCV(**random_search_kwargs)
    
    
    if st.button('Train Ridge Regressor Model'):
        st.text("Model training in progress, this can take up to a minute to complete")
        ridge_random_search = train_model.fit(X, y)

        # Visualise Model: Ridge Regressor
        y_hat_iq = ridge_random_search.predict(X.loc[X.city_iq==1])
        y_hat_sj = ridge_random_search.predict(X.loc[X.city_sj==1])

        y_iq = df["total_cases"].loc[df.city == "iq"]
        y_sj = df["total_cases"].loc[df.city == "sj"]

        week_start_date_iq = df["week_start_date"].loc[df.city == "iq"]
        week_start_date_sj = df["week_start_date"].loc[df.city == "sj"]

        chart1 = plot_time_series(y=y_iq, y_hat=y_hat_iq, week_start_date=week_start_date_iq, title="Iquitos actuals vs model")
        chart2 = plot_time_series(y=y_sj, y_hat=y_hat_sj, week_start_date=week_start_date_sj, title="San Juan actuals vs model")
        
        st.plotly_chart(chart1)
        st.plotly_chart(chart2)

        # Plot results: Ridge Regressor 
        data = ridge_random_search.cv_results_
        var1 = "param_alpha"
        var2 = "mean_test_score"
        title= "Ridge Regressor: Alpha vs MAE"
        #plot_joint_distribution(data, var1, var2, title)
        st.plotly_chart(plot_joint_distribution(data, var1, var2, title))
        
        
# Model Training: XGBOOST Regressor
with XGB_tab:
    
    st.header("XGBoost Regressor")
    
    st.text("Set the minimum and maximum values for reg_lambda")
    min_val = st.number_input("Minimum reg_lambda", value=0)
    max_val = st.number_input("Maximum reg_lambda", value=10)
    
    random_search_kwargs = {
        "estimator": XGBRegressor(n_jobs=4),
        "param_distributions": {"reg_lambda": generate_uniform_distribution(min_val=min_val, max_val=max_val)},
        "n_iter": 500,
        "scoring": "neg_mean_absolute_error",
        "cv": TimeSeriesSplit(**args)
    }
    
    train_model = RandomizedSearchCV(**random_search_kwargs)

    if st.button('Train XGBoost Regressor Model'):
        st.text("Model training in progress, this can take up to a minute to 5 minutes to complete")
        xgboost_random_search = train_model.fit(X, y)

        # Visualise Model: XGBoost Regressor
        y_hat_iq = xgboost_random_search.predict(X.loc[X.city_iq==1])
        y_hat_sj = xgboost_random_search.predict(X.loc[X.city_sj==1])

        y_iq = df["total_cases"].loc[df.city == "iq"]
        y_sj = df["total_cases"].loc[df.city == "sj"]

        week_start_date_iq = df["week_start_date"].loc[df.city == "iq"]
        week_start_date_sj = df["week_start_date"].loc[df.city == "sj"]

        st.plotly_chart(plot_time_series(y=y_iq, y_hat=y_hat_iq, week_start_date=week_start_date_iq, title="Iquitos actuals vs model"))
        st.plotly_chart(plot_time_series(y=y_sj, y_hat=y_hat_sj, week_start_date=week_start_date_sj, title="San Juan actuals vs model"))
        
        # Plot Results: XGBoost Regressor
        data = xgboost_random_search.cv_results_
        var1 = "param_reg_lambda"
        var2 = "mean_test_score"
        title= "XGBRegressor: Reg_Lamba vs MAE"
        st.plotly_chart(plot_joint_distribution(data, var1, var2, title))