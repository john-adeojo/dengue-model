"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.5
"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import plotly.express as px
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def initialise() -> callable:
    def generate_uniform_distribution(min_val, max_val, size=1, seed=19):
        # create a uniform distribution object
        dist = uniform(loc=min_val, scale=max_val-min_val)
        return dist

    def plot_joint_distribution(data, var1, var2, title):
        df = pd.DataFrame({var1: data[var1], var2: data[var2] *-1})
        fig = px.scatter(df, x=var1, y=var2, marginal_x="histogram", marginal_y="histogram", title=title)
        return fig


    def plot_time_series(y, y_hat, week_start_date, title):
        df = pd.DataFrame({'y': y, 'y_hat': y_hat, 'week_start_date': week_start_date})
        fig = px.line(df, x='week_start_date', y=['y', 'y_hat'], title=title,
                      labels={'value': 'Value', 'week_of_year': 'Week of Year'})
        return fig
        
    def model_features(train_df, remove=[]):
        X = train_df.drop(columns=remove)
        return X
        
    return generate_uniform_distribution, plot_joint_distribution, plot_time_series, model_features

def ridge_regressor_train (generate_uniform_distribution, plot_joint_distribution, plot_time_series, model_features, train_engineered_df: pd.DataFrame(), df: pd.DataFrame()):
    
    # process features and labels for training
    remove = ["total_cases", "week_start_date", "year", "weekofyear_binned"]
    X = model_features(train_df=train_engineered_df, remove=remove)
    y = train_engineered_df["total_cases"]
    
    
    #Initialise model and random search cv
    args={
        "n_splits":5,
        "test_size":250,
        "gap": 50
    }

    random_search_kwargs = {
        "estimator": Ridge(),
        "param_distributions": {"alpha": generate_uniform_distribution(min_val=500, max_val=2000)},
        "n_iter": 500,
        "scoring": "neg_mean_absolute_error",
        "cv": TimeSeriesSplit(**args)
    }

    train_model = RandomizedSearchCV(**random_search_kwargs)
    
    # Train model
    ridge_random_search = train_model.fit(X, y)
    
    
    # plot training results
    data = ridge_random_search.cv_results_
    var1 = "param_alpha"
    var2 = "mean_test_score"
    title= "Ridge Regressor: Param vs std MAE"
    ridge_mae_plot = plot_joint_distribution(data, var1, var2, title)
    
    
    # Visualise model against training data 
    y_hat_iq = ridge_random_search.predict(X.loc[X.city_iq==1])
    y_hat_sj = ridge_random_search.predict(X.loc[X.city_sj==1])

    y_iq = df["total_cases"].loc[df.city == "iq"]
    y_sj = df["total_cases"].loc[df.city == "sj"]

    week_start_date_iq = df["week_start_date"].loc[df.city == "iq"]
    week_start_date_sj = df["week_start_date"].loc[df.city == "sj"]


    iq_timeseries_ridge_reg = plot_time_series(y=y_iq, y_hat=y_hat_iq, week_start_date=week_start_date_iq, title="Iq actuals vs model")
    
    sj_timeseries_ridge_reg = plot_time_series(y=y_sj, y_hat=y_hat_sj, week_start_date=week_start_date_sj, title="Sj actuals vs model")
    
    
    return iq_timeseries_ridge_reg, sj_timeseries_ridge_reg, ridge_mae_plot

def xgboost_train (generate_uniform_distribution, plot_joint_distribution, plot_time_series, model_features, train_engineered_df: pd.DataFrame(), df: pd.DataFrame()):
    
    # process features and labels for training
    remove = ["total_cases", "week_start_date", "year", "weekofyear_binned"]
    X = model_features(train_df=train_engineered_df, remove=remove)
    y = train_engineered_df["total_cases"]
    
    # Initialise xgboost and randomised search
    args={
        "n_splits":5,
        "test_size":250,
        "gap": 50
    }
    
    random_search_kwargs = {
        "estimator": XGBRegressor(n_jobs=4),
        "param_distributions": {"reg_lambda": generate_uniform_distribution(min_val=100, max_val=2000)},
        "n_iter": 500,
        "scoring": "neg_mean_absolute_error",
        "cv": TimeSeriesSplit(**args)
    }

    train_model = RandomizedSearchCV(**random_search_kwargs)
    
    # Train model
    xgboost_random_search = train_model.fit(X, y)
    
    # Plot training results
    data = xgboost_random_search.cv_results_
    var1 = "param_reg_lambda"
    var2 = "mean_test_score"
    title= "XGBRegressor: Param vs std MAE"
    xgboost_mae_plot = plot_joint_distribution(data, var1, var2, title)
    
    # visualise model against training data 
    y_hat_iq = xgboost_random_search.predict(X.loc[X.city_iq==1])
    y_hat_sj = xgboost_random_search.predict(X.loc[X.city_sj==1])

    y_iq = df["total_cases"].loc[df.city == "iq"]
    y_sj = df["total_cases"].loc[df.city == "sj"]

    week_start_date_iq = df["week_start_date"].loc[df.city == "iq"]
    week_start_date_sj = df["week_start_date"].loc[df.city == "sj"]


    iq_timeseries_xgboost = plot_time_series(y=y_iq, y_hat=y_hat_iq, week_start_date=week_start_date_iq, title="Iq actuals vs model")
    sj_timeseries_xgboost = plot_time_series(y=y_sj, y_hat=y_hat_sj, week_start_date=week_start_date_sj, title="Sj actuals vs model")
    
    return xgboost_mae_plot, iq_timeseries_xgboost, sj_timeseries_xgboost