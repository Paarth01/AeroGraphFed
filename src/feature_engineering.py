import pandas as pd
import numpy as np


def create_lag_features(df):

    df["pm25_lag1"] = df.groupby("country")["pm25"].shift(1)
    df["pm25_lag2"] = df.groupby("country")["pm25"].shift(2)

    return df


def pollution_growth(df):

    df["pm25_growth"] = df.groupby("country")["pm25"].pct_change()

    return df


def rolling_features(df):

    df["pm25_roll_mean"] = df.groupby("country")["pm25"].rolling(3).mean().reset_index(0,drop=True)

    return df


def population_features(df):

    df["log_population"] = np.log1p(df["population"])

    return df


def feature_pipeline(df):

    df = create_lag_features(df)
    df = pollution_growth(df)
    df = rolling_features(df)
    df = population_features(df)

    df = df.dropna()

    return df