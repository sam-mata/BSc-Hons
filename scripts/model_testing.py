import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingRegressor


def test_independant_models(
    models: list,
    features: pd.DataFrame,
    target: pd.DataFrame,
    split_year: int = 2050,
) -> tuple:
    """Tests target-independant models on data split by year and returns the results.

    Args:
        models (list): Models to be tested
        features (pd.DataFrame): Input features for training
        target (pd.DataFrame): Target to be predicted
        split_year (int, optional): Year to split the data on. Defaults to 2050.
        subset (bool, optional): Whether to use a subset of the training data. Defaults to False.
        subset_size (int, optional): Size of the subset if used. Defaults to 10000.

    Returns:
        tuple: (DataFrame with model performance results, Dict of feature importances for each model)
    """
    X_train, X_test, y_train, y_test = split_data_by_year(features, target, split_year)

    results = {}

    for model in models:
        model_name = type(model).__name__

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"MSE": mse, "MAE": mae, "R2": r2}

    results = pd.DataFrame(results).T
    results = results.sort_values(by="R2", ascending=False)

    return results


def split_data_by_year(features: pd.DataFrame, targets: pd.DataFrame, split_year: int):
    """
    Splits the data into training and testing sets based on a given year.

    Args:
        features (pd.DataFrame): Input features.
        targets (pd.DataFrame): Target variables.
        split_year (int): The year to split the data on.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    train_mask = features["year"] < split_year
    test_mask = features["year"] >= split_year

    X_train = features[train_mask]
    X_test = features[test_mask]
    y_train = targets[train_mask]
    y_test = targets[test_mask]

    return X_train, X_test, y_train, y_test


def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)
