import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def test_independant_models(
    models: list,
    features: pd.DataFrame,
    target: pd.DataFrame,
    subset_size: int = 10000,
    test_split: float = 0.2,
    subset: bool = True,
) -> pd.DataFrame:
    """Tests target-independant models on a subset of the data and returns the results.

    Args:
        models (list): Models to be tested
        features (pd.DataFrame): Input features for training
        target (pd.DataFrame): Target to be predicted
        subset_size (int, optional): Subset size. Defaults to 10000.

    Returns:
        pd.DataFrame: _description_
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_split, random_state=42
    )

    if subset:
        random.seed(42)
        subset_indices = random.sample(range(len(X_train)), subset_size)
        X_train, y_train = (
            X_train.iloc[subset_indices],
            y_train.iloc[subset_indices],
        )

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


def find_top_models(
    results: pd.DataFrame, metric: str = "R2", n: int = 5
) -> pd.DataFrame:
    """Finds the top n models by a given metric.

    Args:
        results (pd.DataFrame): Results dataframe
        metric (str): Metric to sort by
        n (int, optional): Number of top models to return. Defaults to 5.

    Returns:
        pd.DataFrame: Top n models by the given metric
    """
    return results.sort_values(by=metric, ascending=False).head(n)
