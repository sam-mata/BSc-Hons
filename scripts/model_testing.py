import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone


def test_independant_models(
    models: list, features: pd.DataFrame, target: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """Tests target-independant models using TimeSeriesSplit cross-validation and returns the results.

    Args:
        models (list): Models to be tested
        features (pd.DataFrame): Input features for training
        target (pd.Series): Target to be predicted
        n_splits (int, optional): Number of splits for TimeSeriesSplit. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with model performance results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {
        model.__class__.__name__: {"MSE": [], "MAE": [], "R2": []} for model in models
    }

    for train_index, test_index in tscv.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        for model in models:
            model_name = model.__class__.__name__
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            results[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
            results[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
            results[model_name]["R2"].append(r2_score(y_test, y_pred))

    final_results = {}
    for model_name, metrics in results.items():
        final_results[model_name] = {
            f"{metric}_mean": np.mean(values) for metric, values in metrics.items()
        }
        final_results[model_name].update(
            {f"{metric}_std": np.std(values) for metric, values in metrics.items()}
        )

    results_df = pd.DataFrame(final_results).T
    results_df = results_df.sort_values(by="R2_mean", ascending=False)

    return results_df


def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)
