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
    subset: bool = False,
    subset_size: int = 10000
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

    if subset:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=subset_size, random_state=42)

    results = {}
    feature_importances = {}

    for model in models:
        model_name = type(model).__name__

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"MSE": mse, "MAE": mae, "R2": r2}

        try:
            feature_importances[model_name] = get_feature_importances(model, X_train.columns)
        except Exception as e:
            print(f"Could not compute feature importances for {model_name}: {str(e)}")

    results = pd.DataFrame(results).T
    results = results.sort_values(by="R2", ascending=False)

    return results, feature_importances

def display_importances(feature_importances: dict, top_n: int = 10, plot: bool = True):
    """
    Display and optionally plot feature importances for multiple models.

    Args:
        feature_importances (dict): Dictionary of feature importances for each model
        top_n (int, optional): Number of top features to display. Defaults to 10.
        plot (bool, optional): Whether to plot the importances. Defaults to True.
    """
    for model_name, importances in feature_importances.items():
        if importances is not None:
            print(f"\nFeature Importances for {model_name}:")
            
            # Determine the type of importance measure
            if hasattr(importances.index, 'name'):
                importance_type = importances.index.name
            else:
                importance_type = "Importance"
            
            # Print the importance type
            print(f"Importance Measure: {importance_type}")
            
            # Print the top N importances
            print(importances.head(top_n))

            if plot:
                plt.figure(figsize=(10, 6))
                importances.head(top_n).plot(kind='bar')
                plt.title(f'Top {top_n} Feature Importances - {model_name}')
                plt.xlabel('Features')
                plt.ylabel(importance_type)
                plt.tight_layout()
                plt.show()
        else:
            print(f"\nFeature Importances for {model_name} could not be computed.")

def find_top_models(
    results_tuple: tuple, metric: str = "R2", n: int = 6
) -> pd.DataFrame:
    """Finds the top n models by a given metric.

    Args:
        results_tuple (tuple): Tuple containing (results DataFrame, feature_importances dict)
        metric (str): Metric to sort by
        n (int, optional): Number of top models to return. Defaults to 6.

    Returns:
        pd.DataFrame: Top n models by the given metric
    """
    results, _ = results_tuple  # Unpack the tuple, ignore feature_importances
    return results.sort_values(by=metric, ascending=False).head(n)

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

    train_mask = features['year'] < split_year
    test_mask = features['year'] >= split_year

    X_train = features[train_mask]
    X_test = features[test_mask]
    y_train = targets[train_mask]
    y_test = targets[test_mask]

    return X_train, X_test, y_train, y_test

def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

def get_feature_importances(model, feature_names):
    """
    Extract feature importances from a fitted model.

    Args:
        model: Fitted model object
        feature_names (list): List of feature names

    Returns:
        pd.Series: Feature importances sorted in descending order
    """
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names, name="Feature Importance")
    elif hasattr(model, 'coef_'):
        importances = pd.Series(np.abs(model.coef_), index=feature_names, name="Absolute Coefficient")
    elif isinstance(model, BaggingRegressor):
        importances = pd.Series(
            np.mean([get_feature_importances(estimator, feature_names) for estimator in model.estimators_], axis=0),
            index=feature_names,
            name="Mean Feature Importance"
        )
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attributes")
    
    return importances.sort_values(ascending=False)