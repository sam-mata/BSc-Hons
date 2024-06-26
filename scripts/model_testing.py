import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def test_independant_models(
    models: list,
    features: pd.DataFrame,
    target: pd.DataFrame,
    split_year: int = 2050,
    subset: bool = False,
    subset_size: int = 10000
) -> pd.DataFrame:
    """Tests target-independant models on data split by year and returns the results.

    Args:
        models (list): Models to be tested
        features (pd.DataFrame): Input features for training
        target (pd.DataFrame): Target to be predicted
        split_year (int, optional): Year to split the data on. Defaults to 2050.
        subset (bool, optional): Whether to use a subset of the training data. Defaults to False.
        subset_size (int, optional): Size of the subset if used. Defaults to 10000.

    Returns:
        pd.DataFrame: DataFrame with model performance results
    """
    X_train, X_test, y_train, y_test = split_data_by_year(features, target, split_year)

    if subset:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=subset_size, random_state=42)

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
    results: pd.DataFrame, metric: str = "R2", n: int = 6
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

import time 
def test_dependent_models(models, features, targets, target_order, split_year=2050):
    print(f"Starting test_dependent_models with split_year: {split_year}")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    X_train, X_test, y_train, y_test = split_data_by_year(features, targets, split_year)
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    results = {}
    
    X_train_augmented = X_train.copy()
    X_test_augmented = X_test.copy()

    for target, model in zip(target_order, models):
        print(f"\nTraining model for {target}")
        print(f"Features used: {', '.join(X_train_augmented.columns)}")
        start_time = time.time()
        
        # Train and predict
        y_pred_train = train_and_predict(model, X_train_augmented, y_train[target], X_train_augmented)
        y_pred_test = train_and_predict(model, X_train_augmented, y_train[target], X_test_augmented)

        # Add predictions to features for next model
        X_train_augmented[f'predicted_{target}'] = y_pred_train
        X_test_augmented[f'predicted_{target}'] = y_pred_test

        # Calculate metrics
        mse = mean_squared_error(y_test[target], y_pred_test)
        mae = mean_absolute_error(y_test[target], y_pred_test)
        r2 = r2_score(y_test[target], y_pred_test)
        
        results[target] = {"MSE": mse, "MAE": mae, "R2": r2}
        
        end_time = time.time()
        print(f"Finished training {target} in {end_time - start_time:.2f} seconds")

    return pd.DataFrame(results).T