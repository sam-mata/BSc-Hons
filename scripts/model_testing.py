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

def test_dependent_models(models, features, targets, split_year=2050, subset=False, subset_size=10000):
    """
    Tests target-dependent models on data split by year and returns the results.

    Args:
        models (list): List of three models for ice_thickness, ice_mask, and ice_velocity
        features (pd.DataFrame): Input features
        targets (pd.DataFrame): Target variables
        split_year (int): Year to split the data on
        subset (bool): Whether to use a subset of the training data
        subset_size (int): Size of the subset if used

    Returns:
        pd.DataFrame: DataFrame with model performance results
    """
    X_train, X_test, y_train, y_test = split_data_by_year(features, targets, split_year)

    if subset:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=subset_size, random_state=42)

    results = {}
    
    # Predict ice thickness
    thickness_model = models[0]
    y_thickness_pred_train = train_and_predict(thickness_model, X_train, y_train['ice_thickness'], X_train)
    y_thickness_pred_test = train_and_predict(thickness_model, X_train, y_train['ice_thickness'], X_test)

    # Add predicted ice thickness to features
    X_train_with_thickness = X_train.copy()
    X_test_with_thickness = X_test.copy()
    X_train_with_thickness['predicted_ice_thickness'] = y_thickness_pred_train
    X_test_with_thickness['predicted_ice_thickness'] = y_thickness_pred_test

    # Predict ice mask
    mask_model = models[1]
    y_mask_pred_train = train_and_predict(mask_model, X_train_with_thickness, y_train['ice_mask'], X_train_with_thickness)
    y_mask_pred_test = train_and_predict(mask_model, X_train_with_thickness, y_train['ice_mask'], X_test_with_thickness)

    # Add predicted ice mask to features
    X_train_with_thickness_mask = X_train_with_thickness.copy()
    X_test_with_thickness_mask = X_test_with_thickness.copy()
    X_train_with_thickness_mask['predicted_ice_mask'] = y_mask_pred_train
    X_test_with_thickness_mask['predicted_ice_mask'] = y_mask_pred_test

    # Predict ice velocity
    velocity_model = models[2]
    y_velocity_pred = train_and_predict(velocity_model, X_train_with_thickness_mask, y_train['ice_velocity'], X_test_with_thickness_mask)

    # Calculate metrics for each target
    for target in ['ice_thickness', 'ice_mask', 'ice_velocity']:
        if target == 'ice_thickness':
            y_pred = y_thickness_pred_test
        elif target == 'ice_mask':
            y_pred = y_mask_pred_test
        else:  # ice_velocity
            y_pred = y_velocity_pred

        mse = mean_squared_error(y_test[target], y_pred)
        mae = mean_absolute_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)
        
        results[target] = {"MSE": mse, "MAE": mae, "R2": r2}

    return pd.DataFrame(results).T