import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from IPython.display import display


def fit_and_evaluate_model(
    model, X_train, y_train, X_test, y_test, multi=False, cv=None
):
    """
    Fits a model on the training data and evaluates it on the test data.

    Parameters:
    model : sklearn-compatible model
        The model to fit and evaluate
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.DataFrame
        Training and test target(s)
    multi : bool
        If True, treat as multi-target regression. If False, evaluate each target separately.
    cv : int, optional
        Number of cross-validation splits. If provided, use cross-validation.
        If None, don't use cross-validation.

    Returns:
    pd.DataFrame : DataFrame containing MSE, MAE, and R2 for each target variable
    """

    def evaluate(y_true, y_pred):
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    if multi:
        if cv is not None:
            y_pred = cross_val_predict(model, X_train, y_train, cv=cv)
            results = pd.DataFrame(
                {
                    "Target": y_train.columns,
                    **{
                        metric: [
                            evaluate(y_train[col], y_pred[:, i])[metric]
                            for i, col in enumerate(y_train.columns)
                        ]
                        for metric in ["MSE", "RMSE", "MAE", "R2"]
                    },
                }
            )
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results = pd.DataFrame(
                {
                    "Target": y_test.columns,
                    **{
                        metric: [
                            evaluate(y_test[col], y_pred[:, i])[metric]
                            for i, col in enumerate(y_test.columns)
                        ]
                        for metric in ["MSE", "RMSE", "MAE", "R2"]
                    },
                }
            )
    else:
        results = []
        for column in y_train.columns:
            model_copy = clone(model)
            if cv is not None:
                y_pred = cross_val_predict(model_copy, X_train, y_train[column], cv=cv)
                metrics = evaluate(y_train[column], y_pred)
            else:
                model_copy.fit(X_train, y_train[column])
                y_pred = model_copy.predict(X_test)
                metrics = evaluate(y_test[column], y_pred)
            results.append({"Target": column, **metrics})
        results = pd.DataFrame(results)

    return results

def collate_model_results(model_results):
    """
    Collates the results from multiple models into three DataFrames, one for each target.

    Parameters:
    model_results : list of tuples
        List of tuples, where each tuple contains (model_name, results_dataframe)
        results_dataframe is the output from fit_and_evaluate_model function

    Returns:
    dict : A dictionary containing three DataFrames, one for each target
        (ice_thickness, ice_velocity, ice_mask), sorted by MSE
    """
    targets = ['ice_thickness', 'ice_velocity', 'ice_mask']
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    
    # Initialize empty lists to store data for each target
    target_data = {target: [] for target in targets}
    
    # Process results for each model
    for model_name, results_df in model_results:
        for target in targets:
            target_results = results_df[results_df['Target'] == target].iloc[0]
            row_data = {'Model': model_name}
            row_data.update({metric: target_results[metric] for metric in metrics})
            target_data[target].append(row_data)
    
    # Create DataFrames for each target
    target_dfs = {target: pd.DataFrame(data) for target, data in target_data.items()}
    
    # Sort each DataFrame by MSE and set 'Model' as index
    for target in targets:
        target_dfs[target] = target_dfs[target].sort_values('MSE').set_index('Model')
    
    return target_dfs

def test_models(models, multi, cv, refined, X_train, y_train, X_test, y_test):
    """
    Test multiple models and display results in an organized manner.

    Parameters:
    models : list of tuples
        List of (name, model) tuples to be evaluated
    multi : bool
        If True, use multi-target regression. If False, use single-target regression.
    cv : int or None
        Number of cross-validation folds. If None, no cross-validation is used.
    refined : bool
        If True, indicates refined models are being used.
    X_train, y_train, X_test, y_test : pd.DataFrame
        Training and test data

    Returns:
    dict : A dictionary containing three DataFrames, one for each target
        (ice_thickness, ice_velocity, ice_mask), sorted by MSE
    """
    all_results = []
    for name, model in models:
        print(f"Evaluating {name} for {'refined' if refined else 'broad'} {'multi' if multi else 'single'}-target regression testing...")
        results = fit_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, multi=multi, cv=cv
        )
        all_results.append((name, results))

    collated_results = collate_model_results(all_results)

    for target, df in collated_results.items():
        print(f"\nCollated Results for {target}:")
        display(df)

    return collated_results