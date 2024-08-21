import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def linear_model_interpretation(model, X, y):
    return model.coef_

def tree_based_interpretation(model, X, y):
    return model.feature_importances_

def knn_interpretation(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    return result.importances_mean

def svm_interpretation(model, X, y):
    if hasattr(model, 'coef_'):
        return np.abs(model.coef_[0])
    else:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        return result.importances_mean

def ensemble_interpretation(model, X, y):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    else:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        return result.importances_mean

def neural_network_interpretation(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    return result.importances_mean

def gaussian_process_interpretation(model, X, y):
    return model.kernel_.get_params()

model_interpretation_dict = {
    'LinearRegression': linear_model_interpretation,
    'Ridge': linear_model_interpretation,
    'Lasso': linear_model_interpretation,
    'ElasticNet': linear_model_interpretation,
    'TheilSenRegressor': linear_model_interpretation,
    'HuberRegressor': linear_model_interpretation,
    'DecisionTreeRegressor': tree_based_interpretation,
    'RandomForestRegressor': ensemble_interpretation,
    'GradientBoostingRegressor': ensemble_interpretation,
    'AdaBoostRegressor': ensemble_interpretation,
    'ExtraTreesRegressor': ensemble_interpretation,
    'KNeighborsRegressor': knn_interpretation,
    'SVR': svm_interpretation,
    'LinearSVR': svm_interpretation,
    'NuSVR': svm_interpretation,
    'MLPRegressor': neural_network_interpretation,
    'GaussianProcessRegressor': gaussian_process_interpretation,
    'XGBRegressor': ensemble_interpretation,
    'LGBMRegressor': ensemble_interpretation
}

def interpret_model(model, X, y, feature_names=None, plot=True):
    model_name = type(model).__name__
    if model_name in model_interpretation_dict:
        interpretation_func = model_interpretation_dict[model_name]
        result = interpretation_func(model, X, y)
    else:
        print(f"No specific interpretation method for {model_name}. Using permutation importance.")
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        result = perm_importance.importances_mean
    
    if plot:
        plot_interpretation_results(result, feature_names, model_name)
        plot_regression_diagnostics(model, X, y)
    
    return result

def plot_interpretation_results(interpretation_result, feature_names=None, model_name="Model"):
    plt.figure(figsize=(12, 6))
    
    if isinstance(interpretation_result, np.ndarray):
        # For feature importances or coefficients
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(interpretation_result))]
        
        # Sort features by importance
        sorted_idx = np.argsort(interpretation_result)
        pos = np.arange(sorted_idx.shape[0]) + .5

        plt.barh(pos, interpretation_result[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title(f"{model_name} - Feature Importance/Coefficients")
        plt.xlabel('Importance/Coefficient Value')
    
    elif isinstance(interpretation_result, dict):
        # For kernel parameters (e.g., Gaussian Process)
        plt.bar(range(len(interpretation_result)), list(interpretation_result.values()), align='center')
        plt.xticks(range(len(interpretation_result)), list(interpretation_result.keys()), rotation=45, ha='right')
        plt.title(f"{model_name} - Kernel Parameters")
        plt.xlabel('Parameter')
        plt.ylabel('Value')
    
    else:
        raise ValueError("Unsupported interpretation result type")

    plt.tight_layout()
    plt.show()

def plot_regression_diagnostics(model, X, y):
    # Make predictions
    y_pred = model.predict(X)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({'Actual': y, 'Predicted': y_pred, 'Residuals': y - y_pred})
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f"Regression Diagnostics for {type(model).__name__}", fontsize=16)
    
    # Actual vs Predicted plot
    sns.scatterplot(x='Actual', y='Predicted', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    
    # Residuals vs Predicted plot
    sns.scatterplot(x='Predicted', y='Residuals', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    
    # Histogram of residuals
    sns.histplot(df['Residuals'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Residuals')
    
    # Q-Q plot
    from scipy import stats
    qq = stats.probplot(df['Residuals'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()


