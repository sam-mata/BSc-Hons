import matplotlib.pyplot as plt
import numpy as np

def plot_average_learning_curves(learning_curves, title):
    """
    Plots the average learning curves for all models.

    Args:
        learning_curves (dict): Dictionary containing learning curve data for each model.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, data in learning_curves.items():
        train_sizes = np.mean(data['train_sizes'], axis=0)
        train_scores_mean = np.mean(data['train_scores'], axis=0)
        train_scores_std = np.std(data['train_scores'], axis=0)
        test_scores_mean = np.mean(data['test_scores'], axis=0)
        test_scores_std = np.std(data['test_scores'], axis=0)

        plt.semilogx(train_sizes, train_scores_mean, 'o-', label=f"{model_name} (Train)")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1)
        plt.semilogx(train_sizes, test_scores_mean, 'o-', label=f"{model_name} (Test)")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)

    plt.xscale('log')
    plt.xlabel("Training Examples")
    plt.ylabel("R2 Score")
    plt.title(f"Average Learning Curves - {title}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def plot_metric_comparison(results_df, metric='R2'):
    """
    Plots a comparison of a specific metric across all models.

    Args:
        results_df (pd.DataFrame): DataFrame containing model results.
        metric (str): Metric to plot (e.g., 'R2', 'MSE', 'MAE'). Defaults to 'R2'.
    """
    plt.figure(figsize=(10, 6))
    
    models = results_df.index
    mean_values = results_df[f'{metric}_mean']
    std_values = results_df[f'{metric}_std']

    plt.bar(models, mean_values, yerr=std_values, capsize=5)
    plt.title(f"{metric} Comparison Across Models")
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()