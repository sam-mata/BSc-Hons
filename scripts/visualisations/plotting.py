import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.visualisations.helpers import average_years, rescale_features

def plot_spatial_heatmap(df, feature, ax=None, title=None, cmap='viridis', figsize=(12, 10), units=None):
    """
    Create a spatial heatmap of a selected feature using x and y coordinates.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    feature (str): The name of the feature to be plotted.
    ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure is created.
    title (str, optional): The title of the plot. If None, uses the feature name.
    cmap (str, optional): The colormap to use for the heatmap. Default is 'viridis'.
    figsize (tuple, optional): The size of the figure. Default is (12, 10).
    units (str, optional): The units of the feature. If provided, it will be appended to the colorbar label.
    
    Returns:
    matplotlib.figure.Figure: The created figure object.
    """    
    # Create a pivot table for the heatmap
    pivot_df = df.pivot(index='y', columns='x', values=feature)
    
    # Create the plot if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot the heatmap
    cbar_label = f"{feature} ({units})" if units else feature
    sns.heatmap(pivot_df, cmap=cmap, ax=ax, cbar_kws={'label': cbar_label})
    
    # Set the title
    if title is None:
        title = f"Spatial Distribution of {feature}"
        if units:
            title += f" ({units})"
    ax.set_title(title)
    
    # Adjust labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xticklabels([f'{float(t.get_text()):.0f}' for t in ax.get_xticklabels()])
    ax.set_yticklabels([f'{float(t.get_text()):.0f}' for t in ax.get_yticklabels()])
    
    # Invert y-axis to match geographical orientation
    ax.invert_yaxis()
    
    return fig

def plot_averaged_heatmap(df, feature, title=None, cmap='viridis', figsize=(10, 8), save_path=None, units=None):
    """
    Create an averaged spatial heatmap of a selected feature.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    feature (str): The name of the feature to be plotted.
    title (str, optional): The title of the plot. If None, uses the feature name.
    cmap (str, optional): The colormap to use for the heatmap. Default is 'viridis'.
    figsize (tuple, optional): The size of the figure. Default is (10, 8).
    save_path (str, optional): The file path to save the figure. If None, the figure is not saved.
    units (str, optional): The units of the feature. If provided, it will be appended to the colorbar label.
    
    Returns:
    matplotlib.figure.Figure: The created figure object.
    """
    averaged_df = average_years(df)
    fig = plot_spatial_heatmap(averaged_df, feature, title=title, cmap=cmap, figsize=figsize, units=units)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_group_averaged_spatial_heatmap(df, features, titles=None, units=None, cmap='viridis', figsize=(20, 15), save_path=None):
    """
    Create a figure with multiple averaged spatial heatmaps as subplots.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    features (list): List of feature names to be plotted.
    titles (list, optional): List of titles for each subplot. If None, uses feature names.
    units (list, optional): List of units for each feature. If provided, it will be appended to the colorbar labels.
    cmap (str, optional): The colormap to use for the heatmaps. Default is 'viridis'.
    figsize (tuple, optional): The size of the figure. Default is (20, 15).
    save_path (str, optional): The file path to save the figure. If None, the figure is not saved.
    
    Returns:
    matplotlib.figure.Figure: The created figure object.
    """
    # Calculate the number of rows and columns for subplots
    n = len(features)
    nrows = int(np.ceil(n / 2))
    ncols = 2 if n > 1 else 1
    
    # Create the figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Average the data across years
    averaged_df = average_years(df)
    
    # Create a heatmap for each feature
    for i, feature in enumerate(features):
        title = titles[i] if titles and i < len(titles) else None
        unit = units[i] if units and i < len(units) else None
        plot_spatial_heatmap(averaged_df, feature, ax=axes[i], title=title, cmap=cmap, units=unit)
    
    # Remove any unused subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust the layout and add a main title
    plt.tight_layout()
    fig.suptitle("Averaged Spatial Heatmaps", fontsize=16, y=1.02)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig