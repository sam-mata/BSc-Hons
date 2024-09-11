import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.visualisations.helpers import average_years, rescale_features

def plot_spatial_heatmap(df, feature, title=None, cmap='viridis', figsize=(12, 10), save_path=None):
    """
    Create a spatial heatmap of a selected feature using x and y coordinates.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    feature (str): The name of the feature to be plotted.
    title (str, optional): The title of the plot. If None, uses the feature name.
    cmap (str, optional): The colormap to use for the heatmap. Default is 'viridis'.
    figsize (tuple, optional): The size of the figure. Default is (12, 10).
    save_path (str, optional): The file path to save the figure. If None, the figure is not saved.
    
    Returns:
    matplotlib.figure.Figure: The created figure object.
    """    
    # Create a pivot table for the heatmap
    pivot_df = df.pivot(index='y', columns='x', values=feature)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap
    sns.heatmap(pivot_df, cmap=cmap, ax=ax, cbar_kws={'label': feature})
    
    # Set the title
    if title is None:
        title = f"Spatial Distribution of {feature}"
    ax.set_title(title)
    
    # Adjust labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xticklabels([f'{float(t.get_text()):.0f}' for t in ax.get_xticklabels()])
    ax.set_yticklabels([f'{float(t.get_text()):.0f}' for t in ax.get_yticklabels()])
    
    # Invert y-axis to match geographical orientation
    ax.invert_yaxis()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_averaged_heatmap(df, feature, title=None, cmap='viridis', figsize=(10, 8), save_path=None):
    averaged_df = average_years(df)
    fig = plot_spatial_heatmap(averaged_df, feature, title, cmap, figsize, save_path)
    return fig