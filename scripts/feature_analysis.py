import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

GROUP_FIG_SIZE = (15, 8)
TEXT_SIZE = 12


def plot_histogram(df: pd.DataFrame, feature: str, save: bool = False, path: str = ""):
    """Analyses a feature's distribution and plots a histogram.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    if not save:
        print(f"📊\n{feature}\n{df[feature].describe()}")
    sns.histplot(df[feature], bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_line(df: pd.DataFrame, feature: str, save: bool = False, path: str = ""):
    """Line chart of a feature's average (mean) per year.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    df.groupby("year")[feature].mean().plot(
        kind="line",
        title=f"Average {feature} per year",
        xlabel="Year",
        ylabel=feature,
    )
    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_boxplot(df: pd.DataFrame, feature: str, save: bool = False, path: str = ""):
    """Boxplot of a feature.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_spatial_heatmap(
    df: pd.DataFrame,
    feature: str,
    save: bool = False,
    path: str = None,
):
    """Plots a heatmap of a feature's spatial distribution.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    pivot_data = df.pivot_table(index="y", columns="x", values=feature, aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_data,
        cmap="viridis",
        square=True,
        cbar_kws={"label": f"{get_units(feature)}"},
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Spatial Distribution of Average {get_title(feature)}")
    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def get_units(feature: str, parenthesis: bool = False) -> str:
    """Returns the units of a feature.
    Args:
        feature (str): Feature name.
    Returns:
        str: Units of the feature.
    """
    units = {
        "ocean_temperature": "°K",
        "air_temperature": "°K",
        "ice_velocity": "m/year",
        "ice_mask": "2: Grounded Ice, 3: Floating Ice, 4: Open Ocean",
        "ice_thickness": "m",
        "precipitation": "mm/year",
        "dtp": "cells",
        "precipitation_rolling_std": "mm/year",
        "air_temperature_rolling_std": "°K",
        "temp_diff": "°K",
        "log_air_temperature": "°K",
    }
    return (
        f"({units.get(feature, '')})"
        if parenthesis
        else units.get(feature, "") if feature else ""
    )


def get_title(feature: str) -> str:
    """Converts a feature name to title case.
    Args:
        feature (str): Feature name.
    Returns:
        str: Feature name in title case.
    """
    return feature.replace("_", " ").title()


def save_plots(df: pd.DataFrame, base_path: str, feature: str):
    plot_boxplot(df, feature, save=True, path=f"{base_path}/{feature}_boxplot.png")
    plot_histogram(df, feature, save=True, path=f"{base_path}/{feature}_histogram.png")
    plot_line(df, feature, save=True, path=f"{base_path}/{feature}_line.png")
    plot_spatial_heatmap(
        df, feature, save=True, path=f"{base_path}/{feature}_spatial_heatmap.png"
    )


def plot_group_histogram(
    df: pd.DataFrame,
    features: list,
    rows: int,
    cols: int,
    save: bool = False,
    path: str = "",
):
    """Plots histograms of multiple features in a single figure.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        features (list): List of features to be plotted.
        rows (int): Number of rows in the figure grid.
        cols (int): Number of columns in the figure grid.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    fig, axes = plt.subplots(rows, cols, figsize=GROUP_FIG_SIZE)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(df[feature], bins=30, ax=axes[i])
        axes[i].set_title(f"Distribution of {get_title(feature)}")
        axes[i].set_xlabel(get_title(feature) + " " + get_units(feature, True))
        axes[i].set_ylabel("Frequency")

    # Remove empty subplots if there are more subplots than features
    for i in range(len(features), rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_group_line_chart(
    df: pd.DataFrame,
    features: list,
    rows: int,
    cols: int,
    save: bool = False,
    path: str = "",
):
    """Plots line charts of multiple features in a single figure.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        features (list): List of features to be plotted.
        rows (int): Number of rows in the figure grid.
        cols (int): Number of columns in the figure grid.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        df.groupby("year")[feature].mean().plot(
            kind="line",
            title=f"Average {get_title(feature)} per year",
            xlabel="Year",
            ylabel=(
                get_title(feature)
                if feature == "ice_mask"
                else get_title(feature) + " " + get_units(feature, True)
            ),
            ax=axes[i],
        )

    # Remove empty subplots if there are more subplots than features
    for i in range(len(features), rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_group_spatial_heatmap(
    df: pd.DataFrame,
    features: list,
    rows: int,
    cols: int,
    save: bool = False,
    path: str = "",
):
    """Plots spatial heatmaps of multiple features in a single figure.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        features (list): List of features to be plotted.
        rows (int): Number of rows in the figure grid.
        cols (int): Number of columns in the figure grid.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        pivot_data = df.pivot_table(
            index="y", columns="x", values=feature, aggfunc="mean"
        )
        sns.heatmap(
            pivot_data,
            cmap="viridis",
            square=True,
            cbar_kws={
                "label": (
                    f"Average {get_title(feature)}"
                    if feature == "ice_mask"
                    else f"Average {get_title(feature)} {get_units(feature, True)}"
                )
            },
            ax=axes[i],
        )
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].set_title(f"Spatial Distribution of Average {get_title(feature)}")

    # Remove empty subplots if there are more subplots than features
    for i in range(len(features), rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_group_boxplot(
    df: pd.DataFrame,
    features: list,
    rows: int,
    cols: int,
    save: bool = False,
    path: str = "",
):
    """Plots boxplots of multiple features in a single figure.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        features (list): List of features to be plotted.
        rows (int): Number of rows in the figure grid.
        cols (int): Number of columns in the figure grid.
        save (bool, optional): Whether to save the plot. Defaults to False.
        path (str, optional): Path to save the plot. Defaults to an empty string.
    """
    fig, axes = plt.subplots(rows, cols, figsize=GROUP_FIG_SIZE)
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.boxplot(x=df[feature], ax=axes[i])
        axes[i].set_title(f"Boxplot of {get_title(feature)}")
        axes[i].set_xlabel(get_title(feature) + " " + get_units(feature, True))

    # Remove empty subplots if there are more subplots than features
    for i in range(len(features), rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
