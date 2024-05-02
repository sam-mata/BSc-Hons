import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_histogram(df: pd.DataFrame, feature: str):
    """Analyses a feature's distribution and plots a histogram.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
    """
    print(f"📊\n{df[feature].value_counts()}\n")
    sns.histplot(df[feature], bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


def plot_line(df: pd.DataFrame, feature: str):
    """Line chart of a feature's average (mean) per year.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
    """

    df.groupby("year")[feature].mean().plot(
        kind="line",
        title=f"Average {feature} per year",
        xlabel="Year",
        ylabel=feature,
    )
    plt.show()


def plot_boxplot(df: pd.DataFrame, feature: str):
    """Boxplot of a feature.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
    """
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.show()


def plot_spatial_heatmap(df: pd.DataFrame, feature: str):
    """Plots a heatmap of a feature's spatial distribution.
    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
    """
    pivot_data = df.pivot_table(index="y", columns="x", values=feature, aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_data,
        cmap="viridis",
        square=True,
        cbar_kws={"label": f"Average {feature}"},
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Spatial Distribution of Average {feature}")
    plt.show()


def create_heatmap_gif(df: pd.DataFrame, feature: str):
    """Creates an animated gif of a feature's spatial distribution over time.

    Args:
        df (pd.DataFrame): Dataframe containing feature.
        feature (str): Feature to be analysed.
    """
    years = df["year"].unique()
    fig, ax = plt.subplots(figsize=(8, 6))

    initial_year = years[0]
    initial_data = df[df["year"] == initial_year].pivot_table(
        index="y", columns="x", values=feature, aggfunc="mean"
    )
    heatmap = sns.heatmap(
        initial_data,
        cmap="viridis",
        square=True,
        ax=ax,
        vmin=df[feature].min(),
        vmax=df[feature].max(),
        cbar=False,  # Disable the default colorbar
    )
    quad_mesh = heatmap.collections[0]

    # Create a custom colorbar
    cbar = fig.colorbar(quad_mesh, ax=ax)
    cbar.set_label(feature)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Spatial Distribution of {feature}")

    year_text = ax.text(0.5, 1.10, "", transform=ax.transAxes, ha="center", fontsize=12)

    def update(year):
        df_year = df[df["year"] == year]
        pivot_data = df_year.pivot_table(
            index="y", columns="x", values=feature, aggfunc="mean"
        )
        quad_mesh.set_array(pivot_data.values.ravel())
        year_text.set_text(f"Year: {year}")

    animation = FuncAnimation(fig, update, frames=years, interval=100)
    animation.save(f"out/gifs/{feature}_heatmap.gif", writer="pillow")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame, method: str = "all", size: tuple[int, int] = (7, 7)
):
    """Plots a heatmap of the correlation matrix of features.
    Args:
        df (pd.DataFrame): Dataframe containing features.
        method (str, optional): Correlation method/s to use. Defaults to "all".
        size (tuple[int, int], optional): Size of the heatmap. Defaults to (7, 7).
    """
    plt.clf()
    plt.figure(figsize=size)

    if method == "all":
        for method in ["pearson", "spearman", "kendall"]:
            plt.title(f"Correlation matrix using {method} method")
            sns.heatmap(df.corr(method=method), annot=True)
            plt.show()
    elif method in ["pearson", "spearman", "kendall"]:
        plt.title(f"Correlation matrix using {method} method")
        sns.heatmap(df.corr(method=method), annot=True)
        plt.show()
