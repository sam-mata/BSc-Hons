import pandas as pd
import os
import logging

"""Functions for loading data from the data directory and splitting it into features and targets.
"""


def load_data(data_dir: str):
    """Loads all matching files in given directory into a single dataframe.

    Args:
        data_dir (str): Path to directory containing data files.

    Returns:
        pd.DataFrame: DataFrame of data.
    """

    logging.info(f"\n📦Loading data from {data_dir}")
    columns = [
        "x",
        "y",
        "bedrock_elevation",
        "ice_thickness",
        "ice_velocity",
        "ice_mask",
        "precipitation",
        "air_temperature",
        "ocean_temperature",
    ]

    dfs = []
    for year in range(2015, 2101):
        file_path = f"{data_dir}/vars-{year}-lowRes.txt"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="\t", header=None, names=columns)
            df["year"] = year
            dfs.append(df)
        else:
            logging.warning(f"\t⚠️File not found: {file_path}")

    dataframe: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    logging.info(f"\t📊Data Loaded: {dataframe.shape}")
    return dataframe


def split_features_targets(
    dataframe: pd.DataFrame,
    target_names: list = ["ice_thickness", "ice_velocity", "ice_mask"],
):
    """Split a dataframe into features and targets.

    Args:
        dataframe (pd.DataFrame): DataFrame to be split.
        target_names (list): List of target column names to split by.

    Returns:
        [pd.DataFrame, pd.DataFrame]: Tuple of (features, targets) dataframes.
    """
    targets: pd.DataFrame = dataframe[target_names]
    features: pd.DataFrame = dataframe.drop(columns=target_names)
    return features, targets
