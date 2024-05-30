import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler


"""Functions for preprocessing data.
"""


def remove_fillers(
    df: pd.DataFrame, fillers: list = [np.NaN, -4.865496]
) -> pd.DataFrame:
    """Removes filler values from a dataframe.

    Args:
        df (pd.Dataframe): Dataframe to be cleaned.
        fillers (list, optional): List of filler values to be removed. Defaults to [np.NaN, 9.96920996839e36].

    Returns:
        pd.Dataframe: Cleared dataframe.
    """
    logging.info("\n🧹Removing fillers from dataframe:")
    for filler_value in fillers:
        logging.info(f"\tRemoving filler: {filler_value}")
        df = df.apply(lambda x: np.where(np.isclose(x, filler_value), np.NaN, x))

    # Replace all values in ocean_temperature that are greater than 1000 with NaN
    df["ocean_temperature"] = df["ocean_temperature"].apply(
        lambda x: np.where(x > 1000, np.NaN, x)
    )

    # Replace all values in ice_velocity,, ice_thickness, and ice_mask that are less than or equal to 0 with NaN
    for column in ["ice_velocity", "ice_thickness", "ice_mask"]:
        df[column] = df[column].apply(lambda x: np.where(x <= 0, np.NaN, x))

    # Replace all negative precipitation values with NaN
    df["precipitation"] = df["precipitation"].apply(
        lambda x: np.where(x < 0, np.NaN, x)
    )

    logging.info(f"\t✅Fillers removed: {df.shape}")
    return df


def set_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """Sets the data types of columns in a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to be typed.
        type_map (dict): Dictionary of column names and types.

    Returns:
        pd.DataFrame: Typed dataframe.
    """
    logging.info("\n🔠Setting data types of dataframe:")
    for column, dtype in type_map.items():
        logging.info(f"\tSetting {column} to {dtype}")
        df[column] = df[column].astype(dtype)
    logging.info("\t✅Data types set")
    return df


def fill_missing(df: pd.DataFrame, filltype: str = "mean") -> pd.DataFrame:
    """Fills missing values in a dataframe with the mean of the column.

    Args:
        df (pd.DataFrame): Dataframe to be cleaned.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    logging.info(f"\n🔠Filling missing values in dataframe with {filltype}:")
    # Drop all columns where ocean_temperature is NaN
    df = df.dropna(subset=["ocean_temperature"])

    # Fill missing values in ice_velocity and ice_thickness with 0
    for column in ["ice_velocity", "ice_thickness"]:
        df[column] = df[column].fillna(-1)

    # Fill missing values in ice_mask with 4
    df["ice_mask"] = df["ice_mask"].fillna(4)

    # Fill missing values in precipitation with the mean of the column
    if filltype == "mean":
        df["precipitation"] = df["precipitation"].fillna(df["precipitation"].mean())
    elif filltype == "zero":
        df["precipitation"] = df["precipitation"].fillna(0)

    logging.info("\t✅Missing values filled")
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'x' and 'y' coordinates into integer indexes with the center at (0, 0).
    """
    cell_size = 121600
    min_x = -3040000
    min_y = -3040000

    # Calculate the x and y indexes based on the coordinates
    df["x"] = (((df["x"] - min_x) / cell_size) - 25).astype(int)
    df["y"] = (((df["y"] - min_y) / cell_size) - 25).astype(int)

    return df
