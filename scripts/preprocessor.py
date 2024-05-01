import pandas as pd
import numpy as np
import logging

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

    # Replace all
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
    df["ocean_temperature"] = df["ocean_temperature"].fillna(
        275.5
    )  # Special exception for ocean_temperature
    if filltype == "mean":
        df = df.fillna(df.mean())
    elif filltype == "median":
        df = df.fillna(df.median())
    logging.info("\t✅Missing values filled")
    return df
