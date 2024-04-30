import pandas as pd
import numpy as np
import logging

"""Functions for preprocessing data.
"""


def remove_fillers(
    df: pd.DataFrame, fillers: list = [np.NaN, 9.96920996839e36]
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
