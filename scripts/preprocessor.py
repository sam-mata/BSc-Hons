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
        df = df.replace(filler_value, np.NaN)
    logging.info(f"\t✅Fillers removed: {df.shape}")
    return df
