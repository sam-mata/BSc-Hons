import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, RobustScaler


"""Functions for preprocessing data.
"""


def remove_fillers(
    df: pd.DataFrame, fillers: list = [np.nan, -4.865496]
) -> pd.DataFrame:
    """Removes filler values from a dataframe.

    Args:
        df (pd.Dataframe): Dataframe to be cleaned.
        fillers (list, optional): List of filler values to be removed. Defaults to [np.nan, 9.96920996839e36].

    Returns:
        pd.Dataframe: Cleared dataframe.
    """
    logging.info("\n🧹Removing fillers from dataframe:")
    for filler_value in fillers:
        logging.info(f"\tRemoving filler: {filler_value}")
        df = df.apply(lambda x: np.where(np.isclose(x, filler_value), np.nan, x))

    # Replace all values in ocean_temperature that are greater than 1000 with NaN
    df["ocean_temperature"] = df["ocean_temperature"].apply(
        lambda x: np.where(x > 1000, np.nan, x)
    )

    # Replace all values in ice_mask that are not 2, 3, or 4 with 2
    df["ice_mask"] = df["ice_mask"].apply(
        lambda x: np.where((x != 2) & (x != 3) & (x != 4), 2, x)
    )

    # Replace all values in ice_velocity,, ice_thickness, and ice_mask that are less than or equal to 0 with NaN
    for column in ["ice_velocity", "ice_thickness", "ice_mask"]:
        df[column] = df[column].apply(lambda x: np.where(x <= 0, np.nan, x))

    # Replace all negative precipitation values with NaN
    df["precipitation"] = df["precipitation"].apply(
        lambda x: np.where(x < 0, np.nan, x)
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
        df.loc[df[column].isna(), column] = -1

    # Fill missing values in ice_mask with 4
    df.loc[df["ice_mask"].isna(), "ice_mask"] = 4

    # Fill missing values in precipitation with the mean of the column
    if filltype == "mean":
        df.loc[df["precipitation"].isna(), "precipitation"] = df["precipitation"].mean()
    elif filltype == "zero":
        df.loc[df["precipitation"].isna(), "precipitation"] = 0

    logging.info("\t✅Missing values filled")
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:

    # Convert the 'x' and 'y' coordinates into integer indexes with the center at (0, 0)
    cell_size = 121600
    min_x = -3040000
    min_y = -3040000

    # Calculate the x and y indexes based on the coordinates
    df["x"] = (((df["x"] - min_x) / cell_size) - 25).astype(int)
    df["y"] = (((df["y"] - min_y) / cell_size) - 25).astype(int)

    # Apply a robust scaler to 'ocean_temperature', 'precipitation', and 'ice_velocity'
    scaler = RobustScaler()
    df[["ocean_temperature", "precipitation", "ice_velocity"]] = scaler.fit_transform(
        df[["ocean_temperature", "precipitation", "ice_velocity"]]
    )

    # Apply a min-max scaler to 'air_temperature' and 'ice_thickness'
    scaler = MinMaxScaler()
    df[["air_temperature", "ice_thickness"]] = scaler.fit_transform(
        df[["air_temperature", "ice_thickness"]]
    )

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing columns in a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to create features from.

    Returns:
        pd.DataFrame: Dataframe with new features included.
    """
    # Distance to Pole
    df["distance_to_pole"] = np.sqrt((df["x"] - 4) ** 2 + df["y"] ** 2)  # offset x by 4

    # Rolling Standard Deviation
    for feature in ["precipitation", "air_temperature"]:
        df[f"{feature}_rolling_std"] = df[feature].rolling(window=3).std()

    # Log Transformation of air_temperature
    df["log_air_temperature"] = np.log(df["air_temperature"] + 1)

    # Coastline Encoding
    df["coastline"] = 0
    for index, row in df.iterrows():
        if row["ice_mask"] == 2:
            x = row["x"]
            y = row["y"]
            if (
                ((df["x"] == x) & (df["y"] == y + 1) & (df["ice_mask"] == 4)).any()
                or ((df["x"] == x) & (df["y"] == y - 1) & (df["ice_mask"] == 4)).any()
                or ((df["x"] == x + 1) & (df["y"] == y) & (df["ice_mask"] == 4)).any()
                or ((df["x"] == x - 1) & (df["y"] == y) & (df["ice_mask"] == 4)).any()
            ):
                df.at[index, "coastline"] = 1
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to be preprocessed.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    logging.info("\n🛠 Preprocessing data:")
    df = remove_fillers(df)
    df = fill_missing(df, "mean")
    df = transform_data(df)
    df = set_types(
        df,
        {
            "x": int,
            "y": int,
            "year": int,
            "ice_mask": int,
        },
    )
    logging.info("\t✅ Data preprocessed")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Distance to Pole
    df["dtp"] = np.sqrt((df["x"] - 4) ** 2 + df["y"] ** 2)

    # Rolling Standard Deviation
    for feature in ["precipitation", "air_temperature"]:
        df[f"{feature}_rolling_std"] = df[feature].rolling(window=3).std()

    # Log Transformation of air_temperature
    df["log_air_temperature"] = np.log(df["air_temperature"] + 1)

    # Coastline Encoding
    df["coastline"] = 0
    for index, row in df.iterrows():
        if row["ice_mask"] == 2:
            x = row["x"]
            y = row["y"]
            if (
                ((df["x"] == x) & (df["y"] == y + 1) & (df["ice_mask"] == 4)).any()
                or ((df["x"] == x) & (df["y"] == y - 1) & (df["ice_mask"] == 4)).any()
                or ((df["x"] == x + 1) & (df["y"] == y) & (df["ice_mask"] == 4)).any()
                or ((df["x"] == x - 1) & (df["y"] == y) & (df["ice_mask"] == 4)).any()
            ):
                df.at[index, "coastline"] = 1
