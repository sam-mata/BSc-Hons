from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def preprocess_data(df):
    # Convert 'NaN' strings to np.nan
    df = df.replace("NaN", np.nan)

    # 1. Convert every feature to either float or integer
    float_columns = [
        "bedrock_elevation",
        "ice_thickness",
        "ice_velocity",
        "precipitation",
        "air_temperature",
        "ocean_temperature",
    ]
    int_columns = ["x", "y", "ice_mask", "year"]

    for col in float_columns:
        df[col] = df[col].astype(float)
    for col in int_columns:
        df[col] = df[col].astype(int)

    # 2. Convert x and y to cell coordinates
    df["x"] = (df["x"] - df["x"].min()) // (df["x"].max() - df["x"].min()) * 50
    df["y"] = (df["y"] - df["y"].min()) // (df["y"].max() - df["y"].min()) * 50

    # 3. Replace any ice_mask values that aren't 2, 3, or 4, with 2
    df.loc[~df["ice_mask"].isin([2, 3, 4]), "ice_mask"] = 2

    # 4. Remove any cells with ocean_temperature values above 1000
    df = df[df["ocean_temperature"] <= 1000]

    # 5. Replace any "ice_velocity", "ice_thickness", and "ice_mask" values below 0
    df.loc[df["ice_velocity"] < 0, "ice_velocity"] = 0
    df.loc[df["ice_thickness"] < 0, "ice_thickness"] = 0
    df.loc[df["ice_mask"] < 0, "ice_mask"] = 4

    # 6. Replace any negative precipitation values with 0
    df.loc[df["precipitation"] < 0, "precipitation"] = 0

    # 7. Replace any missing NaN values in "ice_velocity"
    df.loc[df["ice_velocity"].isna(), "ice_velocity"] = 0

    return df


def apply_minmax_scaling(X, y):
    """
    Apply min-max scaling to X and y without leaking information between sets.

    Parameters:
    X (pd.DataFrame): Feature set
    y (pd.DataFrame): Target set

    Returns:
    X_scaled (pd.DataFrame): Scaled feature set
    y_scaled (pd.DataFrame): Scaled target set
    X_scaler (MinMaxScaler): Scaler for X
    y_scaler (MinMaxScaler): Scaler for y
    """
    # Create copies to avoid modifying the original data
    X_copy = X.copy()
    y_copy = y.copy()

    # Initialize scalers
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Scale X
    columns_to_scale = [col for col in X_copy.columns]
    X_copy[columns_to_scale] = X_scaler.fit_transform(X_copy[columns_to_scale])

    # Scale y
    y_copy[:] = y_scaler.fit_transform(y_copy)

    return X_copy, y_copy, X_scaler, y_scaler


def inverse_minmax_scaling(
    X_scaled, y_scaled, X_scaler, y_scaler, exclude_columns=["year"]
):
    """
    Inverse the min-max scaling applied to X and y.

    Parameters:
    X_scaled (pd.DataFrame): Scaled feature set
    y_scaled (pd.DataFrame): Scaled target set
    X_scaler (MinMaxScaler): Scaler used for X
    y_scaler (MinMaxScaler): Scaler used for y
    exclude_columns (list): Columns that were excluded from scaling (e.g., 'year')

    Returns:
    X (pd.DataFrame): Original feature set
    y (pd.DataFrame): Original target set
    """
    # Create copies to avoid modifying the input data
    X = X_scaled.copy()
    y = y_scaled.copy()

    # Inverse transform X
    columns_to_scale = [col for col in X.columns if col not in exclude_columns]
    X[columns_to_scale] = X_scaler.inverse_transform(X[columns_to_scale])

    # Inverse transform y
    y[:] = y_scaler.inverse_transform(y)

    return X, y


def derive_features(X, y):
    """
    Derive additional features from the given dataset, including both X and y.

    Parameters:
    X (pd.DataFrame): Feature set
    y (pd.DataFrame): Target set

    Returns:
    X_derived (pd.DataFrame): Feature set with additional derived features
    y_derived (pd.DataFrame): Target set with additional derived features
    """
    # Create copies to avoid modifying the original data
    X_derived = X.copy()
    y_derived = y.copy()

    combined = pd.concat([X_derived, y_derived], axis=1)

    center_x, center_y = 25, 25
    combined["distance_to_pole"] = np.sqrt(
        (combined["x"] - center_x) ** 2 + (combined["y"] - center_y) ** 2
    )

    combined["bedrock_below_sea_level"] = (combined["bedrock_elevation"] < 0).astype(
        int
    )

    # Separate the combined dataframe back into X and y
    X_columns = X.columns.tolist() + [
        "distance_to_pole",
        "bedrock_below_sea_level",
    ]
    y_columns = y.columns.tolist()

    X_derived = combined[X_columns]
    y_derived = combined[y_columns]

    return X_derived, y_derived
