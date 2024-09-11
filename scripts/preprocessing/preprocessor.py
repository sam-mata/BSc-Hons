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
        
        
    # 2. Convert "x" and "y" to integer divisions of 121600
    df["x"] = (df["x"] / 121600).astype(int)
    df["y"] = (df["y"] / 121600).astype(int)

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
    feature_ranges (dict): Dictionary containing min and max values for each feature
    """
    # Create copies to avoid modifying the original data
    X_copy = X.copy()
    y_copy = y.copy()

    # Initialize scalers
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Scale X
    columns_to_scale = [col for col in X_copy.columns ]
    X_copy[columns_to_scale] = X_scaler.fit_transform(X_copy[columns_to_scale])

    # Scale y
    y_copy[:] = y_scaler.fit_transform(y_copy)

    # Create dictionary with feature ranges
    feature_ranges = {}
    for col in columns_to_scale:
        feature_ranges[col] = (X[col].min(), X[col].max())
    
    # Add y column(s) to feature_ranges
    for col in y.columns:
        feature_ranges[col] = (y[col].min(), y[col].max())

    return X_copy, y_copy, feature_ranges


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

    # 1. Distance to pole
    center_x = (combined["x"].min() + combined["x"].max()) / 2
    center_y = (combined["y"].min() + combined["y"].max()) / 2
    combined["distance_to_pole"] = np.sqrt(
        (combined["x"] - center_x) ** 2 + (combined["y"] - center_y) ** 2
    )

    # 2. Bedrock below sea level
    combined["bedrock_below_sea_level"] = (combined["bedrock_elevation"] < 0).astype(
        int
    )
    
    # 3. Temperature difference
    combined["temperature_difference"] = combined["ocean_temperature"] - combined["air_temperature"]
    
    # 4. Log of Air Temperature
    combined["log_air_temperature"] = np.log(combined["air_temperature"])
    
    # 5. Rolling std of precipitation
    combined["rolling_std_precipitation"] = combined["precipitation"].rolling(window=3).std()
    combined["rolling_std_precipitation"] = combined["rolling_std_precipitation"].fillna(0)

    
    # 6. Rolling std of air temperature
    combined["rolling_std_air_temperature"] = combined["air_temperature"].rolling(window=3).std()
    combined["rolling_std_air_temperature"] = combined["rolling_std_air_temperature"].fillna(0)
    
    # 7. One-hot encode any air temperature values in the lower 45%
    combined["air_temperature_low_45"] = (combined["air_temperature"] < np.percentile(combined["air_temperature"], 45)).astype(int)
    
    # Separate the combined dataframe back into X and y
    X_columns = X.columns.tolist() + [
        "distance_to_pole",
        "bedrock_below_sea_level",
        "temperature_difference",
        "log_air_temperature",
        "rolling_std_precipitation",
        "rolling_std_air_temperature",
        "air_temperature_low_45",
    ]
    y_columns = y.columns.tolist()

    X_derived = combined[X_columns]
    y_derived = combined[y_columns]

    return X_derived, y_derived
