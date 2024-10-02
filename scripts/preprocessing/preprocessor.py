from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


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
    
    features = [col for col in X_derived.columns if col not in ["x", "y", "year"]]
    
    def safe_divide(a, b, fill_value=0):
        return np.divide(a, b, out=np.full_like(a, fill_value), where=b!=0)
    
    def safe_log(x, fill_value=0):
        return np.log(np.where(x > 0, x, np.exp(fill_value)))

    for feature_1 in features:
        # 3. Multi-feature interactions
        for feature_2 in features:
            if feature_1 != feature_2:
                # Difference
                combined[f"{feature_1}_{feature_2}_difference"] = combined[feature_1] - combined[feature_2]
                
                # Ratio
                combined[f"{feature_1}_{feature_2}_ratio"] = safe_divide(combined[feature_1], combined[feature_2])
                
                # Product
                combined[f"{feature_1}_{feature_2}_product"] = combined[feature_1] * combined[feature_2]
                
        # 4. Cubes and squares
        for power in range(1, 3):  
            if power == 2:
                combined[f"{feature_1}_squared"] = combined[feature_1] ** power
            elif power == 3:
                combined[f"{feature_1}_cubed"] = combined[feature_1] ** power
        
        # 5. Log Transforms
        if feature_1 != "bedrock_elevation":
            combined[f"log_{feature_1}"] = safe_log(combined[feature_1])
        
        # 6. Rolling Statistics
        combined[f"rolling_mean_{feature_1}"] = combined[feature_1].rolling(window=3).mean().fillna(0)
        combined[f"rolling_std_{feature_1}"] = combined[feature_1].rolling(window=3).std().fillna(0)
        combined[f"rolling_min_{feature_1}"] = combined[feature_1].rolling(window=3).min().fillna(0)
        combined[f"rolling_max_{feature_1}"] = combined[feature_1].rolling(window=3).max().fillna(0)
        
        # 7. Slopes
        combined[f"slope_{feature_1}_x"] = combined[feature_1].diff().fillna(0)
        combined[f"slope_{feature_1}_y"] = combined[feature_1].diff().fillna(0)
        combined[f"slope_{feature_1}_magnitude"] = np.sqrt(combined[f"slope_{feature_1}_x"]**2 + combined[f"slope_{feature_1}_y"]**2)
        
    
    # 8. Surface mass balances
    for melting_factor in np.linspace(0.01, 0.1, 10):
        combined[f'surface_mass_balance_{melting_factor}'] = combined['precipitation'] - combined['air_temperature'] * melting_factor # Melting factor of 1 mm / d-C
    
    # 9. Years since start
    combined['years_since_start'] = combined['year'] - combined['year'].min()

    # Separate the combined dataframe back into X and y
    y_columns = y.columns.tolist()
    X_columns = [col for col in combined.columns if col not in y_columns]

    X_derived = combined[X_columns]
    y_derived = combined[y_columns]

    return X_derived, y_derived


def select_features(X, y, n_features=20, method='combined'):
    """
    Select the most important features using various methods.

    Parameters:
    X (pd.DataFrame): Feature set
    y (pd.DataFrame): Target set
    n_features (int): Number of features to select
    method (str): Method to use for feature selection. 
                  Options: 'f_regression', 'mutual_info', 'random_forest', 'combined'

    Returns:
    pd.DataFrame: DataFrame with selected features
    """

    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=n_features)
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = X.columns[mask]

    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_regression, k=n_features)
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = X.columns[mask]

    elif method == 'random_forest':
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        selected_features = X.columns[indices[:n_features]]

    elif method == 'combined':
        # Combine results from all methods
        f_regression_selector = SelectKBest(f_regression, k='all')
        f_regression_selector.fit(X, y)
        f_scores = f_regression_selector.scores_

        mutual_info_selector = SelectKBest(mutual_info_regression, k='all')
        mutual_info_selector.fit(X, y)
        mi_scores = mutual_info_selector.scores_

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_

        # Normalize scores
        f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
        mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        rf_scores = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min())

        # Combine scores
        combined_scores = f_scores + mi_scores + rf_scores
        indices = np.argsort(combined_scores)[::-1]
        selected_features = X.columns[indices[:n_features]]
        
    return X[selected_features]
