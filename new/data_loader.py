import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from preprocessor import preprocess_data, derive_features


def load_data(data_folder, file_pattern="vars-*-lowRes.txt"):
    all_files = glob.glob(os.path.join(data_folder, file_pattern))
    data_list = []
    years = []

    column_names = [
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

    for filename in all_files:
        df = pd.read_csv(filename, sep="\t", header=None, names=column_names)
        year = int(os.path.basename(filename).split("-")[1])
        years.append(year)
        df["year"] = year
        data_list.append(df)
    return pd.concat(data_list, ignore_index=True)


def convert_and_split(df):
    # Convert 'NaN' strings to np.nan
    df = df.replace("NaN", np.nan)

    # Define input features and target variables
    input_features = [
        "x",
        "y",
        "bedrock_elevation",
        "precipitation",
        "air_temperature",
        "ocean_temperature",
        "year",
    ]
    target_variables = ["ice_thickness", "ice_velocity", "ice_mask"]

    # Convert columns to appropriate types
    float_columns = [
        "bedrock_elevation",
        "precipitation",
        "air_temperature",
        "ocean_temperature",
    ] + target_variables
    for col in float_columns:
        df[col] = df[col].astype(float)

    # Separate features and targets
    X = df[input_features]
    y = df[target_variables]

    return X, y


def split_data_by_year(X, y, test_size=0.2, random_state=42):
    unique_years = sorted(X["year"].unique())
    n_test_years = int(len(unique_years) * test_size)

    test_years = unique_years[-n_test_years:]  # Take the last n_test_years as test set
    train_years = unique_years[:-n_test_years]  # Take the rest as train set

    X_train = X[X["year"].isin(train_years)]
    X_test = X[X["year"].isin(test_years)]
    y_train = y[X["year"].isin(train_years)]
    y_test = y[X["year"].isin(test_years)]

    return X_train, X_test, y_train, y_test


def get_train_test_splits(test_size=0.2):
    df = load_data("data")

    df = preprocess_data(df)

    X, y = convert_and_split(df)
    X, y = derive_features(X, y)
    X_train, X_test, y_train, y_test = split_data_by_year(X, y, test_size=test_size)

    print(f"Train years: {X_train['year'].min()} to {X_train['year'].max()}")
    print(f"Test years: {X_test['year'].min()} to {X_test['year'].max()}")

    return X_train, X_test, y_train, y_test


def get_combined_dataset(X_train, y_train, X_test, y_test):
    X_total = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_total = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    X_total["set"] = ["train"] * len(X_train) + ["test"] * len(X_test)
    combined = X_total.copy()
    for col in y_total.columns:
        if col not in combined.columns:
            combined[col] = y_total[col]
    return combined
