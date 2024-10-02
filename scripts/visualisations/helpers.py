import pandas as pd
import numpy as np

def average_years(df):
    # Identify numeric and non-numeric columns, excluding 'x' and 'y'
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['x', 'y'])
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # Group by x and y coordinates
    grouped = df.groupby(['x', 'y'])

    # Calculate mean for numeric columns
    numeric_mean = grouped[numeric_columns].mean()

    # Calculate mode for non-numeric columns
    non_numeric_mode = grouped[non_numeric_columns].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)

    # Combine the results
    result = pd.concat([numeric_mean, non_numeric_mode], axis=1).reset_index()

    # Drop the 'year' column if it exists
    if 'year' in result.columns:
        result = result.drop('year', axis=1)

    return result

def rescale_features(df, to_scale):
    """
    Rescale specified features in the DataFrame to a new range.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    to_scale (dict): A dictionary where keys are column names and values are tuples (min, max)
                    representing the desired range for each feature.

    Returns:
    pd.DataFrame: A new DataFrame with specified features rescaled.
    """
    df_scaled = df.copy()

    for feature, (new_min, new_max) in to_scale.items():
        if feature in df.columns:
            old_min, old_max = df[feature].min(), df[feature].max()
            df_scaled[feature] = (df[feature] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return df_scaled