import numpy as np

def remove_constant_features(df, sensor_columns):
    return [col for col in sensor_columns if df[col].min() != df[col].max()]

def remove_high_corr_features(df, sensor_columns, threshold=0.95):
    corr_matrix = df[sensor_columns].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return to_drop
