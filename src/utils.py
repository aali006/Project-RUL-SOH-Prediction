import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def split_train_test(df, id_column='Battery ID', threshold=9):
    train = df[df[id_column] < threshold]
    test = df[df[id_column] >= threshold]
    return train, test

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)
