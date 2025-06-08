import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    predictions = model.predict(X_test)
    end_predict = time.time()

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    accuracy = (np.abs(y_test - predictions) <= 10).sum() / len(y_test) * 100

    return {
        'model': model.__class__.__name__,
        'R2': r2,
        'RMSE': rmse,
        'Accuracy(±10)': accuracy,
        'Train Time': end_train - start,
        'Predict Time': end_predict - end_train,
        'Total Time': end_predict - start,
        'Prediction Example': predictions[0]
    }
