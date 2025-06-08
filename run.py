from src.data_loader import load_battery_data
from src.utils import split_train_test, scale_data
from src.feature_engineering import remove_constant_features, remove_high_corr_features
from src.model_training import train_and_evaluate_model

import pandas as pd

# Step 1: Load Data
df = load_battery_data()
sensor_columns = df.columns[1:-2]  # adjust as needed

# Step 2: Split Train/Test
train_df, test_df = split_train_test(df)

# Step 3: Feature Engineering
const_cols = [col for col in sensor_columns if train_df[col].min() == train_df[col].max()]
train_df.drop(columns=const_cols, inplace=True)
test_df.drop(columns=const_cols, inplace=True)

corr_cols = remove_high_corr_features(train_df, train_df.columns[1:-2])
train_df.drop(columns=corr_cols, inplace=True)
test_df.drop(columns=corr_cols, inplace=True)

# Step 4: Prepare Data
X_train = train_df.iloc[:, :-2].values
y_train = train_df['RUL'].values
X_test = test_df.iloc[:, :-2].values
y_test = test_df['RUL'].values
X_train, X_test = scale_data(X_train, X_test)

# Step 5: Train & Evaluate
models = [
    KNeighborsRegressor(n_neighbors=3),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor()
]

results = []
for model in models:
    res = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    results.append(res)

performance_df = pd.DataFrame(results)
print(performance_df)
