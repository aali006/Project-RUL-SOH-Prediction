import pandas as pd
from src.config import CSV_PATH

def load_battery_data():
    df = pd.read_csv(CSV_PATH)
    battery_id = 1
    battery_ids = []

    for rul in df['RUL']:
        battery_ids.append(battery_id)
        if rul == 0:
            battery_id += 1
    df['Battery ID'] = battery_ids
    return df
