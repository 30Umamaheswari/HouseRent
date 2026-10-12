"""
Train the House Rent model and save ONE pipeline file (house_rent_pipeline.pkl)
that the Streamlit app (house.py) loads.

Why a pipeline: encoding + model are saved together, and inputs are matched by
column NAME, so the app can never send features in the wrong order.

Run:  python train_model.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------- features
NUMERIC_FEATURES = [
    'Property Size', 'Property Age', 'No of Bathroom', 'No of Cup Board',
    'Floor No ', 'Total No of Floor', 'Balconies',
]
AMENITY_FEATURES = [  # 0/1 columns
    'Negotiable', 'Lift', 'Gym', 'Internet', 'Air Conditioning', 'Club',
    'Intercom', 'Swimming Pool', 'Car Parking', 'Servent', 'Security',
    'Shopping Center', 'Gas Pipeline', 'Park', 'Rain Water Harvesting',
    'Sewage Treatment Plant', 'House Keeping', 'Power Backup', 'Visitor Parking',
]
CATEGORICAL_FEATURES = [
    'House Type', 'Lease Type', 'Furnishing', 'Parking', 'House Facing',
    'Water Supply', 'Building Type',
]
FEATURES = NUMERIC_FEATURES + AMENITY_FEATURES + CATEGORICAL_FEATURES
TARGET = 'Rent'

# ---------------------------------------------------------------- data
df = pd.read_excel('Data Set/HR_Train.xlsx')
df['House Type'] = df['House Type'].str.upper()          # 'bhk2' -> 'BHK2'

# Remove impossible values only in TRAINING data (never drop test rows)
df = df[(df['Property Size'] >= 100) & (df['Property Size'] <= 10000)]

X, y = df[FEATURES], df[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
], remainder='passthrough')                              # numeric + amenities as-is


def evaluate(name, model):
    pipe = Pipeline([('prep', preprocess), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    mae, r2 = mean_absolute_error(y_val, pred), r2_score(y_val, pred)
    print(f'{name:<22} MAE = Rs.{mae:,.0f}   R2 = {r2:.3f}')
    return pipe, mae


baseline_mae = mean_absolute_error(y_val, np.full(len(y_val), y_train.mean()))
print(f'{"Baseline (mean rent)":<22} MAE = Rs.{baseline_mae:,.0f}')
evaluate('Linear Regression', LinearRegression())
evaluate('Random Forest', RandomForestRegressor(
    n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1))
evaluate('Gradient Boosting', HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, random_state=42))

# Final model: Gradient Boosting (as accurate as Random Forest, but the saved
# file is under 1 MB instead of ~100 MB, so it fits on GitHub / Streamlit Cloud)
final = Pipeline([('prep', preprocess), ('model', HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, random_state=42))])
final.fit(X, y)

with open('house_rent_pipeline.pkl', 'wb') as f:
    pickle.dump({
        'pipeline': final,
        'features': FEATURES,
        'categories': {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
    }, f)
print('Saved house_rent_pipeline.pkl')
