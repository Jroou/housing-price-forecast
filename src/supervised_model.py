from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def data_preprocessing():
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "housing-in-london", "5", "kaggle_london_house_price_data.csv"))
    #clearing the data
    cols_to_drop = [
        "rentEstimate_lowerPrice",
        "rentEstimate_upperPrice",
        "saleEstimate_lowerPrice",
        "saleEstimate_upperPrice",
        "saleEstimate_ingestedAt",
        "saleEstimate_valueChange.saleDate",
        "saleEstimate_confidenceLevel",
        "saleEstimate_valueChange.numericChange",
        "saleEstimate_valueChange.percentageChange",
        "fullAddress",
        "history_percentageChange",
        "history_numericChange",
        "country",
        "history_date",
        "history_price",             
        "rentEstimate_currentPrice", 
        "postcode"
    ]

    df = df.drop(columns=cols_to_drop)
    df = df.dropna(subset=['saleEstimate_currentPrice'])

    y = df["saleEstimate_currentPrice"]
    X = df.drop(columns=["saleEstimate_currentPrice"])
    X = X.select_dtypes(include=['number', 'object', 'str'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tenure_mode = X_train['tenure'].mode()[0]
    property_mode = X_train['propertyType'].mode()[0]

    for data_set in [X_train, X_test]:
        data_set['tenure'] = data_set['tenure'].fillna(tenure_mode)
        data_set['propertyType'] = data_set['propertyType'].fillna(property_mode)
        data_set['currentEnergyRating'] = data_set['currentEnergyRating'].fillna('Missing')

    cols_to_fill = ['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms']

    for col in cols_to_fill:
        train_median = X_train[col].median()
        X_train[col] = X_train[col].fillna(train_median)
        X_test[col] = X_test[col].fillna(train_median)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    train_data = X_train.copy()
    train_data['saleEstimate_currentPrice'] = y_train_log

    outcode_means = train_data.groupby('outcode')['saleEstimate_currentPrice'].mean()
    y_train_log_mean = y_train_log.mean()

    for data_set in [X_train, X_test]:
        data_set['outcode'] = data_set['outcode'].map(outcode_means)
        data_set['outcode'] = data_set['outcode'].fillna(y_train_log_mean)
    

    features_to_ohe = ['tenure', 'propertyType', 'currentEnergyRating']

    preprocessor = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(sparse_output=False, drop='first'), features_to_ohe)
        ],
        remainder='passthrough'
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    y_test_log = np.log1p(y_test)

    return X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_test, y, df, preprocessor

X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_test, y, df, preprocessor = data_preprocessing()

model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train_log)

y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

mse = mean_squared_error(y_test, y_pred)

print(f"Mean of Sale Estimate Current Price: {y.mean()}")
print(f"Root Mean Squared Error: {mse**0.5:.2f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

