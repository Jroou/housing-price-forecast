from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def data():
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
    ]
    df = df.drop(columns=cols_to_drop)

    y = df["saleEstimate_currentPrice"]
    X = df.drop(columns=["saleEstimate_currentPrice"])

    X = X.select_dtypes(include=['number'])
    return df, X, y

df, X, y = data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['tenure'] = df['tenure'].fillna(df['tenure'].mode()[0])
df['propertyType'] = df['propertyType'].fillna(df['propertyType'].mode()[0])
df['currentEnergyRating'] = df['currentEnergyRating'].fillna('Missing')
df = df.dropna(subset=['rentEstimate_currentPrice', 'saleEstimate_currentPrice'])

cols_to_fill = ['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms']
for col in cols_to_fill:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

print(df.isnull().sum())



#model = LinearRegression()
#model.fit(X_train, y_train)
#
#y_pred = model.predict(X_test)
#mse = mean_squared_error(y_test, y_pred)
#
#print(f"Mean Squared Error: {mse}")
