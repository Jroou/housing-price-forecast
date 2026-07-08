from sklearn.model_selection import train_test_split
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
        "saleEstimate_valueChange.percentageChange"
    ]
    df = df.drop(columns=cols_to_drop)

    y = df["saleEstimate_currentPrice"]
    X = df.drop(columns=["saleEstimate_currentPrice"])

    return df, X, y

df, X, y = data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


