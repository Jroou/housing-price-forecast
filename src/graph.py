import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from supervised_model import data


df, X, y = data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
train_df["saleEstimate_currentPrice"].plot(kind="hist", bins=30, title="Distribution of Sale Estimate Current Price")
plt.show()
