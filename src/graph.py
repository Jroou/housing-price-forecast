import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from supervised_model import data_preprocessing
from supervised_model import data_preprocessing

X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_test, y, df = data_preprocessing()

plt.figure(figsize=(10, 6))
y_train_log.plot(kind="hist", bins=50, edgecolor="black", color="skyblue")

plt.title("Distribution of Log-Transformed Sale Estimate Current Price")
plt.xlabel("Log(Price)")
plt.ylabel("Frequency")
plt.show()
