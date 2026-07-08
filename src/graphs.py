import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from supervised_model import data

df, X, y = data()
#exclude non-numeric columns for correlation analysis, need to be changed with categorical encoding for future analysis
df = df.select_dtypes(exclude=['object'])

corr_matrix = df.corr()
correlations = corr_matrix["saleEstimate_currentPrice"].drop("saleEstimate_currentPrice")
num_features = len(correlations)
num_cols = 3
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, feature in enumerate(correlations.index):
    sns.regplot(x=feature, y=y, data=df, ax=axes[i], scatter_kws={'alpha': 0.1, 's': 5})
    axes[i].set_title(f'Кореляція saleEstimate_currentPrice з {feature}')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()