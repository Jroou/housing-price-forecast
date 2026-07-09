import matplotlib.pyplot as plt
import seaborn as sns

def col_graphs(df):
    corr_matrix = df.corr(numeric_only=True) 
    correlations = corr_matrix["saleEstimate_currentPrice"].drop("saleEstimate_currentPrice")

    top_n = 6
    top_correlations = correlations.abs().sort_values(ascending=False).head(top_n)

    top_features = top_correlations.index
    real_correlations = correlations[top_features]

    num_features = len(top_features)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        sns.regplot(x=feature, y="saleEstimate_currentPrice", data=df, ax=axes[i], scatter_kws={'alpha': 0.1, 's': 5})
        
        corr_value = real_correlations[feature]
        axes[i].set_title(f'{feature} (corr: {corr_value:.2f})')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()