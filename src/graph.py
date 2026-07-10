import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def distribution_graph(y_train_log):
    plt.figure(figsize=(10, 6))
    y_train_log.plot(kind="hist", bins=50, edgecolor="black", color="skyblue")

    plt.title("Distribution of Log-Transformed Sale Estimate Current Price")
    plt.xlabel("Log(Price)")
    plt.ylabel("Frequency")
    plt.show()

def actual_vs_predicted_graph(model, X_train_scaled, X_test_scaled, y_train_log, y_test):
    y_train_pred_log = model.predict(X_train_scaled)
    y_test_pred_log = model.predict(X_test_scaled)

    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)
    
    y_train_true = np.expm1(y_train_log) 

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=y_train_true, y=y_train_pred, ax=axes[0], alpha=0.3, color='royalblue', s=15)
    
    min_train = min(y_train_true.min(), y_train_pred.min())
    max_train = max(y_train_true.max(), y_train_pred.max())
    axes[0].plot([min_train, max_train], [min_train, max_train], '--b', linewidth=2)
    
    axes[0].set_title('Training Dataset')
    axes[0].set_xlabel('Actual price (£)')
    axes[0].set_ylabel('Forecasted price (£)')

    sns.scatterplot(x=y_test, y=y_test_pred, ax=axes[1], alpha=0.3, color='seagreen', s=15)
    
    min_test = min(y_test.min(), y_test_pred.min())
    max_test = max(y_test.max(), y_test_pred.max())
    axes[1].plot([min_test, max_test], [min_test, max_test], '--g', linewidth=2)
    
    axes[1].set_title('Testing Dataset')
    axes[1].set_xlabel('Actual price (£)')
    axes[1].set_ylabel('Forecasted price (£)')

    plt.tight_layout()
    plt.show()