import matplotlib.pyplot as plt

def distribution_graph(y_train_log):
    plt.figure(figsize=(10, 6))
    y_train_log.plot(kind="hist", bins=50, edgecolor="black", color="skyblue")

    plt.title("Distribution of Log-Transformed Sale Estimate Current Price")
    plt.xlabel("Log(Price)")
    plt.ylabel("Frequency")
    plt.show()
