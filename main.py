from src.data_preprocessing import data_prep
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from src.graph import distribution_graph
from src.graph import distribution_graph, actual_vs_predicted_graph
import numpy as np

from src.graphs import col_graphs

X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_test, y, df, preprocessor = data_prep()

model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train_log)

y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean of Sale Estimate Current Price: {y.mean()}")
print(f"Root Mean Squared Error: {mse**0.5:.2f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

#col_graphs(df)
distribution_graph(y_train_log)
actual_vs_predicted_graph(model, X_train_scaled, X_test_scaled, y_train_log, y_test)