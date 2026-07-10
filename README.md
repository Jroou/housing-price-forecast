# 🏠 London Housing Price Forecast

This project implements an automated machine learning pipeline to forecast real estate prices in London. The model analyzes property features (floor area, neighborhood, property type, energy rating) and predicts prices using a regularized linear regression model (Ridge).

## 📊 Data

The project is built on open data from the UK real estate market.
* **Source:** [Kaggle: London House Price Data](https://www.kaggle.com/datasets/jakewright/house-price-data)

## 🛠 Tech Stack

* **Language:** Python 3.12+
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Ridge Regression, Target Encoding, OneHotEncoder, StandardScaler)
* **Visualization:** `matplotlib`, `seaborn`

## ⚙️ Pipeline Features

* **Data Leakage Prevention:** Removed historical prices and rent estimates to ensure realistic and fair forecasting.
* **Categorical Feature Engineering:** Applied Target Encoding for London postal codes (outcodes) and One-Hot Encoding for property types.
* **Log Transformation:** The target variable (price) was transformed using `np.log1p()` to handle price distribution skewness and super-prime real estate outliers.
* **Model Evaluation:** Evaluated using RMSE and MAE metrics, alongside *Actual vs. Predicted* scatter plots to detect overfitting.

## 📂 Project Structure

* `main.py` — The entry point of the application that integrates data loading, model training, and evaluation.
* `src/data_preprocessing.py` — Contains the core logic for data cleaning, missing value imputation, and feature matrix preparation.
* `src/graphs.py` & `src/graph.py` — Visualization modules for generating correlation matrices, feature distributions, and prediction results.

## 🚀 How to Run

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt