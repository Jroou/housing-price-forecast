import pandas as pd
import numpy as np
import os

df = pd.read_csv(os.path.join(os.getcwd(), "data", "housing-in-london", "5", "kaggle_london_house_price_data.csv")) 
num_cols = df.shape[1]
print(num_cols)