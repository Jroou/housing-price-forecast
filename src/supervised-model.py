import pandas as pd
import numpy as np
import os


df = pd.read_csv(os.path.join(os.getcwd(), "data", "housing-in-london", "housing_in_london_yearly_variables.csv"))
#sort the cols in the alphabetical order
common_cols = sorted(set(df.columns))
#using pattern
df = df[common_cols]

print(df.head(), '\n', df["life_satisfaction"])

#clearing missing values
df = df.dropna()

print(df.head(), '\n', df["life_satisfaction"])

