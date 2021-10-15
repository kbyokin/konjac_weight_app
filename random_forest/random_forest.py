import operator

import numpy as np
import pandas as pd
import seaborn as sns
import csv
import pickle
from joblib import dump, load
import matplotlib.pyplot as plt
from scipy import stats
from dtreeviz.trees import tree, dtreeviz

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, ShuffleSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor

# from .evaluation import evaluate

# load data
df = pd.read_csv("../datasets/area_800_.csv")

X = df[["area"]]
y = df["weighta"]

X = X.to_numpy()
y = y.to_numpy()

# split setting
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# cross-validation
for train_index, test_index in ss.split(X):
  X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
  
  rf_model = RandomForestRegressor(n_estimators=40, 
                                bootstrap=True, 
                                max_depth=90, 
                                max_features='sqrt', 
                                min_samples_leaf=4, 
                                min_samples_split=2)
                                
  rf_model.fit(X_train, y_train)
  prediction_rf = rf_model.predict(X_train)
#   print(evaluate("rf_train_3", y_train, prediction_rf))

# s = pickle.dumps(rf_model)

dump(rf_model, "rdf.joblib")