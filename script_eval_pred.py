import pickle
import pandas as pd
import numpy as np
import os.path as osp
from time import time

from toolbox import make_prediction, get_error_dfs, load_submission

T0 = time()

with open("target_dates_1.pkl") as f:
    dates = pickle.load(f)
    # date n1677, n3051 and n3451 cause trouble
    dates = dates.delete([1677, 3051, 3451])

X_df_2011 = pd.DataFrame.from_csv("datasets/2011.csv")
X_df_2012 = pd.DataFrame.from_csv("datasets/2012.csv")
X_df_2013 = pd.DataFrame.from_csv("datasets/2013.csv")
X_df = pd.concat([X_df_2011, X_df_2012, X_df_2013], axis=0)

print "make the prediction..."
# make prediction
pred = make_prediction(dates)
print "acquire the true value..."
target = X_df.loc[dates]
print "compute error..."
# get the error
err = get_error_dfs(pred, target)
print "LinExp error: ", err, "run in :", time() - T0, "s"
