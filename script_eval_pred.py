import pandas as pd
import numpy as np
import os.path as osp
from time import time

from toolbox import make_prediction, get_error_dfs, SUBSET_ASSIGNEMENT

T0 = time()

errors = []
for n_month in [3, 4, 5]:
    # choice a target
    print "choice a date target..."
    X_df_2012 = pd.DataFrame.from_csv("datasets/2012.csv").drop('day', axis=1)
    X_df_2013 = pd.DataFrame.from_csv("datasets/2013.csv").drop('day', axis=1)
    dates = X_df_2012.index[X_df_2012.index.month == n_month]
    dates = dates.append(X_df_2013.index[X_df_2013.index.month == n_month])
    print "make the prediction..."
    # make prediction
    pred = make_prediction(dates)
    print "acquire the true value..."
    target = pd.concat([X_df_2012[X_df_2012.index.month == n_month],
                        X_df_2013[X_df_2013.index.month == n_month]], axis=0)
    print "compute error..."
    # get the error
    err = get_error_dfs(pred[SUBSET_ASSIGNEMENT], target[SUBSET_ASSIGNEMENT])
    errors.append(err)
    print "LinExp error: ", err, "run in :", time() - T0, "s"

errors = np.array(errors)
print "Each err: ", errors
print "Mean err: ", errors.mean()
