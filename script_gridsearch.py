import pickle
import pandas as pd
import numpy as np
import os.path as osp
from time import time

from toolbox import load_all_data, load_submission
from regressor import Regressor

T0 = time()


print "load dataset..."
X_df_2011 = pd.DataFrame.from_csv("datasets/2011.csv")
X_df_2012 = pd.DataFrame.from_csv("datasets/2012.csv")
X_df_2013 = pd.DataFrame.from_csv("datasets/2013.csv")
X_df = pd.concat([X_df_2011, X_df_2012, X_df_2013], axis=0)

print "load dates..."
with open("target_dates_1.pkl") as f:
    dates = pickle.load(f)
    # date n1677, n3051 and n3451 cause trouble
    dates = dates.delete([1677, 3051, 3451])
sub = load_submission("data/submission.txt")
pred_dates = sub.index
fit_dates = load_all_data().index

fit_dates = fit_dates.delete(range(18024)) # hack

print "loading searching parameters..."
# Implement parameters to grid search here
n_estimators = range(300,900,50) 
param_grid = [
    {
        'n_estimators': n_estimators,
    }]
    
print "start gridsearch..."
# make prediction
reg = Regressor()
reg.gridsearch(fit_dates, param_grid)





