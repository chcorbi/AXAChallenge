import pickle
import pandas as pd
import numpy as np
import os.path as osp
from time import time

from toolbox import make_submission, load_submission, load_all_data
from regressor import Regressor

print "load dates..."
sub = load_submission("data/submission.txt")
pred_dates = sub.index
fit_dates = load_all_data().index
fit_dates = fit_dates.delete(range(18024)) # hack

print "make the prediction..."
reg = Regressor()
reg.fit(fit_dates)
pred = reg.predict(pred_dates)

print "write submission..."
make_submission(pred)
