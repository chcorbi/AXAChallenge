import pickle
import pandas as pd
import numpy as np
import os.path as osp
from time import time

from toolbox import make_prediction, make_submission, load_submission

print "load dates..."
sub = load_submission("data/submission.txt")
dates = sub.index
print "make the prediction..."
pred = make_prediction(dates)
print "write submission..."
make_submission(pred)
