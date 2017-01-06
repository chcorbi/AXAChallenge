import pandas as pd
import numpy as np
import os.path as osp
from time import time

from data_generate_toolbox import generate_year_stats, generate_month_stats, \
                                  generate_week_stats, \
                                  generate_preprocessed_data

data_dir = "data/"
data_filename = "train_2011_2012_2013.csv"
data_path = osp.join(data_dir, data_filename)
output_dir = "datasets/"

T0 = time() # run in approx 20min

print "running 'generate_preprocessed_data'..."
t0=time()
generate_preprocessed_data(data_path, output_dir)
print "--finish", time() - t0, "s"

print "running 'generate_year_stats'..."
t0=time()
generate_year_stats(data_path, output_dir)
print "--finish", time() - t0, "s"

print "running 'generate_month_stats'..."
t0=time()
generate_month_stats(data_path, output_dir)
print "--finish", time() - t0, "s"

print "running 'generate_week_stats'..."
t0=time()
generate_week_stats(data_path, output_dir)
print "--finish", time() - t0, "s"

print "21 files generated in", time() - T0, "s"
