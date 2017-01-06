import csv
import pandas as pd
import numpy as np
import os.path as osp
from itertools import izip

from predictors import *
from constants import ASSIGNEMENT, MONTH, WEEK


def get_raw_data(data_path, date_regex, max_lines=None):
    """
    Loaded the AXA dataset, The data is filter by the date_regex, only DATE and
    ASSIGNEMENT columns are keep.

    Parameters:
    ----------

    data_path: string, the data filepath

    date_regex: string, regular expression to select the date

    max_lines: int, maximuns lines to load

    Outputs:
    --------

    X: DataFrame of the given length

    """
    if not osp.exists(data_path):
        raise ValueError("data_path: datasets do not exist")
    if not ((max_lines is None) or (isinstance(max_lines, int))):
        raise ValueError("max_lines: should be int or None")
    if not isinstance(date_regex, str):
        raise ValueError("date_regex: regular Expression must be string")
    with open(data_path, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\n')
        header = csv_reader.next()[0].split(';') # get header
        data = [[header[0], header[12], header[81]]]
        count = 0
        for row in csv_reader:
            if max_lines and not (count < max_lines):
                break
            current_date = row[0].split(';')[0].split(' ')[0]
            if date_regex in current_date:
                line = row[0].split(';')
                data.append([line[0], line[12], line[81]]) # row is a one-elt list
                count += 1
    return pd.DataFrame(data[1:], columns=data[0])


def preprocessing(X_df):
    """Basic preprocessing. Return the basic-preprocessed DataFrame.
    """
    # drop useless features
    X_df = X_df[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']]
    X_df.loc[:,'DATE'] = pd.to_datetime(X_df['DATE'])
    # sort the time serie
    idx = X_df['DATE'].argsort()
    X_df.reindex(idx)
    # binaries encoding assignement feature
    X_df.CSPL_RECEIVED_CALLS = X_df.CSPL_RECEIVED_CALLS.values.astype(int)
    X_df = X_df.groupby(['DATE', 'ASS_ASSIGNMENT']).sum().reset_index()
    tmp_frame = pd.get_dummies(X_df['ASS_ASSIGNMENT'])
    for col in tmp_frame.columns:
        tmp_frame[col] = tmp_frame[col].values * X_df.CSPL_RECEIVED_CALLS
    X_df = pd.concat([X_df, tmp_frame], axis=1)
    X_df.drop('ASS_ASSIGNMENT', axis=1, inplace=True)
    X_df.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
    # cumulate calls for the same 30min period
    X_df = X_df.groupby('DATE').sum()
    return X_df


def make_prediction(dates):
    """Create a prediction for the given date.
    """
    X_df_2011 = pd.DataFrame.from_csv("datasets/2011.csv")
    X_df_2012 = pd.DataFrame.from_csv("datasets/2012.csv")
    X_df_2013 = pd.DataFrame.from_csv("datasets/2013.csv")
    X_df = pd.concat([X_df_2011, X_df_2012, X_df_2013], axis=0)
    pred = []
    pred.append(pred_last_week(dates, X_df))
    pred.append(pred_evolast_week(dates, X_df))
    pred.append(pred_last_year(dates, X_df))
    pred.append(pred_evolast_year(dates, X_df))
    concat_pred = pd.concat(pred, axis=0)
    concat_pred *= 1.22
    return concat_pred.groupby(concat_pred.index).max()


def make_submission(sub, filename="submission.txt"):
    """Create a submission file based on the X_df DataFrame.
    """
    date = pd.DataFrame(sub.index, columns=['DATE'])
    date['key'] = 1
    assignements = sub.columns
    assignement = pd.DataFrame(assignements, columns=['ASS_ASSIGNMENT'])
    assignement['key'] = 1
    pred = pd.merge(date, assignement, on='key').drop('key', axis=1)
    pred['prediction'] = sub[assignements].values.ravel()
    pred.index = pred.DATE
    pred.drop('DATE', axis=1, inplace=True)
    pred.to_csv(path_or_buf=filename, sep='\t')
    decimate_submission(fname_ref="data/submission.txt", fname=filename)


def decimate_submission(fname_ref="data/submission.txt", fname="submission.txt"):
    """Decimate the given submission to hold only the assignement in the ref one.
    """
    with open(fname_ref, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\n')
        data = [csv_reader.next()[0].split('\t')] # get header
        for row in csv_reader:
            data.append( row[0].split('\t') )
    sub = pd.DataFrame(data[1:], columns=data[0])
    with open(fname, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\n')
        data = [csv_reader.next()[0].split('\t')] # get header
        for row in csv_reader:
            data.append( row[0].split('\t') )
    pred = pd.DataFrame(data[1:], columns=data[0])
    npASSIGNEMENT = np.array(ASSIGNEMENT)
    mask = np.ndarray((0,))
    for date in np.unique(sub.DATE):
        current_assignement = np.array(sub[sub.DATE == date].ASS_ASSIGNMENT)
        res = np.searchsorted(npASSIGNEMENT, current_assignement)
        bin_res = np.zeros_like(npASSIGNEMENT, dtype=bool)
        bin_res[res] = True
        mask = np.r_[mask, bin_res]
    pred['mask'] = mask
    pred = pred[pred['mask'] == 1]
    pred.drop('mask', axis=1, inplace=True)
    pred.index = pred['DATE']
    pred.drop('DATE', axis=1, inplace=True)
    pred.to_csv(fname, sep='\t')


def load_submission(data_path):
    """Return a df_target from the submission file.
    """
    with open(data_path, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\n')
        data = [csv_reader.next()[0].split('\t')] # get header
        for row in csv_reader:
            data.append( row[0].split('\t') )
    sub = pd.DataFrame(data[1:], columns=data[0])
    assignements = np.unique(sub.ASS_ASSIGNMENT)
    sub.prediction = sub.prediction.astype(float)
    for assignement in assignements:
        sub[assignement] = sub.prediction*(sub.ASS_ASSIGNMENT == assignement)
    sub.drop(['ASS_ASSIGNMENT', 'prediction'], axis=1, inplace=True)
    sub.DATE = pd.to_datetime(sub.DATE)
    return sub.groupby('DATE').sum()


def get_error_from_files(data_path_pred, data_path_target, alpha=0.1):
    """Return the LinExp error for the given prediction and target from two
    submission file.
    """
    df_pred = load_submission(data_path_pred)
    df_target = load_submission(data_path_target)
    return get_error_df(df_pred, df_target, alpha=alpha)


def get_error_dfs(df_pred, df_target, alpha=0.1):
    """Return the LinExp error for the given prediction and target from two
    DataFrame.
    """
    tmp = alpha * (df_target - df_pred).values
    return (np.exp(tmp) - tmp - np.ones_like(tmp)).mean()
