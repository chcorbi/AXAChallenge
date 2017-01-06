import pandas as pd
import numpy as np
from progressbar import Percentage, Bar, RotatingMarker, ETA, ProgressBar

from constants import YEAR, WEEK, ASSIGNEMENT


def get_last(date, X_df, period_type, nlast=1, nbtry=4):
    """Helper to get the same prediction than last period_type.
    """
    for i in range(nbtry+1): # try at least nbtry times
        old_date = date - (nlast*period_type + i*WEEK)
        try:
            tmp = X_df.loc[old_date] # old_date may not be in X_df
        except Exception as e:
            continue # try one week before
        return tmp # ok return


def pred_last_week(dates, X_df):
    """Return the same prediction than last week.
    """
    pred = []
    for date in dates:
        pred.append( get_last(date, X_df, WEEK) )
    return pd.DataFrame(pred, index=dates)


def pred_evolast_week(dates, X_df):
    """Return the same prediction than last year.
    """
    pred = []
    for date in dates:
        lastw = get_last(date, X_df, WEEK, 1)
        last2w = get_last(date, X_df, WEEK, 2)
        tmp = lastw + (lastw - last2w)
        pred.append( tmp )
    return pd.DataFrame(pred, index=dates)


def pred_last_year(dates, X_df):
    """Return the same prediction than last year.
    """
    pred = []
    for date in dates:
        pred.append( get_last(date, X_df, YEAR) )
    return pd.DataFrame(pred, index=dates)


def pred_evolast_year(dates, X_df):
    """Return the same prediction than last year.
    """
    pred = []
    for date in dates:
        if date.year == 2013:
            lasty = get_last(date, X_df, YEAR, 1)
            last2y = get_last(date, X_df, YEAR, 2)
            tmp = lasty + (lasty - last2y)
            pred.append( tmp )
        elif date.year == 2012:
            lastw = get_last(date, X_df, WEEK, 1)
            lasty = get_last(date, X_df, YEAR, 1)
            tmp = lasty + (lastw - lasty)
            pred.append( tmp )
    return pd.DataFrame(pred, index=dates)
