import pandas as pd
import numpy as np
import os.path as osp

from toolbox import get_raw_data, preprocessing

def get_stats(X_df):
    """Produce the stats on the all given dataset.
    """
    df = pd.DataFrame()
    header = X_df.columns
    new_stats = pd.DataFrame(X_df.mean(axis=0).values[None, :],
                             columns=list(header+" mean")) # mean
    df = pd.concat([df, new_stats], axis=1)
    new_stats = pd.DataFrame(X_df.var(axis=0).values[None, :],
                             columns=list(header+" var")) # variance
    df = pd.concat([df, new_stats], axis=1)
    new_stats = pd.DataFrame(X_df.min(axis=0).values[None, :],
                             columns=list(header+" min")) # min
    df = pd.concat([df, new_stats], axis=1)
    new_stats = pd.DataFrame(X_df.max(axis=0).values[None, :],
                             columns=list(header+" max")) # max
    df = pd.concat([df, new_stats], axis=1)
    new_stats = pd.DataFrame(np.percentile(X_df, 80, axis=0)[None, :],
                             columns=list(header+" per80")) # percentile 80%)
    df = pd.concat([df, new_stats], axis=1)
    new_stats = pd.DataFrame(np.percentile(X_df, 90, axis=0)[None, :],
                             columns=list(header+" per90")) # percentile 90%)
    return pd.concat([df, new_stats], axis=1)


def generate_preprocessed_data(data_path, saving_dir):
    """Generate a preprocessed dataset.
    """
    year_regexs = ['2011', '2012', '2013']
    for year_regex in year_regexs:
        X_df = get_raw_data(data_path=data_path, date_regex=year_regex)
        X_df = preprocessing(X_df)
        filename = osp.join(saving_dir, year_regex+".csv")
        X_df.to_csv(filename)


def generate_year_stats(data_path, saving_dir):
    """Generate three the stats produce by get_stats by year.
    """
    year_regexs = ['2011', '2012', '2013']
    for year_regex in year_regexs:
        X_df = get_raw_data(data_path=data_path, date_regex=year_regex)
        X_df = preprocessing(X_df)
        # add True if open hours (day)
        MINHOUR, MAXHOUR = 7, 19
        day = (X_df.index.hour >= MINHOUR) * (X_df.index.hour < MAXHOUR)
        X_df['day'] = day
        # day case
        X_df_day = X_df[X_df['day'] == True].drop('day', axis=1)
        df_day = get_stats(X_df_day)
        filename = osp.join(saving_dir, "year_"+year_regex+"_stats_day.csv")
        df_day.to_csv(filename)
        # night case
        X_df_night = X_df[X_df['day'] == False].drop('day', axis=1)
        df_night = get_stats(X_df_night)
        filename = osp.join(saving_dir, "year_"+year_regex+"_stats_night.csv")
        df_night.to_csv(filename)


def generate_month_stats(data_path, saving_dir):
    """Generate three the stats produce by get_stats by month.
    """
    year_regexs = ['2011', '2012', '2013']
    for year_regex in year_regexs:
        X_df = get_raw_data(data_path=data_path, date_regex=year_regex)
        X_df = preprocessing(X_df)
        # add True if open hours (day)
        MINHOUR, MAXHOUR = 7, 19
        day = (X_df.index.hour >= MINHOUR) * (X_df.index.hour < MAXHOUR)
        X_df['day'] = day
        X_df['Month'] = X_df.index.month
        months = X_df['Month'].unique()
        # day case
        X_df_day = X_df[X_df['day'] == True].drop('day', axis=1)
        month = months[0]
        X_df_month = X_df_day[X_df_day['Month'] == month].drop('Month', axis=1)
        df_month = get_stats(X_df_month)
        for month in months[1:]:
            X_df_month = X_df[X_df['Month'] == month].drop('Month', axis=1)
            df_tmp = get_stats(X_df_month)
            df_month = pd.concat([df_month, df_tmp], axis=0)
        df_month.index = months
        filename = osp.join(saving_dir, "month_"+year_regex+"_stats_day.csv")
        df_month.to_csv(filename)
        # night case
        X_df_night = X_df[X_df['day'] == False].drop('day', axis=1)
        month = months[0]
        X_df_month = X_df_night[X_df_night['Month'] == month].drop('Month', axis=1)
        df_month = get_stats(X_df_month)
        for month in months[1:]:
            X_df_month = X_df[X_df['Month'] == month].drop('Month', axis=1)
            df_tmp = get_stats(X_df_month)
            df_month = pd.concat([df_month, df_tmp], axis=0)
        df_month.index = months
        filename = osp.join(saving_dir, "month_"+year_regex+"_stats_night.csv")
        df_month.to_csv(filename)


def generate_week_stats(data_path, saving_dir):
    """Generate three the stats produce by get_stats by week.
    """
    year_regexs = ['2011', '2012', '2013']
    for year_regex in year_regexs:
        X_df = get_raw_data(data_path=data_path, date_regex=year_regex)
        X_df = preprocessing(X_df)
        # add True if open hours (day)
        MINHOUR, MAXHOUR = 7, 19
        day = (X_df.index.hour >= MINHOUR) * (X_df.index.hour < MAXHOUR)
        X_df['day'] = day
        X_df['Week'] = X_df.index.week
        weeks = X_df['Week'].unique()
        # day case
        X_df_day = X_df[X_df['day'] == True].drop('day', axis=1)
        week = weeks[0]
        X_df_week = X_df_day[X_df_day['Week'] == week].drop('Week', axis=1)
        df_week = get_stats(X_df_week)
        for week in weeks[1:]:
            X_df_week = X_df[X_df['Week'] == week].drop('Week', axis=1)
            df_tmp = get_stats(X_df_week)
            df_week = pd.concat([df_week, df_tmp], axis=0)
        df_week.index = weeks
        filename = osp.join(saving_dir, "week_"+year_regex+"_stats_day.csv")
        df_week.to_csv(filename)
        # night case
        X_df_night = X_df[X_df['day'] == False].drop('day', axis=1)
        week = weeks[0]
        X_df_week = X_df_night[X_df_night['Week'] == week].drop('Week', axis=1)
        df_week = get_stats(X_df_week)
        for week in weeks[1:]:
            X_df_week = X_df[X_df['Week'] == week].drop('Week', axis=1)
            df_tmp = get_stats(X_df_week)
            df_week = pd.concat([df_week, df_tmp], axis=0)
        df_week.index = weeks
        filename = osp.join(saving_dir, "week_"+year_regex+"_stats_night.csv")
        df_week.to_csv(filename)
