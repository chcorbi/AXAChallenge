import csv
import pandas as pd
import numpy as np
import os.path as osp


DATADIR = "data/"
DATAFILENAME = "train_2011_2012_2013.csv"
SUBFILENAME = "submission.txt"
DATAPATH = osp.join(DATADIR, DATAFILENAME)
SUBPATH = osp.join(DATADIR, SUBFILENAME)

SUBSET_ASSIGNEMENT = ['CMS', 'Crises', 'Domicile', 'Gestion', \
  'Gestion - Accueil Telephonique', 'Gestion Assurances', \
  'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', \
  'M\xc3\xa9dical', 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP', \
  'Services', 'Tech. Axa', 'Tech. Inter', 'T\xc3\xa9l\xc3\xa9phonie', \
  'Tech. Total', 'M\xc3\xa9canicien', 'CAT', 'Manager', \
  'Gestion Clients', 'Gestion DZ', 'RTC', 'Prestataires']

ASSIGNEMENT = ['CAT', 'CMS', 'Crises', 'Domicile', 'Evenements', \
       'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex', \
       'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', \
       'Gestion Relation Clienteles', 'Gestion Renault', 'Japon', \
       'Manager', 'M\xc3\xa9canicien', 'M\xc3\xa9dical', 'Nuit', \
       'Prestataires', 'RENAULT', 'RTC', 'Regulation Medicale', 'SAP', \
       'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', \
       'T\xc3\xa9l\xc3\xa9phonie']


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
        data = [[header[0], header[12]]]
        count = 0
        for row in csv_reader:
            if max_lines and not (count < max_lines):
                break
            current_date = row[0].split(';')[0].split(' ')[0]
            if date_regex in current_date:
                line = row[0].split(';')
                data.append([line[0], line[12]]) # row is a one-elt list
                count += 1
    return pd.DataFrame(data[1:], columns=data[0])


def preprocessing(X_df):
    """Basic preprocessing. Return the basic-preprocessed DataFrame.
    """
    # drop useless features
    X_df = X_df[['DATE', 'ASS_ASSIGNMENT']]
    X_df.loc[:,'DATE'] = pd.to_datetime(X_df['DATE'])
    # sort the time serie
    idx = X_df['DATE'].argsort()
    X_df.reindex(idx)
    # binaries encoding assignement feature
    tmp_frame = pd.get_dummies(X_df['ASS_ASSIGNMENT'])
    X_df = pd.concat([X_df, tmp_frame], axis=1)
    X_df.drop('ASS_ASSIGNMENT', axis=1, inplace=True)
    # cumulate calls for the same 30min period
    X_df = X_df.groupby('DATE').sum()
    # add True if open hours (day)
    MINHOUR, MAXHOUR = 7, 19
    day = (X_df.index.hour >= MINHOUR) * (X_df.index.hour < MAXHOUR)
    X_df['day'] = day
    return X_df


def get_preprocess_data(data_path, date_regex, max_lines=int(1e6)):
    """Helper to get preprocess data
    """
    X_df = get_raw_data(data_path=data_path, date_regex=date_regex,
                        max_lines=max_lines)
    return preprocessing(X_df)


def _pred_by_prev_year(dates):
    """Predictor based on the year before.
    """
    dates_2012 = dates[dates.year == 2012]
    dates_2013 = dates[dates.year == 2013]
    X_df_2011 = pd.DataFrame.from_csv("datasets/2011.csv").drop('day', axis=1)
    X_df_2012 = pd.DataFrame.from_csv("datasets/2012.csv").drop('day', axis=1)
    f = lambda t: t.replace(year=2011)
    pred_2012_by_prev_year = X_df_2011.loc[dates_2012.map(f)]
    pred_2012_by_prev_year.index = dates_2012
    f = lambda t: t.replace(year=2012)
    pred_2013_by_prev_year = X_df_2012.loc[dates_2013.map(f)]
    pred_2013_by_prev_year.index = dates_2013

    return pd.concat([pred_2012_by_prev_year, pred_2013_by_prev_year], axis=0)


def _pred_by_evo_prev_week_(dates, period_type, stat):
    """Predictor based on the week before and the evolution.
    """
    pred_prev_week = _pred_by_prev_period(dates, period_type, 1, stat)
    pred_prevprev_week = _pred_by_prev_period(dates, period_type, 2, stat)
    evo_prev_week = pred_prevprev_week - pred_prev_week
    pred_prevprev_week += evo_prev_week
    return pred_prevprev_week


def _pred_by_evo_prev_year_(dates):
    """Predictor based on the year before and the evolution.
    """
    filename = "datasets/year_2011_stats_day.csv"
    year_2011_stats_day = pd.DataFrame.from_csv(filename)
    year_2011_mean_day = year_2011_stats_day.filter(regex=(".* mean"))
    filename = "datasets/year_2012_stats_day.csv"
    year_2012_stats_day = pd.DataFrame.from_csv(filename)
    year_2012_mean_day = year_2012_stats_day.filter(regex=(".* mean"))
    filename = "datasets/year_2013_stats_day.csv"
    year_2013_stats_day = pd.DataFrame.from_csv(filename)
    year_2013_mean_day = year_2013_stats_day.filter(regex=(".* mean"))

    evo_2013_2012_mean_day = year_2013_mean_day - year_2012_mean_day
    evo_2012_2011_mean_day = year_2012_mean_day - year_2011_mean_day

    pred_by_prev_year = _pred_by_prev_year(dates)

    n = pred_by_prev_year[pred_by_prev_year.index.year == 2012].shape[0],
    evo_2012_2011_mean_day = np.repeat(evo_2012_2011_mean_day.values, n,0)
    n = pred_by_prev_year[pred_by_prev_year.index.year == 2013].shape[0],
    evo_2013_2012_mean_day = np.repeat(evo_2013_2012_mean_day.values, n,0)

    pred_by_prev_year[pred_by_prev_year.index.year == 2012] += evo_2012_2011_mean_day
    pred_by_prev_year[pred_by_prev_year.index.year == 2013] += evo_2013_2012_mean_day

    return pred_by_prev_year


def _pred_by_prev_period(dates, period_type, n_prev, stat):
    """Predictor based on the stat of the period before.
    """
    dates_2012 = dates[dates.year == 2012]
    dates_2013 = dates[dates.year == 2013]

    day_mask = (dates_2013.hour>=7) * (dates_2013.hour<19)
    dates_2013_day = dates_2013[day_mask]
    dates_2013_night = dates_2013[np.invert(day_mask)]
    day_mask = (dates_2012.hour>=7) * (dates_2012.hour<19)
    dates_2012_day = dates_2012[day_mask]
    dates_2012_night = dates_2012[np.invert(day_mask)]
    if period_type == "week":
        first_per_2012 = dates_2012.week[0]-n_prev
        first_per_2013 = dates_2013.week[0]-n_prev
    if period_type == "month":
        first_per_2012 = dates_2012.month[0]-n_prev
        first_per_2013 = dates_2013.month[0]-n_prev
    filename = "datasets/" + period_type + "_2012_stats_day.csv"
    per_2012_stats_day = pd.DataFrame.from_csv(filename)
    filename = "datasets/" + period_type + "_2012_stats_night.csv"
    per_2012_stats_night = pd.DataFrame.from_csv(filename)
    filename = "datasets/" + period_type + "_2013_stats_day.csv"
    per_2013_stats_day = pd.DataFrame.from_csv(filename)
    filename = "datasets/" + period_type + "_2013_stats_night.csv"
    per_2013_stats_night = pd.DataFrame.from_csv(filename)

    stat_per_2013_day = per_2013_stats_day.filter(regex=(".* " + stat))
    stat_per_2013_day = stat_per_2013_day.drop("day " + stat, axis=1)
    stat_per_2013_day = stat_per_2013_day.loc[first_per_2013]

    stat_per_2013_night = per_2013_stats_night.filter(regex=(".* " + stat))
    stat_per_2013_night = stat_per_2013_night.drop("day " + stat, axis=1)
    stat_per_2013_night = stat_per_2013_night.loc[first_per_2013]

    stat_per_2012_day = per_2012_stats_day.filter(regex=(".* " + stat))
    stat_per_2012_day = stat_per_2012_day.drop("day " + stat, axis=1)
    stat_per_2012_day = stat_per_2012_day.loc[first_per_2013]

    stat_per_2012_night = per_2012_stats_night.filter(regex=(".* " + stat))
    stat_per_2012_night = stat_per_2012_night.drop("day " + stat, axis=1)
    stat_per_2012_night = stat_per_2012_night.loc[first_per_2013]
    assignements = ASSIGNEMENT
    pred_2013_by_stat_prev_per_day = pd.DataFrame(
                           np.repeat(stat_per_2013_day.values[None, :],
                                     len(dates_2013_day), 0),
                           columns=assignements,
                           index=dates_2013_day)
    pred_2013_by_stat_prev_per_night = pd.DataFrame(
                           np.repeat(stat_per_2013_night.values[None, :],
                                     len(dates_2013_night), 0),
                           columns=assignements,
                           index=dates_2013_night)
    pred_2013_by_stat_prev_per = pd.concat([pred_2013_by_stat_prev_per_night,
                                            pred_2013_by_stat_prev_per_day],
                                           axis=0)
    pred_2013_by_stat_prev_per.sort_index(inplace=True)

    pred_2012_by_stat_prev_per_day = pd.DataFrame(
                           np.repeat(stat_per_2012_day.values[None, :],
                                     len(dates_2012_day), 0),
                           columns=assignements,
                           index=dates_2012_day)
    pred_2012_by_stat_prev_per_night = pd.DataFrame(
                           np.repeat(stat_per_2012_night.values[None, :],
                                     len(dates_2012_night), 0),
                           columns=assignements,
                           index=dates_2012_night)
    pred_2012_by_stat_prev_per = pd.concat([pred_2012_by_stat_prev_per_night,
                                           pred_2012_by_stat_prev_per_day],
                                           axis=0)
    pred_2012_by_stat_prev_per.sort_index(inplace=True)

    return pd.concat([pred_2012_by_stat_prev_per, pred_2013_by_stat_prev_per], axis=0)


def make_prediction(dates):
    """Create a prediction for the given date.
    """
    pred = []
    # predict the same as the last year
    pred.append(_pred_by_prev_year(dates).fillna(method='ffill'))
    # predict the max of the week before the dates of prediction
    pred.append(_pred_by_prev_period(dates, "week", 1, "per80"))
    # predict the max of the month before the dates of prediction
    pred.append(_pred_by_prev_period(dates, "month", 1, "per80"))
    # predict the same as the last year with the year mean evolution
    pred.append(_pred_by_evo_prev_year_(dates).fillna(method='ffill'))
    # get the percentile max of the different prediction
    concat_pred = pd.concat(pred, axis=0)
    return concat_pred.groupby(concat_pred.index).max()


def make_submission(sub, filename="submission.txt"):
    """Create a submission file based on the X_df DataFrame.
    """
    date = pd.DataFrame(sub.index, columns=['DATE'])
    date['key'] = 1
    assignements = SUBSET_ASSIGNEMENT
    assignement = pd.DataFrame(assignements, columns=['ASS_ASSIGNMENT'])
    assignement['key'] = 1
    pred = pd.merge(date, assignement, on='key').drop('key', axis=1)
    pred['prediction'] = sub[assignements].values.ravel()
    pred.index = pred.DATE
    pred.drop('DATE', axis=1, inplace=True)
    pred.to_csv(path_or_buf=filename, sep='\t')


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
    sub.prediction = sub.prediction.astype(int)
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
    return (np.exp(tmp) - tmp - np.ones_like(tmp)).sum()


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
