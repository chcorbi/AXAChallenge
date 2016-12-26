import csv
from datetime import datetime
import pandas as pd
import os.path as osp


data_dir = "data/"
data_filename = "train_2011_2012_2013.csv"
data_path = osp.join(data_dir, data_filename)


def get_raw_data(data_path, date=None, n=500, r=0.2):
    """
    Loaded the AXA dataset, train and test. The dataset start at date 'date' and
    is n length. No preprocess done on the data.

    Parameters:
    ----------

    data_path: string, the data filepath

    date: string or datetime classe, format:'%Y-%m-%d',
          date when to start the dataset, if ignored, the epoc will be the first
          date found, default None

    n: int, the dataset length, default 500

    r: float, the ratio of the test dataset, between 0.0 and 1.0, default 0.2

    Outputs:
    --------

    X: DataFrame of the given length

    """
    if isinstance(date, str):
        try:
            ref_date = datetime.strptime(row[0], '%Y-%m-%d')
        except Exception as e:
            print "Error on the format?"
            raise e
    if isinstance(date, datetime):
        ref_date = date
    if date is None:
        ref_date = date
    else:
        raise ValueError("Date arg: not the right type")

    with open(data_path, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\n')
        data = [csv_reader.next()[0].split(';')] # get header
        count = 0
        for row in csv_reader:
            if not (count < n):
                break
            current_date = row[0].split(';')[0].split(' ')[0]
            current_date = datetime.strptime(current_date, '%Y-%m-%d')
            if (ref_date is None) or (ref_date <= current_date):
                data.append(row[0].split(';')) # row is a one-elt list
                count += 1
    return pd.DataFrame(data[1:], columns=data[0])
