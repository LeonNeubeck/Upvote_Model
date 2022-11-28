import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import time
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def basic(original_df,keep_timestamp=False):
    """
    Transforms 'time_stamp' column from df into individual components 'year',
    'month','day','weekday','hour','minute'
    """
    df = original_df.copy()

    if 'time_stamp' not in df.columns:
        raise ValueError("df has no column named 'time_stamp'")
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='s')

    df['year'] = df.time_stamp.dt.year
    df['month'] = df.time_stamp.dt.month
    df['day'] = df.time_stamp.dt.day
    df['weekday'] = df.time_stamp.dt.weekday
    df['hour'] = df.time_stamp.dt.hour
    df['minute'] = df.time_stamp.dt.minute

    if keep_timestamp is False:
        df = df.drop(columns='time_stamp')
    return df

def cyclize(original_df):
    """
    Transforms columns named 'month','day','hour','minute' into sin and cos
    cyclic values for use with machine learning models
    """
    df = original_df.copy()

    need_list = ['month','day','hour','minute']
    max_dict = {
        'month':12,
        'day': 31,
        'hour': 23,
        'minute': 59
    }

    for column in need_list:
        if column in df.columns:
            def sin_trans(number):
                return math.sin(number * (2. * math.pi / max_dict[column]))
            def cos_trans(number):
                return math.cos(number * (2. * math.pi / max_dict[column]))
            df['sin_' + column] = df[column].apply(sin_trans)
            df['cos_' + column] = df[column].apply(cos_trans)
            df = df.drop(columns=column, axis=1)

    return df

def encode_weekday(original_df, keep_weekday_column=False):
    """
    OneHotEncodes column from df column named 'weekday'
    """
    df = original_df.copy()

    enc = OneHotEncoder(handle_unknown='ignore')
    df_wkdy = pd.DataFrame(enc.fit_transform(df[['weekday']]).toarray())
    df = pd.concat([df.reset_index(), df_wkdy], axis=1)
    df = df.set_index('index')
    if keep_weekday_column==False:
        df = df.drop('weekday', axis=1)
    return df

def transform_timestamp(original_df):
    """
    Takes 'time_stamp' column from df and returns df preprocessed and
    ready for machine learning
    """
    df = original_df.copy()
    df = basic(df)
    df = cyclize(df)
    df = encode_weekday(df)
    if 'year' in df.columns:
        scaler = MinMaxScaler()
        df['year'] = scaler.fit_transform(df[['year']].copy())
    return df
