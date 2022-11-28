''' Linear Algebra'''
import pandas as pd
import numpy as np

''' Data visualization'''
import matplotlib.pyplot as plt
import warnings

''' Scikit-Learn'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

''' Tensorflow Keras'''
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

'''Other'''
import datetime as dt
import matplotlib.pyplot as plt
import time
import math

data_path = '../raw_data/100k_data.csv'
df_all = pd.read_csv(data_path)
df_all = df_all.dropna()
df_sample = df_all.dropna().sample(10000,random_state=0)

def binary_cat_upvotes(original_df, threshold=30):
    """
    Takes column from df called 'upvotes' and returns df with new column
    'cat_upvotes' which is 1 if upvotes is above threshold, and 0 otherwise.
    """
    df = original_df.copy()
    if 'upvotes' not in original_df.columns:
        raise ValueError("df has no column named 'upvotes'")
    def trans(number):
        if number >= threshold:
            return 1
        else:
            return 0
    df['cat_upvotes'] = df['upvotes'].apply(trans)
    return df

def multi_cat_upvotes(original_df, int_list=[10,100,1000]):
    """
    Takes column from df and returns df with new
    column 'cat_upvotes' based on list passed as an argument
    """
    df = original_df.copy()
    def trans(number):
        for index, integer in enumerate(int_list):
            if number < integer:
                return index
        return len(int_list)
    df['cat_upvotes'] = df['upvotes'].apply(trans)
    return df

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

df = df_all[['time_stamp','upvotes']]
df = transform_timestamp(df)
df = binary_cat_upvotes(df, threshold=30)
df = df.drop(columns='upvotes')

X = df.drop(columns='cat_upvotes')
y = df['cat_upvotes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

from tensorflow.keras.layers import Normalization
from tensorflow.keras.metrics import Recall, Precision

metrics = [
    keras.metrics.Recall(),
    keras.metrics.Precision(thresholds=0.38),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    keras.metrics.BinaryAccuracy(threshold=0.38)
]

def init_model():
    input_shape = X_train.shape[1:]
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = models.Sequential()
    model.add(normalizer)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    optimizer = Adam(learning_rate=0.0003)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)
    return model

model = init_model()

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=12,
                    batch_size=16,
                    shuffle=True)

y_pred_proba = model.predict(X_test)
threshold = 0.38 # 50%
y_pred_binary = np.where(y_pred_proba > threshold, 1, 0 )
cm = confusion_matrix(y_test,y_pred_binary)
