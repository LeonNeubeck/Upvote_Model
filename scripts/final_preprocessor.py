import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import IPython
import urllib.request
from urllib.error import HTTPError
from PIL import UnidentifiedImageError
import requests
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" # get rid of all tensorflow warnings
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from os import listdir
#from gensim.models import Word2Vec###### <---- change
import datetime as dt
import time
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import re
import pickle
import string
#### contraction ####
### nltk ###
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#### might be wrong
from binglin import word2vec
from contractions import CONTRACTION_MAP



##Davids timestamper


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

###Binglins NLP


def count_len(text):
    # add a column to the dataframe, showing the length of each 'title'
    text = text.split(' ')
    length = len(text)
    return length

def preprocessing(text, contraction_mapping=CONTRACTION_MAP):

    # 1. Expand Contractions
    """Expand the contractions in English. e.g. I'm ==> I am"""
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    # 2. Basic Cleaning
    sentence = expanded_text.lower()
    sentence = ''.join(char for char in sentence if not char.isdigit())
    ## punctuation dictionary ##
    my_punc = string.punctuation
    my_punc += '—'
    my_punc += '“”’'
    ############################
    for punctuation in my_punc:
        sentence = sentence.replace(punctuation, '')
    sentence = sentence.strip()

    # 2. Remove Stopwords
    STOPWORDS = set(stopwords.words('english'))
    remove_s = " ".join([word for word in str(sentence).split() if word not in STOPWORDS])

    # 3. Word Tokenize
    word_tokens = word_tokenize(remove_s)

    # 4. Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_n = [lemmatizer.lemmatize(word,pos='n') for word in word_tokens]
    lemmatized_v = [lemmatizer.lemmatize(word,pos='v') for word in lemmatized_n]
    return lemmatized_v

def embedding(text,word2vec):
    # 5. Embedding
    word2vec,
    wv = word2vec.wv
    to_array = []
    for word in text:
        if word in wv.key_to_index:
            to_array.append(wv[word])
    return np.array(to_array)




#final preprocessor
def preprocess(data):
    df = data
    dataframe=[]

    for i in range(6):
    # get the path/directory
        folder_dir = f"images_for_model/category_{i}"
        for images in os.listdir(folder_dir):
            yeet = []
            path = os.path.join(folder_dir, images)
            image = Image.open(path)
            id_, size, upvotes = images.replace(".png", "").split("_")
            yeet.append(id_)
            yeet.append(size)
            arr = np.array(image)
            try:
                A,B,C = arr.shape
                if C == 4:
                    arr = arr[:,:,:3]
                    image = Image.fromarray(arr)
                    image.save(path)
                yeet.append(path)
                yeet.append(i)
                dataframe.append(yeet)
            except ValueError:
                os.remove(path)
    data_arrys =pd.DataFrame(dataframe)
    data_arrys.rename(columns={0 :'id', 1:"size", 2:"image_path", 3:"y_cat"}, inplace=True)
    #merge
    df = pd.merge(data_arrys, df)
    df = transform_timestamp(df)
    ### Add column: length of Title
    df['title_len']=df['title'].apply(count_len)
    ### Preprocessing ###
    df['preprocessing'] = df['title'].apply(lambda sentence: preprocessing(sentence))
    ## Embedding ###
    vec_size = 40
    max_length = 10
    #word2vec = Word2Vec(sentences=df["preprocessing"], vector_size=vec_size, min_count=10, window=4)####CHANGE DIS
    df['embedding'] = df['preprocessing'].apply(lambda x: embedding(x,word2vec))
    ### Padding ###

    t = pad_sequences(df['embedding'], dtype='float32', padding='post', maxlen=max_length)
    tes = []
    for i in range(t.shape[0]):
        tes.append(t[i])
    df['padding'] = tes

    X_im = df["image_path"]
    X_im_size = df["size"]
    X_timestep = df[["year", "sin_month", "cos_month", "sin_day", "cos_day", "sin_hour", "cos_hour", "sin_minute","cos_minute", 0, 1, 2, 3, 4, 5, 6]].values
    X_t_size = df["title_len"]


    X_NLP = df["padding"]
    X_NLP =[np.expand_dims(x, axis=0) for x in X_NLP]
    X_NLP = np.array(X_NLP)
    X_NLP = np.concatenate(X_NLP, axis = 0)


    y = df["y_cat"]
    return { "input_Im": X_im, "input_size_im": X_im_size, "input_size_title": X_t_size,"input_timestep":X_timestep,"input_NLP": X_NLP}, y
