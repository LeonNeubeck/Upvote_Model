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
from keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from os import listdir
from gensim.models import Word2Vec
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
from keras.utils import pad_sequences
#### might be wrong


from models import loader
#from models.loader import get_Word2vec


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

word2vec = loader.get_Word2vec()
vec_size = 40
max_length = 10


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
        folder_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), f"images/category_{i}"))
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

    #word2vec = Word2Vec(sentences=df["preprocessing"], vector_size=vec_size, min_count=10, window=4)####CHANGE DIS
    df['embedding'] = df['preprocessing'].apply(lambda x: embedding(x,word2vec))
    ### Padding ###

    t = pad_sequences(df['embedding'], dtype='float32', padding='post', maxlen=max_length)
    tes = []
    for i in range(t.shape[0]):
        tes.append(t[i])
    df['padding'] = tes

    X_im = df["image_path"]
    df["size"] = df["size"].astype("float")
    X_im_size = df["size"]
    X_timestep = df[["year", "sin_month", "cos_month", "sin_day", "cos_day", "sin_hour", "cos_hour", "sin_minute","cos_minute", 0, 1, 2, 3, 4, 5, 6]].values
    X_t_size = df["title_len"]


    X_NLP = df["padding"]
    X_NLP =[np.expand_dims(x, axis=0) for x in X_NLP]
    X_NLP = np.array(X_NLP)
    X_NLP = np.concatenate(X_NLP, axis = 0)


    df["y_cat"] = df["y_cat"].astype("string")
    y = df["y_cat"]
    file_path  = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data/processed_df.csv'))
    df.to_csv(file_path)
    return { "input_Im": X_im, "input_size_im": X_im_size, "input_size_title": X_t_size,"input_timestep":X_timestep,"input_NLP": X_NLP}, y, df
