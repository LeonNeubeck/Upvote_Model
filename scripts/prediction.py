import numpy as np
import pandas as pd
import keras

from gensim.models import Word2Vec
from scripts.final_preprocessor import count_len, transform_timestamp, preprocessing, embedding
from keras.utils import pad_sequences
word2vec = Word2Vec.load("models/w2v_150k")

vec_size = 40
max_length = 10




def pred_preproc(time_stamp, image_arr, image_size, title):

    X_im = image_arr
    X_im_size = image_size
    X_t_size = count_len(title)
    time_list = [time_stamp]
    time_df = pd.DataFrame(time_list)
    time_df = time_df.rename(columns={0:"time_stamp"})
    X_timestep = transform_timestamp(time_df)

    X_NLP = preprocessing(title)
    X_NLP = embedding(title,word2vec)
    X_NLP = pad_sequences(X_NLP, dtype='float32', padding='post', maxlen=max_length)
    return { "input_Im": X_im, "input_size_im": X_im_size, "input_size_title": X_t_size,"input_timestep":X_timestep,"input_NLP": X_NLP}



def predict(time_stamp, image_arr, image_size, title, model):




    X_dict = pred_preproc(time_stamp, image_arr, image_size, title)
    y_pred = model.predict(X_dict)
    return y_pred
