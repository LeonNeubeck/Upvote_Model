import numpy as np
import pandas as pd
import keras

from gensim.models import Word2Vec
from final_preprocessor import count_len, transform_timestamp, preprocessing, embedding

word2vec = Word2Vec.load("../models/w2v_150k")

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
    #padding?
    return { "input_Im": X_im, "input_size_im": X_im_size, "input_size_title": X_t_size,"input_timestep":X_timestep,"input_NLP": X_NLP}



def predict(time_stamp, image_arr, image_size, title, model_name):




    X_dict = pred_preproc(time_stamp, image_arr, image_size, title)
    model = keras.models.load_model(f'../models/{model_name}.h5')
    y_pred = model.predict(X_dict)
    return y_pred
