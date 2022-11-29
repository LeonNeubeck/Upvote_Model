import numpy as np
import pandas as pd
import keras

from gensim.models import Word2Vec
from scripts.final_preprocessor import count_len, preprocessing, embedding, transform_timestamp_single_input
from keras.utils import pad_sequences
word2vec = Word2Vec.load("models/w2v_150k")

vec_size = 40
max_length = 10




def pred_preproc(time_stamp, image_arr, image_size, title):

    X_im = np.array([image_arr])
    X_im_size = np.array([image_size])
    X_t_size = np.array([count_len(title)])
    time_list = [time_stamp]
    time_df = pd.DataFrame(time_list)
    time_df = time_df.rename(columns={0:"time_stamp"})
    X_timestep = transform_timestamp_single_input(time_df)
    X_timestep = X_timestep[["year", "sin_month", "cos_month", "sin_day", "cos_day", "sin_hour", "cos_hour", "sin_minute","cos_minute", 0, 1, 2, 3, 4, 5, 6]].to_numpy()


    title_df = pd.DataFrame([title])
    title_df = title_df.rename(columns={0:"title"})

    title_df['preprocessing'] = title_df['title'].apply(lambda sentence: preprocessing(sentence))
    ## Embedding ###

    #word2vec = Word2Vec(sentences=df["preprocessing"], vector_size=vec_size, min_count=10, window=4)####CHANGE DIS
    title_df['embedding'] = title_df['preprocessing'].apply(lambda x: embedding(x,word2vec))
    ### Padding ###

    t = pad_sequences(title_df['embedding'], dtype='float32', padding='post', maxlen=max_length)
    tes = []
    for i in range(t.shape[0]):
        tes.append(t[i])
    title_df['padding'] = tes
    X_NLP = preprocessing(title)
    X_NLP = title_df["padding"]
    X_NLP =[np.expand_dims(x, axis=0) for x in X_NLP]
    X_NLP = np.array(X_NLP)
    X_NLP = np.concatenate(X_NLP, axis = 0)

    print(X_im.shape, X_im_size.shape,  X_t_size.shape, X_timestep.shape, X_NLP.shape)
    return { "input_Im": X_im, "input_size_im": X_im_size, "input_size_title": X_t_size,"input_timestep":X_timestep,"input_NLP": X_NLP}



def predict(time_stamp, image_arr, image_size, title, model):

    X_dict = pred_preproc(time_stamp, image_arr, image_size, title)
    print(X_dict)
    y_pred = model.predict(X_dict)
    return y_pred
