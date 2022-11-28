import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# package for RNN model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision, Accuracy
from tensorflow.keras import regularizers, Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping



####### Preprocessing ########
### read dataset/sample data for modeling
data_path = '../raw_data/SEND_150k_data.csv'
data = pd.read_csv(data_path)
df = data.dropna().sample(30000,random_state=0)
### Add column: length of Title
df['title_len']=df['title'].apply(count_len)
### Preprocessing ###
df['preprocessing'] = df['title'].apply(lambda sentence: preprocessing(sentence))
### Embedding ###
df['embedding'] = df['lemmatize'].apply(lambda x: embedding(x,word2vec))
### Padding ###
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = 9 # avarage title length is about 8.5
df['padding'] = pad_sequences(df['embedding'], dtype='float32', padding='post', maxlen=max_length)
### Setting Binary Output Y
df = binary_cat_upvotes(df, 15)

########################
### Test_Train_Split ###
X = pad_sequences(df['padding'], dtype='float32', padding='post', maxlen=max_length)
y = df['cat_upvotes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#######################
## Build LSTM Model ###
l_model = Sequential()
l_model.add(layers.Masking())

# initializer = tf.keras.initializers.LecunUniform() #  TruncatedNormal()  ,kernel_initializer=initializer
initializer = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
l_model.add(layers.LSTM(32, activation = "tanh",kernel_initializer=initializer, kernel_regularizer='l1'))
l_model.add(layers.Dense(20, activation = "relu"))
l_model.add(layers.Dense(1, activation="sigmoid"))
# compile
l_model.compile(loss='binary_crossentropy',
              optimizer= 'rmsprop', # customize learning rate
              metrics='accuracy')
# fit model
es = EarlyStopping(patience=10, restore_best_weights=True)

l_history = l_model.fit(X_train, y_train,
    batch_size=8,
    epochs=1000,
    validation_split=0.3,
    shuffle=True,
    callbacks=[es])
