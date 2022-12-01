import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import pandas as pd
import numpy as np

# Multiple Inputs usin https://machinelearningmastery.com/keras-functional-api-deep-learning/
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Masking
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Normalization



import keras
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator


from scripts.final_preprocessor import preprocess
from models import loader
BATCH_SIZE = 32

def initialize_model(learning=0.001):
    #Image convolution branch
    input_Im = Input(shape=(128,128,3), name="input_Im")
    conv1 = Conv2D(128, kernel_size=(3, 3),activation='relu')(input_Im)#64
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3),activation='relu')(pool1)#32
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(32, kernel_size=(3, 3),activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv2 = Conv2D(16, kernel_size=(3, 3),activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv2)
    flat1 = Flatten()(pool4)

    #image_size branch
    input_size_im = Input(shape=(1,), name="input_size_im")
    hidden1 = Dense(1, activation='relu')(input_size_im)
    flat2 = Flatten()(hidden1)



    initializer = keras.initializers.VarianceScaling(scale=0.1, mode="fan_in", distribution="uniform")
    #NLP branch
    input_NLP = Input(shape = (10,40), name="input_NLP")###dont know dim padded and embedded inputs
    mask = Masking()(input_NLP)
    lstm = LSTM(64, activation = "tanh",kernel_initializer=initializer, kernel_regularizer="l1")(mask)#32
    dense1 = Dense(32, activation = "relu")(lstm)#20
    flat3 = Flatten()(dense1)

    #title_size branch
    input_size_title = Input(shape=(1,), name="input_size_title")
    layer1 = Dense(1, activation='relu')(input_size_title)
    flat4 = Flatten()(layer1)


    #normalizer = Normalization()
    #normalizer.adapt("X_train")
    #davids timestep
    input_timestep = Input(shape=(16,),name="input_timestep")#dont know dims
    norm = Normalization()(input_timestep)
    step1 = Dense(64, activation='relu')(norm)#32
    #drop1 = Dropout(0.3)(step1)
    step2 = Dense(32, activation='relu')(step1)#26
    #drop2 = Dropout(0.2)(step2)
    step3 = Dense(16, activation='relu')(step2)#8
    #drop3 = Dropout(0.2)(step3)
    step4 = Dense(8, activation='relu')(step3)#4
    #drop4 = Dropout(0.2)(step4)
    #drop4 = Dropout(0.1)(step5)
    flat5 = Flatten()(step4)




    #concat them
    merge = concatenate([flat1, flat2, flat3, flat4, flat5])
    final1 = Dense(156, activation='relu')(merge)#128
    final2 = Dense(64, activation='relu')(final1)
    final2 = Dense(16, activation='relu')(final1)
    output = Dense(6, activation='softmax')(final2)


    #final model
    model = Model(inputs=[input_Im, input_size_im, input_NLP, input_size_title, input_timestep], outputs=output)
    #optimizer = keras.optimizers.experimental.RMSprop()
    #print(model.summary())
    model.compile(loss='categorical_crossentropy',
                optimizer="adam",
                metrics=['accuracy'])


    return model

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def createGenerator(dff, batch_size=BATCH_SIZE):
    #dff["y_cat"] = dff["y_cat"].astype("string")
    # Shuffles the dataframe, and so the batches as well
    dff = dff.sample(frac=1)

    # Shuffle=False is EXTREMELY important to keep order of image and coord
    flow = datagen.flow_from_dataframe(
                                        dataframe=dff,
                                        directory=None,
                                        x_col="image_path",
                                        y_col="y_cat",
                                        batch_size=batch_size,
                                        shuffle=False,
                                        class_mode="categorical",
                                        target_size=(128,128),
                                        seed=42
                                      )
    idx = 0
    n = len(dff) - batch_size
    batch = 0
    while True :
        # Get next batch of images
        X1 = flow.next()
        # idx to reach
        end = idx + X1[0].shape[0]
        # get next batch of lines from df
        X_im_size = dff["size"].iloc[idx:end].to_numpy()
        X_timestep = dff[["year", "sin_month", "cos_month", "sin_day", "cos_day", "sin_hour", "cos_hour", "sin_minute","cos_minute", 0, 1, 2, 3,4, 5, 6]].iloc[idx:end].to_numpy()
        X_t_size = dff["title_len"].iloc[idx:end].to_numpy()
        X_NLP = dff["padding"].iloc[idx:end]
        X_NLP =[np.expand_dims(x, axis=0) for x in X_NLP]
        X_NLP = np.array(X_NLP)
        X_NLP = np.concatenate(X_NLP, axis = 0)
        # Updates the idx for the next batch
        idx = end
#         print("batch nb : ", batch, ",   batch_size : ", X1[0].shape[0])
        batch+=1
        # Checks if we are at the end of the dataframe
        if idx==len(dff):
#             print("END OF THE DATAFRAME\n")
            idx = 0
        y = X1[1]
        X_im = X1[0]
        # Yields the image, metadata & target batches
        yield { "input_Im": X_im, "input_size_im": X_im_size, "input_size_title": X_t_size,"input_timestep":X_timestep,"input_NLP": X_NLP},y




import os


def train_model( model_name, epochs = 100, new = False, old_model = "Model_predictor", learning = 0.001):
    if new:
        model = initialize_model(learning)
    else:
        model = loader.get_model(old_model)
        pass
    file_path  = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'data/balanced_35k.csv'))
    df = pd.read_csv(file_path, index_col=0)
    X_dict, y, df = preprocess(df)
    GENERATOR = createGenerator(df)
    checkpoint_path = f"/checkpoint/{model_name}.h5"
    cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only = True,
    save_freq=3*893*BATCH_SIZE)
    es = keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=20,
    restore_best_weights=True,
)
    model.fit(
    GENERATOR,
    epochs=epochs,
    batch_size = 32,
    steps_per_epoch=893,
    workers = 1,
    use_multiprocessing=False,
    callbacks = [cp_callback, es]

    #validation_data = GENERATOR_train
    )
    loader.save_model(model, model_name)
    return model

def run():
    train_model("model_testing_other_layers_maybe_final_2", epochs = 50, new = False, old_model = "model_testing_other_layers_maybe_final")
