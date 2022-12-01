import keras
from gensim.models import Word2Vec
import os



##absolute path this

def get_model(name = "model_testing_other_layers_maybe_final"):
    file_path  = os.path.abspath(os.path.join(os.path.dirname( __file__ ), f'{name}.h5'))
    return keras.models.load_model(file_path)


def get_Word2vec():

    file_path  = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'w2v_150k'))
    return Word2Vec.load(file_path)

def save_model(model, name="Model_predictor_2"):
    file_path  = os.path.abspath(os.path.join(os.path.dirname( __file__ ), f'{name}.h5'))
    model.save(file_path)
    pass
