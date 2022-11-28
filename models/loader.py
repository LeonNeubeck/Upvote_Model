import keras
from gensim.models import Word2Vec






def get_model(name = "Model_predictor"):
    return keras.models.load_model(f'{name}.h5')


def get_Word2vec():
    return Word2Vec.load("w2v_150k")

def save_model(model, name="Model_predictor"):
    model.save(f'{name}')
    pass
