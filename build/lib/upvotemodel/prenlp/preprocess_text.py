import numpy as np
import pandas as pd
import re
import pickle
import string
#### contraction ####
from contractions import CONTRACTION_MAP

from gensim.models import Word2Vec
### nltk ###
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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


def embedding(text,w2v_path="../../models/w2v_150k"):
    # 5. Embedding
    word2vec = Word2Vec.load(w2v_path)
    wv = word2vec.wv
    to_array = []
    for word in text:
        if word in wv.key_to_index:
            to_array.append(wv[word])
    return np.array(to_array)

### Setting Output Y ###
def binary_cat_upvotes(original_df, threshold=15):
    """
    Takes column from df called 'upvotes' and returns df with new column
    'cat_upvotes' which is 1 if upvotes is above threshold, and 0 otherwise.
    """
    dataframe = original_df.copy()
    if 'upvotes' not in original_df.columns:
        raise ValueError("df has no column named 'upvotes'")
    def trans(number):
        if number >= threshold:
            return 1
        else:
            return 0
    dataframe['cat_upvotes'] = dataframe['upvotes'].apply(trans)
    return dataframe
