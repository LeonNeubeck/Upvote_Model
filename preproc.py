import pandas as pd
from scripts.final_preprocessor import preprocess
def run():
    df = pd.read_csv('balanced_35k.csv', index_col=0)

    X_dict ,y = preprocess(df)

    print(X_dict, y)
