import pandas as pd
import numpy as np

def binary_cat_upvotes(original_df, threshold=50):
    """
    Takes column from df called 'upvotes' and returns df with new column
    'cat_upvotes' which is 1 if upvotes is above threshold, and 0 otherwise.
    """
    df = original_df.copy()
    if 'upvotes' not in original_df.columns:
        raise ValueError("df has no column named 'upvotes'")
    def trans(number):
        if number >= threshold:
            return 1
        else:
            return 0
    df['cat_upvotes'] = df['upvotes'].apply(trans)
    return df

def multi_cat_upvotes(original_df, int_list=[2,15,30,100,500]):
    """
    Takes column from df and returns df with new
    column 'cat_upvotes' based on list passed as an argument
    """
    df = original_df.copy()
    def trans(number):
        for index, integer in enumerate(int_list):
            if number < integer:
                return index
        return len(int_list)
    df['cat_upvotes'] = df['upvotes'].apply(trans)
    return df
