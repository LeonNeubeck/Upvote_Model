from fastapi import FastAPI
from datetime import datetime
from scripts import TimestampPreprocessing as tp
import pandas as pd
import datetime as dt


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/transformTimestamp")
def transformTimestamp(timestamp):
    '''Used for testing. Returns timestamp transformed and ready
    to be put into our timestamp model.'''
    timestamp = pd.to_datetime(timestamp)
    df = pd.DataFrame({'time_stamp':[timestamp]})
    df = tp.transform_timestamp(df)
    for column in range(0,7):
        if column not in df.columns:
            df[column] = 0
    df = df.reindex(columns=['year','sin_month','cos_month','sin_day','cos_day',
                             'sin_hour','cos_hour','sin_minute','cos_minute',
                             0, 1, 2, 3, 4, 5, 6])
    return {'array':df.iloc[0].tolist()}
