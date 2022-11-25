from fastapi import FastAPI
from scripts import TimestampPreprocessing as tp
import pandas as pd
from tensorflow import keras

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


#Sample url = http://localhost:8000/getPrediction?timestamp=2012-04-22T04:20:11&title=the%20best%20dog%20ever&url=imgururl.jpg
#Url output = {"prediction":0,"timestamp":"2012-04-22T04:20:11","title":"the best dog ever","url":"imgururl.jpg"}
@app.get("/getPrediction")
def timestampPrediction(timestamp, title, url):
    return {'prediction':0,
            'timestamp':timestamp,
            'title':title,
            'url':url}


@app.get("/timestampPrediction")
def timestampPrediction(timestamp):
    '''Returns probability of getting 30+ upvotes based on timestamp'''
    timestamp = pd.to_datetime(timestamp)
    df = pd.DataFrame({'time_stamp':[timestamp]})
    df = tp.transform_timestamp(df)
    for column in range(0,7):
        if column not in df.columns:
            df[column] = 0
    df = df.reindex(columns=['year','sin_month','cos_month','sin_day','cos_day',
                             'sin_hour','cos_hour','sin_minute','cos_minute',
                             0, 1, 2, 3, 4, 5, 6])
    model = keras.models.load_model('models/timestamp_model')
    result = model.predict(df)
    return {
        'input':timestamp,
        'prediction':result.tolist()[0],
        }

@app.get("/returnTitle")
def returnTitle(title):
    return {'title':title}
