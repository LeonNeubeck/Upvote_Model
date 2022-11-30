from fastapi import FastAPI
from scripts import TimestampPreprocessing as tp
import pandas as pd
from tensorflow import keras
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from scripts.prediction import predict
from models.loader import get_model
from fastapi import FastAPI, File, UploadFile, Form
from keras.applications.resnet import preprocess_input

app = FastAPI()
model = get_model()

def base64_to_pil(img_str):
    if "base64," in img_str:
        img_str = img_str.split(",")[1]
    img_raw = base64.b64decode(img_str)
    img = Image.open(BytesIO(img_raw))

    return img


@app.get("/")
async def root():
    return {"message": "Hello World"}


#Sample url = http://localhost:8000/getPrediction?timestamp=2012-04-22T04:20:11&title=the%20best%20dog%20ever&url=imgururl.jpg
#Url output = {"prediction":0,"timestamp":"2012-04-22T04:20:11","title":"the best dog ever","url":"imgururl.jpg"}
@app.post("/getPrediction")
def getPrediction(title: str = Form(...), time_stamp: int = Form(...), image_size: int = Form(...), filedata: str = Form(...)): #time_stamp, title):
    img = base64_to_pil(filedata)
    im_size = image_size
    im_arr = np.array(img)
    A,B,C = im_arr.shape
    if C == 4:
        im_arr = im_arr[:,:,:3]
    print(im_arr)
    im_arr = preprocess_input(im_arr, data_format=None)
    print(im_arr)
    result_1 = predict(time_stamp, im_arr, im_size, title, model)
    result = result_1[0].tolist()
    print(result)
    max_value = max(result)
    max_index = result.index(max_value)
    print(result, max_index)
    return {'category': max_index,
            'probabilities':{
                "0": result[0],
                "1": result[1],
                "2": result[2],
                "3": result[3],
                "4": result[4],
                "5": result[5]

            }
            }




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
