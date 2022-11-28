import streamlit as st
import datetime
import pandas as pd
import requests
from PIL import Image
import base64
from io import BytesIO
import time


CSS = """
h1 {
    color: red;
}
.stApp {
    background-image: url(https://bestlifeonline.com/wp-content/uploads/sites/3/2019/12/shutterstock_556211362.jpg?quality=82&strip=all);
    background-size: cover;
}
"""
if st.checkbox('Inject CSS'):
    st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)



st.markdown("""
    # Reddit Upvote Model 🐶
    ### Introduction:
""")
if st.checkbox('Show Description'):
    st.write('''
        This code will only be executed when the check box is checked

        Here is the description to introduce the Upvote model
        ''')

st.markdown("""

    # Evaluation 🧮
    #### Input Your Post:
    - Title
""")


### Title
st.markdown("""

    ##### Title
""")
title = st.text_input('Post title', '')


st.markdown("""

    ##### DateTime
""")
### Datetime
d = st.date_input(
    "Date",
    datetime.date(2022, 11, 10),
    )
t = st.time_input(
    "Time",
    datetime.time(8, 00)
)
st.write('DateTime:', d, t)

### Images
st.markdown("""

    ##### Image
""")

time_string = str(d)+ " " +str(t)

time_stamp = int(time.mktime(time.strptime(time_string, '%Y-%m-%d %H:%M:%S'))) - time.timezone
st.write(time_stamp)
def load_image(image_file):
	img = Image.open(image_file)
	return img

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose a PNG or JPEG file", type = ['png','jpg','jpeg'], accept_multiple_files =False)
show_file = st.empty
if not uploaded_file:
    st.write('Please upload a file:'.format(''.join(['png','jpg','jpeg'])))
else:

    st.write("start")
    img = Image.open(uploaded_file)
    im_file = BytesIO()
    img.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)

    st.write("done!!!")
    params = {
        "title": title,
        "image": im_b64,
        "time_stamp": time_stamp,
    }


if st.button('predict score'):
    st.write('Calculating...')
    r = requests.get(f"http://localhost:8000/getPrediction",params = params).json()
    st.write(f'{r}')
else:
    st.write('Click the button once all the data has been inputed')






# with st.form(key='params_for_api'):

#     post_date = st.date_input('Date', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
#     post_time = st.time_input('Time', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
#     pickup_datetime = f'{post_date} {post_time}'
#     title = st.text_input('Post title')
#     # images ...
#     passenger_count = st.number_input('passenger_count', min_value=1, max_value=8, step=1, value=1)

#     st.form_submit_button('Make prediction')
