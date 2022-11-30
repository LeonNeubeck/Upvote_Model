import streamlit as st
import datetime
import pandas as pd
import requests
from PIL import Image
import base64
from io import BytesIO
import time

import matplotlib.pyplot as plt
import seaborn as sns
from json import JSONDecodeError


CSS = """
h1 {
    color: black;
}
.stApp {
    background-image: url(https://static.vecteezy.com/system/resources/previews/005/230/466/non_2x/cute-pug-dog-cartoon-seamless-pattern-illustration-vector.jpg);
    # background-size: cover;
}
"""
# if st.checkbox('Inject CSS'):
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown("""
    # Reddit Upvote Model 🐶
    #
""")

### Title
st.markdown("""
    #####  Post Title
""")
title = st.text_input("",'')

st.markdown("""
    #
    #####  Post Datetime
""")
columns = st.columns(2)

d = columns[0].date_input("Date", datetime.date(2022, 12, 2))
columns[0].write(d)

t = columns[1].time_input("Time", datetime.time(8, 00))
columns[1].write(t)


### Images
st.markdown("""

    ##### Image
""")

time_string = str(d)+ " " +str(t)

time_stamp = int(time.mktime(time.strptime(time_string, '%Y-%m-%d %H:%M:%S'))) - time.timezone
# st.write(time_stamp)
def load_image(image_file):
	img = Image.open(image_file)
	return img

def find_square(tuple_):
    width, height = tuple_
    if width>height:
        lower = height
        upper = 0
        delta = int((width-height)/2)
        left = delta
        right = width-delta
    elif width<height:
        left = 0
        right = width
        delta = int((height-width)/2)
        upper = delta
        lower = height-delta
    else:
        left = 0
        right = width
        upper = 0
        lower = height
    return (left, upper, right, lower)

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose a PNG or JPEG file", type = ['png','jpg','jpeg'], accept_multiple_files =False)
show_file = st.empty
if not uploaded_file:
    st.write('Please upload a file:'.format(''.join(['png','jpg','jpeg'])))
else:

    image = Image.open(uploaded_file)
    w, h  = image.size
    im_size = w*h
    box = find_square(image.size)
    crop_image = image.crop(box)
    img = crop_image.resize((128, 128))
    im_file = BytesIO()
    img.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)

    payload ={"title": title,
              "time_stamp": time_stamp,
              "image_size": im_size,
              "filedata": im_b64}


def show(image, title, d, t, r):
    st.markdown( f"##### Title:  {title}")
    st.write('##### Post time:', d, t)
    st.image(image, caption='Your dog post')
    cat = r["category"]
    st.markdown("""
    ## Results""")
    if cat==5:
        st.success('Congrats you will likely get more than 500 upvotes!!! 🥳')
    elif cat == 4:
        st.info('You will get between 100 and 500 upvotes 🎉')
    elif cat == 3:
        st.warning('You will get between 30 and 100 upvotes')
    elif cat == 2:
        st.warning('Hold on, you will get less than 30 upvotes 🤔')
    elif cat == 1:
        st.warning('Hold on, you will get less than 30 upvotes 🤔')
    else:
        st.error('Damn this post is kinda bad, no one would give you upvotes 😭')
    # st.write(r["probabilities"])




if st.button('predict score'):
    st.write('Connecting...')

    try:
        r= requests.post(f"http://127.0.0.1:8000/getPrediction", data = payload).json()

        show(image, title, d, t, r)
        sizes = []
        for i in range(6):
            sizes.append(r["probabilities"][f'{i}'])

        fig,axes =plt.subplots()
        bar_label = ['0-1 upvotes', '2-15 upvotes', '15-30 upvotes', '30-100 upvotes', '100-500 upvotes','500+ upvotes']
        axes = sns.barplot(x=bar_label,y=sizes)
        axes.set_xticks(axes.get_xticks(), axes.get_xticklabels(), rotation=45, ha='right')

        fig.tight_layout()
        st.markdown('###### 👇 Probabilities for getting each level of upvotes:')
        st.pyplot(fig)

    except JSONDecodeError:
        st.markdown('## ❌ Error, title only containing Unknown or STOPWORDS(e.g. I, he, she)')
    except NameError:
        st.markdown('## ❌ Error, no Image uploaded')
    except KeyError:
        st.markdown('## ❌ Error, no Title inputed')





else:
    st.write('Click the button once all the data has been inputed')
