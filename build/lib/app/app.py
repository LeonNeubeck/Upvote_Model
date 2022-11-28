import streamlit as st
import datetime
import pandas as pd
import requests



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
t = st.date_input(
    "Date",
    datetime.date(2022, 11, 10),
    )
d = st.time_input(
    "Time",
    datetime.time(8, 00)
)
st.write('DateTime:', d, t)

### Images
st.markdown("""

    ##### Image
""")


st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose a PNG or JPEG file", type = ['png','jpg','jpeg'])
show_file = st.empty
if not uploaded_file:
    show_file.info('Please upload a file:'.format(''.join(['png','jpg','jpeg'])))
else:
    show_file.image
    st.write(f"{uploaded_file}")

# with st.form(key='params_for_api'):

#     post_date = st.date_input('Date', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
#     post_time = st.time_input('Time', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
#     pickup_datetime = f'{post_date} {post_time}'
#     title = st.text_input('Post title')
#     # images ...
#     passenger_count = st.number_input('passenger_count', min_value=1, max_value=8, step=1, value=1)

#     st.form_submit_button('Make prediction')
