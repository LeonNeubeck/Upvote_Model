FROM python:3.8.12-buster

COPY app /app
COPY requirements.txt /requirements.txt
COPY Makefile Makefile
COPY main.py /main.py
COPY models /models
COPY __pycache__ /__pycache__
COPY scripts /scripts
COPY Upvote_Model /Upvote_Model
COPY build  /build
COPY MANIFEST.in MANIFEST.in
COPY preproc.py preproc.py
COPY setup.py setup.py
COPY Upvote_Model.egg-info /Upvote_Model.egg-info
COPY balanced_35k.csv balanced_35k.csv


RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install .

CMD make train
