FROM python:3.10.6-buster

RUN mkdir Upvote_Model
WORKDIR Upvote_Model

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

COPY app app
COPY Makefile Makefile
COPY main.py main.py
COPY models models
COPY scripts scripts
COPY Upvote_Model Upvote_Model
COPY preproc.py preproc.py
COPY setup.py setup.py
COPY raw_data raw_data

RUN pip install .
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

CMD uvicorn main:app --reload --host 0.0.0.0 --port 8000
