from telethon import TelegramClient, events, sync
import pandas as pd
import numpy as np
from collections import Counter
import pymorphy2
import re
import warnings
import datetime
import configparser
import json
import asyncio
from datetime import date, datetime
import streamlit as st
from tg_parser_utils import text_preprocessing
morph = pymorphy2.MorphAnalyzer()

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities
from collections import Counter
counter_ = Counter()

from flair.data import Sentence
from flair.models import SequenceTagger
from googletrans import Translator

import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

from tg_parser_utils import lemmatize_ner, make_ner_datasets



@torch.no_grad()
def predict(text):
    inputs = tokenizer(text=text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_score = np.max(predicted.numpy())
    predicted_sentiment = int(torch.argmax(predicted).numpy())
    return predicted_score, predicted_sentiment
  

def get_lda_themes(text, common_dictionary, lda):
  new_texts_to_lda = list(text.split(' '))

  new_texts_to_lda = [common_dictionary.doc2bow(text) for text in [new_texts_to_lda]]
  themes = []
  for row in np.array(lda[new_texts_to_lda])[0]:
    if row[1] > 0.1:
      themes.append(row[0])
     #print([common_dictionary[int(lda.show_topic(int(row[0]))[i][0])] for i in range(len(lda.show_topic(0)))])

  return themes

DATA_PATH = 'data/'

data_new = pd.read_json(DATA_PATH + 'channel_messages_markettwits_new.json')
data = pd.read_csv(DATA_PATH + 'data (4).csv')

data['organisations'] = data['organisations'].fillna('[]').apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', '))
data['pearsons'] = data['pearsons'].fillna('[]').apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', '))
data['locations'] = data['locations'].fillna('[]').apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', '))

per_all = make_ner_datasets(data=data, entity='pearsons')

with open(DATA_PATH + 'stops_russian.txt', newline='\n', encoding='utf-8') as w:
    words = w.readlines()
    words = [line.rstrip() for line in words] 
    
    
stop_words = list(set([word.lower() for word in words]))

with open(DATA_PATH + 'stops_nltk.txt', newline='\n', encoding='utf-8') as w:
    words = w.readlines()
    words = [line.rstrip().replace("'", "").replace(",", "") for line in words] 
    
    
stop_nltk = list(set([word.lower() for word in words]))

stop_words = stop_words + stop_nltk


new_text = data_new['message'].values[0]
new_clean_text = text_preprocessing(text=new_text, lemmatize=True, 
                                              replacement=False, del_stop_words=True, no_connection=False, 
                                              del_word_less_2_symbol=True, stop_words=stop_words)

st.write("Clean text: ", new_clean_text)

texts_to_lda = list(data['clean_text'].apply(lambda x: list(x.split(' '))))
common_dictionary = Dictionary(texts_to_lda)
lda_model =  models.LdaModel.load('lda.model')

new_text_themes = get_lda_themes(new_clean_text, common_dictionary=common_dictionary, lda=lda_model)

st.write(new_text_themes)
