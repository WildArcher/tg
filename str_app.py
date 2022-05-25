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

with open(DATA_PATH + 'stops_russian.txt', newline='\n', encoding='utf-8') as w:
    words = w.readlines()
    words = [line.rstrip() for line in words] 
    
    
stop_words = list(set([word.lower() for word in words]))

with open(DATA_PATH + 'stops_nltk.txt', newline='\n', encoding='utf-8') as w:
    words = w.readlines()
    words = [line.rstrip().replace("'", "").replace(",", "") for line in words] 
    
    
stop_nltk = list(set([word.lower() for word in words]))

stop_words = stop_words + stop_nltk

pearsons_all = []
for i in range(data.shape[0]):
  new_pearsons = data['pearsons'].values[i]
  if new_pearsons != []:
    pearsons_all = pearsons_all + new_pearsons


pearsons_vol = []
for i in range(data.shape[0]):
  new_pearsons = data['pearsons'].values[i]
  if (new_pearsons != []) and (data['make_vol'].values[i] == 1):
    pearsons_vol = pearsons_vol + new_pearsons



per_all = Counter(pearsons_all)
per_vol = Counter(pearsons_vol)

per_all = pd.DataFrame(np.vstack((list(per_all.keys()), list(per_all.values()))).T, columns=['words', 'count_all'])
per_vol = pd.DataFrame(np.vstack((list(per_vol.keys()), list(per_vol.values()))).T, columns=['words', 'count_vol'])
per_all = per_all.merge(per_vol, on=['words'], how='outer').fillna(0)

per_all['count_all'] = per_all['count_all'].astype(int)
per_all['count_vol'] = per_all['count_vol'].astype(int)

per_all['percentage_vol'] = per_all['count_vol'] / per_all['count_all']
per_all['score'] = per_all['percentage_vol']*  (per_all['count_all']-1)**0.3


locations_all = []
for i in range(data.shape[0]):
  new_locations = data['locations'].values[i]
  if new_pearsons != []:
    locations_all = locations_all + new_locations


locations_vol = []
for i in range(data.shape[0]):
  new_locations = data['locations'].values[i]
  if (new_locations != []) and (data['make_vol'].values[i] == 1):
    locations_vol = locations_vol + new_locations



loc_all = Counter(locations_all)
loc_vol = Counter(locations_vol)

loc_all = pd.DataFrame(np.vstack((list(loc_all.keys()), list(loc_all.values()))).T, columns=['words', 'count_all'])
loc_vol = pd.DataFrame(np.vstack((list(loc_vol.keys()), list(loc_vol.values()))).T, columns=['words', 'count_vol'])
loc_all = loc_all.merge(loc_vol, on=['words'], how='outer').fillna(0)

loc_all['count_all'] = loc_all['count_all'].astype(int)
loc_all['count_vol'] = loc_all['count_vol'].astype(int)

loc_all['percentage_vol'] = loc_all['count_vol'] / loc_all['count_all']
loc_all['score'] = loc_all['percentage_vol']*  (loc_all['count_all']-2)**0.3




organisations_all = []
for i in range(data.shape[0]):
  new_organisations = data['organisations'].values[i]
  if new_organisations != []:
    organisations_all = organisations_all + new_organisations


organisations_vol = []
for i in range(data.shape[0]):
  new_organisations = data['organisations'].values[i]
  if (new_organisations != []) and (data['make_vol'].values[i] == 1):
    organisations_vol = organisations_vol + new_organisations



org_all = Counter(organisations_all)
org_vol = Counter(organisations_vol)

org_all = pd.DataFrame(np.vstack((list(org_all.keys()), list(org_all.values()))).T, columns=['words', 'count_all'])
org_vol = pd.DataFrame(np.vstack((list(org_vol.keys()), list(org_vol.values()))).T, columns=['words', 'count_vol'])
org_all = org_all.merge(org_vol, on=['words'], how='outer').fillna(0)

org_all['count_all'] = org_all['count_all'].astype(int)
org_all['count_vol'] = org_all['count_vol'].astype(int)

org_all['percentage_vol'] = org_all['count_vol'] / org_all['count_all']
org_all['score'] = org_all['percentage_vol']*  (org_all['count_all']-1)**0.3

### LDA

texts_to_lda = list(data['clean_text'].apply(lambda x: list(x.split(' '))))

common_dictionary = Dictionary(texts_to_lda)
common_corpus = [common_dictionary.doc2bow(text) for text in texts_to_lda]

num_topics=50
lda = LdaModel(common_corpus, num_topics=num_topics)

### LDA

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
