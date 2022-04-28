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
from tg_parser_utils import text_preprocessing, get_lda_themes
morph = pymorphy2.MorphAnalyzer()

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities

DATA_PATH = 'data/'

data_new = pd.read_json(DATA_PATH + 'channel_messages_markettwits_new.json')
data = pd.read_csv(DATA_PATH + 'data (4).csv')

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
new_text_themes = get_lda_themes(new_clean_text, common_dictionary=common_dictionary, lda=lda_model)

st.write(new_text_themes)
