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
# import streamlit 
# # from tg_parser_utils import text_preprocessing
# from pathlib import Path
# morph = pymorphy2.MorphAnalyzer()


start_date = streamlit.date_input(
     "start date")

DATA_PATH = './data'

data_new = pd.read_json(DATA_PATH / 'channel_messages_markettwits_new.json')

with open('stops_russian.txt', newline='\n', encoding='utf-8') as w:
    words = w.readlines()
    words = [line.rstrip() for line in words] 
    
    
stop_words = list(set([word.lower() for word in words]))

with open('stops_nltk.txt', newline='\n', encoding='utf-8') as w:
    words = w.readlines()
    words = [line.rstrip().replace("'", "").replace(",", "") for line in words] 
    
    
stop_nltk = list(set([word.lower() for word in words]))

stop_words = stop_words + stop_nltk


new_text = data_new['message'].values[0]
new_clean_text = text_preprocessing(text=new_text, lemmatize=True, 
                                              replacement=False, del_stop_words=True, no_connection=False, 
                                              del_word_less_2_symbol=True, stop_words=stop_words)

st.write(new_clean_text)
