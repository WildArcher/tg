import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities
from collections import Counter
counter_ = Counter()

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# from flair.data import Sentence
# from flair.models import SequenceTagger
from googletrans import Translator

# import torch
# from transformers import AutoModelForSequenceClassification
# from transformers import BertTokenizerFast

def text_preprocessing(text, lemmatize, replacement, del_stop_words, no_connection, del_word_less_2_symbol, stop_words):
  
    for i in range(len(stop_words)):
        stop_words[i] = ' '+stop_words[i]+' '

    for i in range(len(stop_words)):
        stop_words[i] = ' '+stop_words[i]+' '

    d = text
    d = d.lower() # приведение к нижнему регистру
    d = re.sub(r'https://[a-z1-9./]+', ' ', d) # удаление ссылок

    d = re.sub('[^a-zа-я]', ' ', d) # удаление символов кроме символов русского и английского алфавитов

    if lemmatize:
         d = " ".join(morph.parse(word)[0].normal_form for word in d.split()) # Лемматизация

    d = re.sub(' нет ', ' не ', d) # Замена 'нет' и 'ни' на 'не'
    d = re.sub(' ни ', ' не ', d)
    d = re.sub(r'\d+', 'number_token', d) # Замена чисел на number_token

    if replacement:
        d = re.sub('ё', 'е', d)
        d = re.sub('й', 'и', d)

    if del_stop_words:
         for stop_word in stop_words:
            if stop_word in d:
                d = re.sub(stop_word, ' ', d) # удаление стоп слов      

    if no_connection:
        words = d.split()
        for i in range(len(words)-1):
            if words[i] == ' не ':
                words[i] = words[i] + words[i+1]
                words[i+1] = ' '
            else:
                d = ' '.join(words)
                d = ' '.join(d.split())

    if del_word_less_2_symbol == True:
        d = " ".join([w for w in d.split() if len(w) > 2])

    return d

def lemmatize_ner(word):
  return morph.parse(word)[0].normal_form

def make_lists_of_ner(text):

  text_en = translator.translate(text, src="ru", dest="en").text
  text_en = Sentence(text_en)

  tagger.predict(text_en)

  list_of_organisation = []
  list_of_pearsons = []
  list_of_locations = []

  for entity in text_en.get_spans('ner'):
    if entity.tag == "PER":
      list_of_pearsons.append(lemmatize_ner(translator.translate(entity.text, src="en", dest="ru").text))
    if entity.tag == "ORG":
      list_of_organisation.append(lemmatize_ner(translator.translate(entity.text, src="en", dest="ru").text))
    if entity.tag == "LOC":
      list_of_locations.append(lemmatize_ner(translator.translate(entity.text, src="en", dest="ru").text))

  return list_of_organisation, list_of_pearsons, list_of_locations 

def make_ner_datasets(data, entity):
  pearsons_all = []
  for i in range(data.shape[0]):
    new_pearsons = data[entity].values[i]
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

  return per_all

def get_lda_themes(text, common_dictionary, lda):
  new_texts_to_lda = list(text.split(' '))

  new_texts_to_lda = [common_dictionary.doc2bow(text) for text in [new_texts_to_lda]]
  themes = []
  for row in np.array(lda[new_texts_to_lda])[0]:
    if row[1] > 0.1:
      themes.append(row[0])
     #print([common_dictionary[int(lda.show_topic(int(row[0]))[i][0])] for i in range(len(lda.show_topic(0)))])

  return themes
