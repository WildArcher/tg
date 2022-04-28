import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities

DATA_PATH = 'data/'
data = pd.read_csv(DATA_PATH + 'data (4).csv')

texts_to_lda = list(data['clean_text'].apply(lambda x: list(x.split(' '))))
common_dictionary = Dictionary(texts_to_lda)
lda_model =  models.LdaModel.load('lda.model')


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
  
def get_lda_themes(text, common_dictionary, lda):
  new_texts_to_lda = list(text.split(' '))

  new_texts_to_lda = [common_dictionary.doc2bow(text) for text in [new_texts_to_lda]]
  themes = []
  for row in np.array(lda[new_texts_to_lda])[0]:
    if row[1] > 0.1:
      themes.append(row[0])
     #print([common_dictionary[int(lda.show_topic(int(row[0]))[i][0])] for i in range(len(lda.show_topic(0)))])

  return themes
