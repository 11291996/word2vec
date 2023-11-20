import re
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#model loading
loaded_model = KeyedVectors.load_word2vec_format("english_model")

model_result = loaded_model.most_similar("man")
print(model_result)