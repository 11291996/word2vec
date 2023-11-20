import re
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

#data processing
targetXML = open('./data/word2vec.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

#getting 
parse_text = '\n'.join(target_text.xpath('//content/text()'))

#getting rid of instructions
content_text = re.sub(r'\([^)]*\)', '', parse_text)

#sentence tokenizing using nltk
sent_text = sent_tokenize(content_text)

#elimnating punctuation and large letters
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

#tokenizing each sentence's word using nltk
result = [word_tokenize(sentence) for sentence in normalized_text]

print('sample numbers : {}'.format(len(result)))

#printing first 3 samples
for line in result[:3]:
    print(line)

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

model_result = model.wv.most_similar("man")
print(model_result)

#saving model
model.wv.save_word2vec_format('english_model')