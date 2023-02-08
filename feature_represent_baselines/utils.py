import random
import string
import codecs
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
import re

# document this function
def random_string(stringLength=5):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def load_embeddings():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('wiki.de.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index


## TF_IDF Utils
class Vectorizer():
    def __init__(self, clean_pattern=None, max_features=None, stop_words='english'):
        self.clean_pattern = clean_pattern
        self.max_features = max_features
        self.stopwords = stopwords.words('german')
        self.stopwords.extend(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
        self.tfidf = TfidfVectorizer(stop_words=self.stopwords, max_features=self.max_features)
        self.builded = False

    def _clean_texts(self, texts):

        cleaned = []
        for text in texts:
            if self.clean_pattern is not None:
                text = re.sub(self.clean_pattern, " ", text)

            text = text.lower().strip()
            cleaned.append(text)

        return cleaned

    def _set_tfidf(self, cleaned_texts):
        self.tfidf.fit(cleaned_texts)

    def build_vectorizer(self, texts):
        cleaned_texts = self._clean_texts(texts)
        self._set_tfidf(cleaned_texts)
        self.builded = True

    def vectorizeTexts(self, texts):
        if self.builded:
            cleaned_texts = self._clean_texts(texts)
            return self.tfidf.transform(cleaned_texts)

        else:
            raise Exception("Vectorizer is not builded.")
class TFIDF_Dataset(Dataset):

    def __init__(self, x_vectorized, y_encoded):
        self.x_vectorized = x_vectorized
        self.y_encoded = y_encoded

    def __len__(self):
        return len(self.x_vectorized)

    def __getitem__(self, index):
        return self.x_vectorized[index], self.y_encoded[index]


def load_hf_data():
    # load
    dataset_train = load_dataset("amazon_reviews_multi", 'de', split='train')
    data = pd.DataFrame(dataset_train)
    data['review_text'] = data['review_title'] + data['review_body']
    data = data.drop(['review_id', 'product_id', 'reviewer_id', 'language', 'review_body', 'review_title','product_category'],
                     axis=1)
    data.dropna(inplace=True)

    return data#.head(2000)
def get_vectorised_data(max_features=512):
    data = load_hf_data()
    x = list(data['review_text'])
    y = list(data["stars"])
    vectorizer = Vectorizer("[^a-zA-Z0-9]", max_features=max_features, stop_words="german")
    vectorizer.build_vectorizer(x)
    vector_x = vectorizer.vectorizeTexts(x).toarray()
    y_en = np.asarray(y)
    return vector_x, y_en
def load_text():
    vectorized_x, y_encoded = get_vectorised_data(max_features=512)
    return TFIDF_Dataset(vectorized_x, y_encoded)
