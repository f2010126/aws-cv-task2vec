import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import partial
from collections import Counter
import re
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import codecs
import fasttext.util
from HanTa import HanoverTagger as ht

tagger = ht.HanoverTagger('morphmodel_ger.pgz')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def de_lemma_noun(text):
    # see dlcumentation of teh tagger. its working on words assuming sentences
    lemma = [lemma for (word, lemma, pos) in tagger.tag_sent(text)]
    return lemma


def de_lemma_verb(text):
    lemma = [lemma for (word, lemma, pos) in tagger.tag_sent(text)]
    return lemma


def tokenize(text, stop_words):
    text = re.sub(r'[^\w\s]', '', text)  # remove special characters
    text = text.lower()  # lowercase
    # what does tokenise do?
    tokens = word_tokenize(text, language='german')  # tokenize
    tokens = de_lemma_noun(tokens)  # noun lemmatizer
    tokens = de_lemma_verb(tokens)  # verb lemmatizer
    tokens = [token for token in tokens if token not in stop_words]  # remove stopwords
    return tokens


def remove_rare_words(tokens, common_tokens, max_len):
    return [token if token in common_tokens else '<UNK>' for token in tokens][-max_len:]


def replace_numbers(tokens):
    return [re.sub(r'[0-9]+', '<NUM>', token) for token in tokens]


def remove_unknown(tokens):
    return [token for token in tokens if token != '<UNK>']


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


class DeDataset(Dataset):
    def __init__(self, max_vocab=5000, max_len=128):
        dataset_train = load_dataset("amazon_reviews_multi", 'de', split='train')
        df = pd.DataFrame(dataset_train)
        df = df.head(2000)
        df['product_category'] = [str(x).lower() for x in df['product_category']]
        self.n_class = len(df['product_category'].unique())
        df.product_category = pd.Categorical(df.product_category)
        df['target'] = df.product_category.cat.codes
        df['review_text'] = df['review_title'] + df['review_body']
        df = df.drop(['review_id', 'product_id', 'reviewer_id', 'stars', 'language', 'review_body', 'review_title',
                      'product_category'],
                     axis=1)

        german_stops = set(stopwords.words('german'))
        german_stops.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
        df['tokens'] = df.review_text.apply(
            partial(
                tokenize,
                stop_words=german_stops,
            ),
        )
        df['tokens'] = df.tokens.apply(de_lemma_noun)

        # FastText changes here
        all_tokens = [token for doc in list(df.tokens) for token in doc]
        # Build most common tokens bound by max vocab size
        common_tokens = set(
            list(
                zip(*Counter(all_tokens).most_common(max_vocab))
            )[0]
        )

        df.loc[:, 'tokens'] = df.tokens.apply(
            partial(
                remove_rare_words,
                common_tokens=common_tokens,
                max_len=max_len,
            ),
        )

        # Replace numbers with <NUM>
        df.loc[:, 'tokens'] = df.tokens.apply(replace_numbers)

        # Remove sequences with only <UNK>
        df = df[df.tokens.apply(
            lambda tokens: any(token != '<UNK>' for token in tokens),
        )]
        # Remove the <UNK>
        df.loc[:, 'tokens'] = df.tokens.apply(remove_unknown)

        # Build vocab
        self.vocab = sorted(set(
            token for doc in list(df.tokens) for token in doc
        ))

        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # Convert tokens to indexes
        df['indexed_tokens'] = df.tokens.apply(
            lambda doc: [self.token2idx[token] for token in doc],
        )

        self.text = df.review_text.tolist()
        self.sequences = df.indexed_tokens.tolist()
        self.targets = df.target.tolist()

    def __getitem__(self, i):
        return (
            self.text[i],
            self.sequences[i],
            self.targets[i],
        )

    def __len__(self):
        return len(self.targets)

    def get_feat_class(self):
        return len(self.token2idx), self.n_class

    def get_vocab(self):
        return self.vocab


def collate(batch):
    # text seq target
    text = [item[0] for item in batch]
    seq = [item[1] for item in batch]
    target = torch.LongTensor([item[2] for item in batch])

    return seq, target, text


def split_train_valid_test(corpus, valid_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(
        corpus, lengths=[train_length, valid_length, test_length],
    )


# Main Runner
def fasttext(config=None):
    MAX_LEN = 512
    MAX_VOCAB = 10000
    dataset = DeDataset(max_vocab=MAX_VOCAB, max_len=MAX_LEN)
    train_dataset, valid_dataset, test_dataset = split_train_valid_test(
        dataset, valid_ratio=0.05, test_ratio=0.05)

    # training params
    batch_size = 256
    num_epochs = 8
    BATCH_SIZE = 256  # config['batch']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

    # model parameters
    embed_dim = 300  # default size of embed from fast text
    weight_decay = 1e-4

    # get word embeddings
    embeddings = load_embeddings()

    print('preparing embedding matrix...')
    words_not_found = []
    # Embedding matrix is of size vocab(from the train set) length
    vocab = dataset.get_vocab()
    nb_words = min(MAX_VOCAB, len(vocab))
    embedding_matrix = np.zeros((len(vocab), embed_dim))
    for i, word in enumerate(vocab):
        if i >= nb_words:
            continue
        embedding_vector = embeddings.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    fasttext()
