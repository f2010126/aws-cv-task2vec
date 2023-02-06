import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.metrics import classification_report
import fasttext.util
import itertools
from HanTa import HanoverTagger as ht
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

tagger = ht.HanoverTagger('morphmodel_ger.pgz')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# get word embeddings, embedding[word] returns the associated vector.
embeddings = load_embeddings()

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


def pad_to_longest(lists):
    pad_token=0
    return list(zip(*itertools.zip_longest(*lists, fillvalue=pad_token)))
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
        # needs to be padded
        self.sequences = pad_to_longest(df.indexed_tokens.tolist())
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

class DenseNetwork(nn.Module):

    def __init__(self, batch, embed_dim,vocab_len, n_classes):
        super(DenseNetwork, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim)
        self.flat=nn.Flatten()
        self.fc1 = nn.Linear(53700,512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.4)
        self.prediction = nn.Linear(256, n_classes)

    def set_embedding_weights(self, embed_w):
        self.embed.weight.data.copy_(embed_w)

    def forward(self, x):
        x= x.to(torch.long)
        x = self.embed(x)
        x= self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.log_softmax(self.prediction(x), dim=1)
        return x
# Main Runner
def fasttext(config=None):
    MAX_LEN = 512
    MAX_VOCAB = 10000
    dataset = DeDataset(max_vocab=MAX_VOCAB, max_len=MAX_LEN)
    train_dataset, valid_dataset, test_dataset = split_train_valid_test(
        dataset, valid_ratio=0.05, test_ratio=0.05)

    # training params
    batch_size = 256
    BATCH_SIZE = 256  # config['batch']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

    # model parameters
    embed_dim = 300  # default size of embed from fast text
    weight_decay = 1e-4

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

    model = DenseNetwork(batch=batch_size,embed_dim=embed_dim,vocab_len=len(vocab),n_classes=dataset.n_class)
    model.set_embedding_weights(torch.FloatTensor(embedding_matrix))
    model=model.to(device)

    learning_rate = 0.001#config['lr']
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    scheduler = CosineAnnealingLR(optimizer, 1)
    n_epochs = 10 #config['epochs']

    TRAIN_ACCURACIES = []
    train_losses, valid_losses = [], []
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader,criterion=criterion, scheduler=scheduler)
        valid_loss = validate_epoch(model, valid_loader, criterion=criterion)

        print(
            f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n \ttrain_acc: {train_acc:.2e}\n',
        )
        train_losses.append(train_loss)
        TRAIN_ACCURACIES.append(train_acc)
        valid_losses.append(valid_loss)

    model.eval()
    test_accuracy, n_examples = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for seq,target, text in test_loader:
            inputs = torch.FloatTensor(seq).to(device)
            probs = model(inputs)

            probs = probs.detach().cpu().numpy()
            predictions = np.argmax(probs, axis=1)
            target = target.cpu().numpy()

            y_true.extend(predictions)
            y_pred.extend(target)

    print(classification_report(y_true, y_pred))


def train_epoch(model, optimizer, train_loader, criterion,scheduler):
    model.train()
    total_loss, total,epoch_true = 0, 0, 0

    for seq, target, text in train_loader:
        inputs = torch.FloatTensor(seq).to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)

        # Compute loss
        loss = criterion(output, target)

        # Perform gradient descent, backwards pass
        loss.backward()

        # Take a step in the right direction
        optimizer.step()
        scheduler.step()

        # Record metrics
        total_loss += loss.item()
        total += len(target)
        _, pred = torch.max(output, dim=1)
        epoch_true = epoch_true + torch.sum(pred == target).item()

    return total_loss / total, epoch_true / total

def validate_epoch(model, valid_loader, criterion):
    model.eval()
    total_loss, total = 0, 0
    with torch.no_grad():
        for seq, target, text in valid_loader:
           inputs = torch.LongTensor(seq).to(device)
           #Forward pass
           output = model(inputs)

           # Calculate how wrong the model is
           loss = criterion(output, target)

            # Record metrics
           total_loss += loss.item()
        total += len(target)

    return total_loss / total


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    fasttext()
