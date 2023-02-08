import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import wandb
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.metrics import classification_report
# HF
from datasets import load_dataset
import evaluate
# German Lemmatizer
from HanTa import HanoverTagger as ht
from functools import partial
from collections import Counter
from transformers import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# local
try:
    from utils import random_string
except ImportError:
    from utils import random_string
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def de_lemma_noun(text):
    # see documentation of the tagger. its working on words assuming sentences
    lemma = [lemma for (word, lemma, pos) in tagger.tag_sent(text)]
    return lemma


def de_lemma_verb(text):
    lemma = [lemma for (word, lemma, pos) in tagger.tag_sent(text)]
    return lemma


def build_bow_vector(sequence, idx2token):
    vector = [0] * len(idx2token)
    for token_idx in sequence:
        if token_idx not in idx2token:
            raise ValueError('Wrong sequence index found!')
        else:
            vector[token_idx] += 1
    return vector


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
        df = df.drop(['review_id', 'product_id', 'reviewer_id', 'stars', 'language', 'review_body', 'review_title'],
                     axis=1)

        german_stops = set(stopwords.words('german'))
        german_stops.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
        print(f'Start Lremmatization')
        df['tokens'] = df.review_text.apply(
            partial(
                tokenize,
                stop_words=german_stops,
            ),
        )
        df['tokens'] = df.tokens.apply(de_lemma_noun)

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
        print(f'End Lremmatization')
        # Build vocab
        vocab = sorted(set(
            token for doc in list(df.tokens) for token in doc
        ))
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # Convert tokens to indexes
        df['indexed_tokens'] = df.tokens.apply(
            lambda doc: [self.token2idx[token] for token in doc],
        )

        # Build BoW vector
        df['bow_vector'] = df.indexed_tokens.apply(
            build_bow_vector, args=(self.idx2token,)
        )
        print('Data loaded!')

        self.text = df.review_text.tolist()
        self.sequences = df.indexed_tokens.tolist()
        self.bow_vector = df.bow_vector.tolist()
        self.targets = df.target.tolist()

    def __getitem__(self, i):
        return (
            self.sequences[i],
            self.bow_vector[i],
            self.targets[i],
            self.text[i],
        )

    def __len__(self):
        return len(self.targets)

    def get_feat_class(self):
        return len(self.token2idx), self.n_class


def split_train_valid_test(corpus, valid_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test."""
    test_length = int(len(corpus) * test_ratio)
    valid_length = int(len(corpus) * valid_ratio)
    train_length = len(corpus) - valid_length - test_length
    return random_split(
        corpus, lengths=[train_length, valid_length, test_length],
    )


def collate(batch):
    seq = [item[0] for item in batch]
    bow = [item[1] for item in batch]
    tfidf = [item[2] for item in batch]
    empty_arr = []
    for item in batch:
        empty_arr.append(item[2])
    target = torch.LongTensor(empty_arr)
    target = torch.LongTensor([item[2] for item in batch])
    text = [item[3] for item in batch]
    return seq, bow, target, text


class DenseNetwork(nn.Module):

    def __init__(self, n_features, n_classes):
        super(DenseNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.4)
        self.prediction = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x.to(torch.float)))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.log_softmax(self.prediction(x), dim=1)
        return x


def train_epoch(model, optimizer, train_loader, criterion, scheduler):
    model.train()
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    for seq, bow, target, text in tqdm(train_loader):
        inputs = torch.FloatTensor(bow).to(device)
        # Reset gradient
        optimizer.zero_grad()
        # Forward pass
        output = model(inputs)
        # Compute loss
        loss = criterion(output, target)
        # Perform gradient descent, backwards pass
        loss.backward()

        # Predict stuff
        _, pred = torch.max(output, dim=1)
        acc_metric.add_batch(predictions=pred, references=target)
        f1_metric.add_batch(predictions=pred, references=target)
        # Log metrics
        wandb.log({"train_batch_loss": loss.item()})
        wandb.log({"train_batch_acc": acc_metric.compute()['accuracy']})
        wandb.log({"train_batch_f1": f1_metric.compute(average="weighted")['f1']})

        # Take a step in the right direction
        optimizer.step()
        scheduler.step()


def validate_epoch(model, valid_loader, criterion, input_type='bow'):
    model.eval()
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    with torch.no_grad():
        for seq, bow, target, text in tqdm(valid_loader):
            inputs = torch.FloatTensor(bow).to(device)
            # Forward pass
            output = model(inputs)
            loss = criterion(output, target)
            # Predict stuff
            _, pred = torch.max(output, dim=1)
            acc_metric.add_batch(predictions=pred, references=target)
            f1_metric.add_batch(predictions=pred, references=target)
            # Log metrics
            wandb.log({"valid_batch_acc": acc_metric.compute()['accuracy']})
            wandb.log({"valid_batch_f1": f1_metric.compute(average="weighted")['f1']})
            wandb.log({"valid_batch_loss": loss.item()})


def run_bow(config, job_type=None):
    if job_type is None:
        job_type = f"bohb_{str(round(config['lr'], 2))}_{config['batch']}_{random_string(5)}"

    MAX_VOCAB = config['vocab']  # 10000
    LEARNING_RATE = config['lr']  # 5e-4
    weight_decay = config['weight_decay']
    opt_type = config['optimizer']
    n_epochs = config['epochs']  # 10
    BATCH_SIZE = config['batch']

    dataset = DeDataset(max_vocab=MAX_VOCAB, max_len=512)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Debug",#"Baselines for Feature Extraction1",
        group="BOW",
        job_type=job_type,
        config={
            "model": 'bow embed classifier',
            "dataset": "amazon-multi",
            "device": device,
        },
    )

    wandb.log({"vocab": MAX_VOCAB})
    wandb.log({"batch": BATCH_SIZE})
    wandb.log({"lr": LEARNING_RATE})
    wandb.log({"weight_decay": weight_decay})
    wandb.log({"optimizer_type": opt_type})
    wandb.log({"epochs": n_epochs})

    train_dataset, valid_dataset, test_dataset = split_train_valid_test(
        dataset, valid_ratio=0.05, test_ratio=0.05)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

    n_features, n_classes = dataset.get_feat_class()
    model = DenseNetwork(n_features=n_features, n_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    if opt_type == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=LEARNING_RATE, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=LEARNING_RATE, weight_decay=weight_decay)
    elif opt_type == 'adamW':
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=LEARNING_RATE, weight_decay=weight_decay)
    else:
        print(f"Unsuported optimizer {opt_type}")

    # TODO: add variation of scheduler to see how it impacts performance
    scheduler = CosineAnnealingLR(optimizer, 1)

    for epoch in range(n_epochs):
        train_epoch(model, optimizer, train_loader, criterion=criterion, scheduler=scheduler)
        validate_epoch(model, valid_loader, criterion=criterion)

    model.eval()
    with torch.no_grad():
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        for seq, bow, target, text in tqdm(test_loader):
            inputs = torch.FloatTensor(bow).to(device)
            probs = model(inputs)

            probs = probs.detach().cpu().numpy()
            predictions = np.argmax(probs, axis=1)
            target = target.cpu().numpy()

            acc_metric.add_batch(predictions=predictions, references=target)
            f1_metric.add_batch(predictions=predictions, references=target)

        f1 = f1_metric.compute(average="weighted")['f1']
        wandb.log({"test_final_acc": acc_metric.compute()['accuracy']})
        wandb.log({"test_final_f1": f1})

    wandb.finish()
    return 1 - f1


if __name__ == '__main__':
    run_bow(config={
        'batch': 32,
        'epochs': 6,
        'lr': 0.004004797203254154,
        'optimizer': 'adam',
        'vocab': 4096,
        'weight_decay': 0.00014330152463163526, },
        job_type='Best bow')
