import numpy as np
import pandas as pd
import re
import wandb
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ConfigSpace import ConfigurationSpace, Configuration
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

# HF
from datasets import load_dataset
from transformers import AdamW

# local
try:
    from utils import random_string
except ImportError:
    from utils import random_string
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeDataset(Dataset):

    def __init__(self, x_vectorized, y_encoded):
        self.x_vectorized = x_vectorized
        self.y_encoded = y_encoded

    def __len__(self):
        return len(self.x_vectorized)

    def __getitem__(self, index):
        return self.x_vectorized[index], self.y_encoded[index]


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


def load_hf_data():
    # load
    dataset_train = load_dataset("amazon_reviews_multi", 'de', split='train')
    data = pd.DataFrame(dataset_train)
    data['review_text'] = data['review_title'] + data['review_body']
    data = data.drop(['review_id', 'product_id', 'reviewer_id', 'stars', 'language', 'review_body', 'review_title'],
                     axis=1)
    data.dropna(inplace=True)
    data.info()

    return data.head(2000)


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


def test(model, validation_loader, test_sampler, criterion=nn.CrossEntropyLoss()):
    test_true = 0
    test_total = len(test_sampler)
    test_loss = 0.0
    with torch.no_grad():
        for data_, target_ in validation_loader:
            data_, target_ = data_.to(device), target_.to(device)

            outputs = model(data_)

            loss = criterion(outputs, target_).item()
            wandb.log({"test_loss": loss})

            _, pred = torch.max(outputs, dim=1)

            test_true += torch.sum(pred == target_).item()
            test_loss += loss

    accuracy = round(test_true / test_total, 4)
    print(f"Validation finished: Accuracy = {round(100 * test_true / test_total, 2)}%, Loss = {test_loss}")
    wandb.log({"test_accuracy": round(100 * test_true / test_total, 2)})
    return 1 - accuracy


def train(model, train_loader, config, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    wandb.log({"lr": config['lr']})

    EPOCHS = config['epochs']
    TRAIN_LOSSES = []
    TRAIN_ACCURACIES = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_true = 0
        epoch_total = 0
        for data_, target_ in train_loader:
            data_ = data_.to(device)
            target_ = target_.to(device)

            # Cleaning optimizer cache.
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(data_)

            # Computing loss & backward propagation
            loss = criterion(outputs, target_)
            wandb.log({"train_loss": loss})
            loss.backward()

            # Applying gradients
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            _, pred = torch.max(outputs, dim=1)
            epoch_true = epoch_true + torch.sum(pred == target_).item()

            epoch_total += target_.size(0)
        TRAIN_LOSSES.append(epoch_loss)
        wandb.log({"train_epoch_loss": epoch_loss})
        wandb.log({"train_epoch_accuracy": 100 * epoch_true / epoch_total})
        TRAIN_ACCURACIES.append(100 * epoch_true / epoch_total)
    accuracy = TRAIN_ACCURACIES[epoch - 1]
    print(
        f"Epoch {epoch + 1}/{EPOCHS} finished: train_loss = {epoch_loss}, train_accuracy = {TRAIN_ACCURACIES[epoch - 1]}")
    return 1 - accuracy


def get_vectorised_data(max_features=512):
    data = load_hf_data()
    label_map = {value: count for count, value in enumerate(set(data['product_category']))}
    x = list(data['review_text'])
    y = list(data["product_category"])
    vectorizer = Vectorizer("[^a-zA-Z0-9]", max_features=max_features, stop_words="german")
    vectorizer.build_vectorizer(x)
    vector_x = vectorizer.vectorizeTexts(x).toarray()
    n_feat = vector_x.shape[1]
    n_cls = len(label_map)
    y_en = []
    for y_sample in y:
        y_en.append(label_map[y_sample])

    y_en = np.asarray(y_en)
    return vector_x, y_en, n_feat, n_cls, label_map


def run_tf_idf(config, job_type=None):
    if job_type is None:
        job_type = f"bohb_{str(round(config['lr'], 2))}_{config['batch']}_{random_string(5)}"

    wandb.init(
        # set the wandb project where this run will be logged
        project="Baselines for Feature Extraction",
        group="TF-IDF",
        job_type=job_type,
        config={
            "model": 'tf-idf classifier',
            "dataset": "amazon-multi",
            "device": device,
        }
    )

    vectorized_x, y_encoded, n_features, n_classes, label_map = get_vectorised_data(max_features=512)
    wandb.log({"n_features": n_features})

    dataset = DeDataset(vectorized_x, y_encoded)
    train_indices, test_indices = train_test_split(list(range(0, len(dataset))), test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    LEARNING_RATE = config['lr']  # 5e-4
    weight_decay = config['weight_decay']
    opt_type = config['optimizer']
    n_epochs = config['epochs']  # 10
    BATCH_SIZE = config['batch']

    wandb.log({"batch": BATCH_SIZE})
    wandb.log({"lr": LEARNING_RATE})
    wandb.log({"weight_decay": weight_decay})
    wandb.log({"optimizer_type": opt_type})
    wandb.log({"epochs": n_epochs})

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=test_sampler)

    model = DenseNetwork(n_features=n_features, n_classes=n_classes).to(device)

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
    # TODO: vary the scheduler for BOHB
    scheduler = CosineAnnealingLR(optimizer, 1)
    score = train(model, train_loader, config, optimizer=optimizer, scheduler=scheduler)
    test(model, validation_loader, test_sampler=test_sampler)
    wandb.finish()
    return score


def write_to_wand():
    {
        'batch': 128,
        'epochs': 10,
        'lr': 0.006876174282463883,
    }


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    run_tf_idf(config={
        'batch': 64,
        'epochs': 10,
        'lr': 0.009672407004688972,
        'optimizer': 'adamW',
        'weight_decay': 0.003052454777425161, },
        job_type="best_tf-idf")
