import numpy as np
import wandb
import random
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from torch.utils.data.sampler import SubsetRandomSampler


# HF
from transformers import AdamW

# local
try:
    from utils import random_string, load_text
except ImportError:
    from utils import random_string, load_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = load_text()


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
    model.eval()
    with torch.no_grad():
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        for data_, target_ in validation_loader:
            data_, target_ = data_.to(device), target_.to(device)
            outputs = model(data_)

            loss = criterion(outputs, target_).item()
            wandb.log({"test_final_loss": loss})

            _, pred = torch.max(outputs, dim=1)

            acc_metric.add_batch(predictions=pred, references=target_)
            f1_metric.add_batch(predictions=pred, references=target_)

    f1 = f1_metric.compute(average="weighted")['f1']
    wandb.log({"test_final_acc": acc_metric.compute()['accuracy']})
    wandb.log({"test_final_f1": f1})
    return f1


def train(model, train_loader, config, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    wandb.log({"lr": config['lr']})
    model.train()
    EPOCHS = config['epochs']

    for epoch in range(EPOCHS):

        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        for data_, target_ in train_loader:
            data_ = data_.to(device)
            target_ = target_.to(device)

            # Cleaning optimizer cache.
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(data_)

            # Computing loss & backward propagation
            loss = criterion(outputs, target_)
            loss.backward()

            # Predict stuff
            _, pred = torch.max(outputs, dim=1)
            acc_metric.add_batch(predictions=pred, references=target_)
            f1_metric.add_batch(predictions=pred, references=target_)
            # Log metrics
            wandb.log({"train_batch_loss": loss})
            wandb.log({"train_batch_acc": acc_metric.compute()['accuracy']})
            wandb.log({"train_batch_f1": f1_metric.compute(average="weighted")['f1']})

            # Applying gradients
            optimizer.step()
            scheduler.step()


def run_tf_idf(config, job_type=None):
    if job_type is None:
        job_type = f"bohb_{str(round(config['lr'], 2))}_{config['batch']}_{random_string(5)}"

    wandb.init(
        # set the wandb project where this run will be logged
        project="Baselines for Feature Extraction1",
        group="TF-IDF",
        job_type=job_type,
        config={
            "model": 'tf-idf classifier',
            "dataset": "amazon-multi",
            "device": device,
        }
    )
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

    model = DenseNetwork(n_features=512, n_classes=6).to(device)

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
    train(model, train_loader, config,optimizer=optimizer, scheduler=scheduler)
    score = test(model, validation_loader, test_sampler=test_sampler)
    wandb.finish()
    # return 1-
    return 1 - score


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
