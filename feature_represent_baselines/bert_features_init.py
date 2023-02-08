from nlp.nlp_model import BERT, _bert_hook
from task2vec_datasets_nlp import amazon_reviews_multi
from task2vec_nlp import convert_features, _get_loader
from torch.utils.data import DataLoader, Dataset
from utils import AverageMeter, get_error, get_device
import torch
import torch.nn as nn
from transformers import AdamW
import math
import wandb
from tqdm import tqdm
import logging
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_bert_feat(config=None):
    train_data, val_data, label_map = amazon_reviews_multi()
    model = BERT(classes=len(label_map))  # T2VBertArch.from_pretrained("bert-base-german-cased")
    model.replace_head(num_labels=len(label_map))
    model.set_layers()
    batch_size = config['batch']

    data_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size,
                             num_workers=2, drop_last=False)

    hook = model.layers[-1].register_forward_pre_hook(_bert_hook)
    max_samples = 1000
    if max_samples is not None:
        n_batches = min(
            math.floor(max_samples / data_loader.batch_size) - 1, len(data_loader))
    else:
        n_batches = len(data_loader)
    targets = []
    loss_fn = nn.NLLLoss()
    optimizer = config['optimizer']  # 'adamW'
    learning_rate = config['lr']  # 5e-5
    weight_decay = config['weight_decay']  # 0.0001
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'adamW':
        optimizer = AdamW(model.classifier.parameters(), lr=5e-5, weight_decay=weight_decay)

    else:
        raise ValueError(f'Unsupported optimizer {optimizer}')

    # Let the classifier features actually learn.
    epoch_true = 0
    epoch_total = 0
    for step, batch in enumerate(data_loader):
        # progress update after every 10 batches.
        if step % 10 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(data_loader)))
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()
        targets.append(labels.clone())
        model.store_input_size(input_ids=sent_id)
        output = model(input_ids=sent_id, attention_mask=mask)
        loss = loss_fn(output, labels)
        error = get_error(output, labels)
        print(f'Accuracy = {100 - error}')
        _, pred = torch.max(output, dim=1)
        epoch_true = epoch_true + torch.sum(pred == labels).item()
        loss.backward()
        optimizer.step()
        epoch_total += labels.size(0)
        print(f"Batch Accuracy----> {100 * epoch_true / epoch_total}")

    hook.remove()
    # Convert the data arrays into a tensor
    convert_features(model.layers[-1])

    model.layers[-1].targets = torch.cat(targets)  # add a prop to the classifier

    if not hasattr(model.classifier, 'input_features'):
        raise ValueError("You need to run `cache_features` on model before running `fit_classifier`")
    targets = model.classifier.targets.to(device)
    # since its classifier, only one input
    features = model.classifier.input_features[0].to(device)
    torch.save(features, 'features.pt')
    torch.save(targets,'targets.pt')
    dataset = torch.utils.data.TensorDataset(features, targets)

    data_loader = _get_loader(dataset, batch_size=batch_size, num_workers=2, drop_last=True)
    optimizer =config['optimizer'] #'adamW'
    learning_rate = config['lr']#5e-5
    weight_decay = config['weight_decay'] #0.0001
    epochs = config['epochs']

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'adamW':
        optimizer = AdamW(model.classifier.parameters(), lr=5e-5, weight_decay=weight_decay)

    else:
        raise ValueError(f'Unsupported optimizer {optimizer}')

    loss_fn = nn.NLLLoss()

    for epoch in tqdm(range(epochs), desc="Fitting classifier", leave=False):
        metrics = AverageMeter()
        epoch_true = 0
        epoch_total = 0
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model.classifier(data)
            loss = loss_fn(output, target)
            error = get_error(output, target)
            print(f'Accuracy = {100-error}')
            # TODO: log the loss and accuracy
            # wandb.log({"loss_train_classifier": loss})
            # wandb.log({"error_train_classifier": error})
            _, pred = torch.max(output, dim=1)
            epoch_true = epoch_true + torch.sum(pred == target).item()

            loss.backward()
            optimizer.step()
            epoch_total += target.size(0)
            metrics.update(n=data.size(0), loss=loss.item(), error=error)
        logging.info(f"[epoch {epoch}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))
        print(f"Accuracy----> {100 * epoch_true / epoch_total}")

    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size,
                             num_workers=2, drop_last=False)
    print(next(iter(val_loader)))

    model.eval()
    epoch_true = 0
    epoch_total = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            output = model(input_ids=sent_id, attention_mask=mask)

            _, pred = torch.max(output, dim=1)
            epoch_true = epoch_true + torch.sum(pred == labels).item()
            epoch_total += labels.size(0)

        print(f"Model Acc--->{100 * epoch_true / epoch_total}")



if __name__ == "__main__":
    train_bert_feat(config={'batch':32,
                            'epochs':1,
                            'optimizer':'adamW',
                            'lr':5e-5,
                            'weight_decay': 0.0001,})
