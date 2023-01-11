# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import collections
import torchvision.transforms as transforms
import os
import json
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,random_split
from transformers import AutoTokenizer
from textutils import load_text_labels, tokenize, encode, load_pretrained_vectors
from nlp.data_processing import LoadingData
from datasets import get_dataset_config_names,load_dataset

_DATASETS = {}

Dataset = collections.namedtuple(
    'Dataset', ['trainset', 'testset'])


def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


def _get_transforms(augment=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    basic_transform = [transforms.ToTensor(), normalize]

    transform_train = []
    if augment:
        transform_train += [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_train += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    transform_train += basic_transform
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    transform_test += basic_transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def _get_mnist_transforms(augment=True, invert=False, transpose=False):
    transform = [
        transforms.ToTensor(),
    ]
    if invert:
        transform += [transforms.Lambda(lambda x: 1. - x)]
    if transpose:
        transform += [transforms.Lambda(lambda x: x.transpose(2, 1))]
    transform += [
        transforms.Normalize((.5,), (.5,)),
        transforms.Lambda(lambda x: x.expand(3, 32, 32))
    ]

    transform_train = []
    transform_train += [transforms.Pad(padding=2)]
    if augment:
        transform_train += [transforms.RandomCrop(32, padding=4)]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test += [transforms.Pad(padding=2)]
    transform_test += transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def _get_cifar_transforms(augment=True):
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    transform_train = []
    if augment:
        transform_train += [
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(32, padding=0),
            transforms.RandomHorizontalFlip()]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)
    transform_test = []
    transform_test += transform
    transform_test = transforms.Compose(transform_test)
    return transform_train, transform_test


def set_metadata(trainset, testset, config, dataset_name):
    trainset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': trainset.task_name,
    }
    testset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': testset.task_name,
    }
    return trainset, testset


@_add_dataset
def inat2018(root, config):
    from dataset.inat import iNat2018Dataset
    transform_train, transform_test = _get_transforms()
    trainset = iNat2018Dataset(root, split='train', transform=transform_train, task_id=config.task_id)
    testset = iNat2018Dataset(root, split='val', transform=transform_test, task_id=config.task_id)
    trainset, testset = set_metadata(trainset, testset, config, 'inat2018')
    return trainset, testset


def load_tasks_map(tasks_map_file):
    assert os.path.exists(tasks_map_file), tasks_map_file
    with open(tasks_map_file, 'r') as f:
        tasks_map = json.load(f)
        tasks_map = {int(k): int(v) for k, v in tasks_map.items()}
    return tasks_map


@_add_dataset
def cub_inat2018(root, config):
    """This meta-task is the concatenation of CUB-200 (first 25 tasks) and iNat (last 207 tasks).

    - The first 10 tasks are classification of the animal species inside one of 10 orders of birds in CUB-200
        (considering all orders except passeriformes).
    - The next 15 tasks are classification of species inside the 15 families of the order of passerifomes
    - The remaining 207 tasks are classification of the species inside each of 207 families in iNat

    As noted above, for CUB-200 10 taks are classification of species inside an order, rather than inside of a family
    as done in the iNat (recall order > family > species). This is done because CUB-200 has very few images
    in each family of bird (expect for the families of passeriformes). Hence, we go up step in the taxonomy and
    consider classification inside a orders and not families.
    """
    NUM_CUB = 25
    NUM_CUB_ORDERS = 10
    NUM_INAT = 207
    assert 0 <= config.task_id < NUM_CUB + NUM_INAT
    transform_train, transform_test = _get_transforms()
    if 0 <= config.task_id < NUM_CUB:
        # CUB
        from dataset.cub import CUBTasks, CUBDataset
        tasks_map_file = os.path.join(root, 'cub/CUB_200_2011', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = tasks_map[config.task_id]

        if config.task_id < NUM_CUB_ORDERS:
            # CUB orders
            train_tasks = CUBTasks(CUBDataset(root, split='train'))
            trainset = train_tasks.generate(task_id=task_id,
                                            use_species_names=True,
                                            transform=transform_train)
            test_tasks = CUBTasks(CUBDataset(root, split='test'))
            testset = test_tasks.generate(task_id=task_id,
                                          use_species_names=True,
                                          transform=transform_test)
        else:
            # CUB passeriformes families
            train_tasks = CUBTasks(CUBDataset(root, split='train'))
            trainset = train_tasks.generate(task_id=task_id,
                                            task='family',
                                            taxonomy_file='passeriformes.txt',
                                            use_species_names=True,
                                            transform=transform_train)
            test_tasks = CUBTasks(CUBDataset(root, split='test'))
            testset = test_tasks.generate(task_id=task_id,
                                          task='family',
                                          taxonomy_file='passeriformes.txt',
                                          use_species_names=True,
                                          transform=transform_test)
    else:
        # iNat2018
        from dataset.inat import iNat2018Dataset
        tasks_map_file = os.path.join(root, 'inat2018', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = tasks_map[config.task_id - NUM_CUB]

        trainset = iNat2018Dataset(root, split='train', transform=transform_train, task_id=task_id)
        testset = iNat2018Dataset(root, split='val', transform=transform_test, task_id=task_id)
    trainset, testset = set_metadata(trainset, testset, config, 'cub_inat2018')
    return trainset, testset


@_add_dataset
def imat2018fashion(root, config):
    NUM_IMAT = 228
    assert 0 <= config.task_id < NUM_IMAT
    from dataset.imat import iMat2018FashionDataset, iMat2018FashionTasks
    transform_train, transform_test = _get_transforms()
    train_tasks = iMat2018FashionTasks(iMat2018FashionDataset(root, split='train'))
    trainset = train_tasks.generate(task_id=config.task_id,
                                    transform=transform_train)
    test_tasks = iMat2018FashionTasks(iMat2018FashionDataset(root, split='validation'))
    testset = test_tasks.generate(task_id=config.task_id,
                                  transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'imat2018fashion')
    return trainset, testset


@_add_dataset
def split_mnist(root, config):
    assert isinstance(config.task_id, tuple)
    from dataset.mnist import MNISTDataset, SplitMNISTTask
    transform_train, transform_test = _get_mnist_transforms()
    train_tasks = SplitMNISTTask(MNISTDataset(root, train=True))
    trainset = train_tasks.generate(classes=config.task_id, transform=transform_train)
    test_tasks = SplitMNISTTask(MNISTDataset(root, train=False))
    testset = test_tasks.generate(classes=config.task_id, transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'split_mnist')
    return trainset, testset


@_add_dataset
def split_cifar(root, config):
    assert 0 <= config.task_id < 11
    from dataset.cifar import CIFAR10Dataset, CIFAR100Dataset, SplitCIFARTask
    transform_train, transform_test = _get_cifar_transforms()
    train_tasks = SplitCIFARTask(CIFAR10Dataset(root, train=True), CIFAR100Dataset(root, train=True))
    trainset = train_tasks.generate(task_id=config.task_id, transform=transform_train)
    test_tasks = SplitCIFARTask(CIFAR10Dataset(root, train=False), CIFAR100Dataset(root, train=False))
    testset = test_tasks.generate(task_id=config.task_id, transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'split_cifar')
    return trainset, testset


@_add_dataset
def cifar10_mnist(root, config):
    from dataset.cifar import CIFAR10Dataset
    from dataset.mnist import MNISTDataset
    from dataset.expansion import UnionClassificationTaskExpander
    transform_train, transform_test = _get_cifar_transforms()
    trainset = UnionClassificationTaskExpander(merge_duplicate_images=False)(
        [CIFAR10Dataset(root, train=True), MNISTDataset(root, train=True, expand=True)], transform=transform_train)
    testset = UnionClassificationTaskExpander(merge_duplicate_images=False)(
        [CIFAR10Dataset(root, train=False), MNISTDataset(root, train=False, expand=True)], transform=transform_test)
    return trainset, testset


@_add_dataset
def cifar10(root):
    from torchvision.datasets import CIFAR10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR10(root, train=True, transform=transform, download=True)
    testset = CIFAR10(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def cifar100(root):
    from torchvision.datasets import CIFAR100
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR100(root, train=True, transform=transform, download=True)
    testset = CIFAR100(root, train=False, transform=transform)
    return trainset, testset


def _convert_transform(x):
    return x.convert("RGB")


@_add_dataset
def mnist(root):
    from torchvision.datasets import MNIST
    transform = transforms.Compose([
        _convert_transform,
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
    ])
    trainset = MNIST(root, train=True, transform=transform, download=True)
    testset = MNIST(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def letters(root):
    from torchvision.datasets import EMNIST
    transform = transforms.Compose([
        _convert_transform,
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
    ])
    trainset = EMNIST(root, train=True, split='letters', transform=transform, download=True)
    testset = EMNIST(root, train=False, split='letters', transform=transform)
    return trainset, testset


@_add_dataset
def kmnist(root):
    from torchvision.datasets import KMNIST
    transform = transforms.Compose([
        _convert_transform,
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    trainset = KMNIST(root, train=True, transform=transform, download=True)
    testset = KMNIST(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def stl10(root):
    from torchvision.datasets import STL10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = STL10(root, split='train', transform=transform, download=True)
    testset = STL10(root, split='test', transform=transform)
    trainset.targets = trainset.labels
    testset.targets = testset.labels
    return trainset, testset



@_add_dataset
def text_cnn(root):
    # tokenising, encoding etc has been done
    # return the trainset and test set here.  TensorDataset
    # Tokenize data
    print("Tokenizing...\n")
    texts, labels = load_text_labels(root = 'data')
    tokenized_texts, word2idx, max_len = tokenize(texts)
    input_ids = encode(tokenized_texts, word2idx, max_len)
    # Load pretrained vectors
    print("Load Vectors...\n")
    embedding_path = os.path.join(os.getcwd(), 'embeddings', 'crawl-300d-2M.vec')
    embeddings = load_pretrained_vectors(word2idx, embedding_path)
    embeddings = torch.tensor(embeddings)

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=42)

    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels = \
        tuple(torch.tensor(data) for data in
              [train_inputs, val_inputs, train_labels, val_labels])

    train_data = TensorDataset(train_inputs, train_labels)
    val_data = TensorDataset(val_inputs, val_labels)
    train_data.targets = train_labels
    val_data.targets = val_labels

    train_data.pretrained_embedding=embeddings

    return train_data, val_data

@_add_dataset
def tenKGNAD(root):
    df = pd.read_csv("https://raw.githubusercontent.com/tblock/10kGNAD/master/articles.csv",
                     encoding="utf-8",
                     delimiter=";",
                     quotechar="'", names=["label", "text"])

    texts = df.text.values
    label_cats = df.label.astype('category').cat

    # List of label names (str)
    label_names = label_cats.categories

    # List of label ids (int, in range (0,num_classes-1))
    labels = label_cats.codes

    # TOKENIZE
    model_name = "bert-base-german-cased"
    MAX_INPUT_LENGTH = 192

    # Load the pretrained BERT tokenizer.
    print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

    # Tokenize all of the sentences and map the tokens to their word IDs
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_INPUT_LENGTH,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    # Tensor DataSet
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 80-10-10 train-validation-test split

    # Calculate the number of samples to include in each set.
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset,label_names




def makedataset_sb_10k(hf_dataset):
    df = pd.DataFrame(hf_dataset)
    df.drop('source', axis=1)
    texts = df.text.values
    label_cats = df.label.astype('category').cat
    # List of label names (str)
    label_names = label_cats.categories

    # List of label ids (int, in range (0,num_classes-1))
    labels = label_cats.codes

    model_name = "bert-base-german-cased"
    MAX_INPUT_LENGTH = 192

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_INPUT_LENGTH,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset,label_names


@_add_dataset
def sb_10k(root='./'):
    dataset_train = load_dataset('tyqiangz/multilingual-sentiments', 'german', split='train')
    train_dataset, _ = makedataset_sb_10k(dataset_train)

    dataset_val = load_dataset('tyqiangz/multilingual-sentiments', 'german', split='validation')
    val_dataset, label_names = makedataset_sb_10k(dataset_val)

    return train_dataset, val_dataset, label_names


@_add_dataset
def benchmark_data(root):
    ld = LoadingData()
    train_df = ld.train_data_frame
    label_map, id2label = ld.intent_to_cat, ld.cat_to_intent

    train_text, val_text, train_labels, val_labels = train_test_split(train_df['query'], train_df['category'],
                                                                      random_state=2018,
                                                                      test_size=0.2,
                                                                      stratify=train_df['category'])

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", )
    seq_len = [len(i.split()) for i in train_text]
    max_seq_len = max(seq_len)

    # tokenize and encode sequences in the training set
    if max_seq_len > 512:
        max_seq_len = 512
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())
    print("train_y:", train_y)
    # for validation set
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())
    print("val_y:", val_y)
    # define a batch size

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)
    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    return train_data, val_data,label_map, id2label


def get_dataset(root, config=None):
    return _DATASETS[config.name](os.path.expanduser(root), config)
