from task2vec import Task2Vec
from task2vec_nlp import Task2VecNLP
from models import get_model
import datasets
from datasets import benchmark_data
import task_similarity
from transformers import BertModel
import torch
from nlp.nlp_model import BERT_Arch, BERT



def small_data():
    # ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    dataset_names = ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    # Change `root` with the directory you want to use to download the datasets
    dataset_list = [datasets.__dict__[name](root='./data')[0] for name in dataset_names]

    embeddings = []
    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets) + 1))  # .cuda()
        embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset))

    task_similarity.plot_distance_matrix(embeddings, dataset_names)


def text_data():
    dataset_names = ('text_cnn',)
    dataset_list = [datasets.__dict__[name](root='./data')[0] for name in dataset_names]
    embeddings = []

    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        probe_network = get_model(model_name='cnn_text', pretrained=True,
                                  pretrained_embedding=dataset.pretrained_embedding,
                                  freeze_embedding=False,
                                  vocab_size=None,
                                  embed_dim=300,
                                  filter_sizes=[3, 4, 5],
                                  num_filters=[100, 100, 100],
                                  num_classes=2,dropout=0.5)  # .cuda()
        embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=2).embed(dataset))

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, label_map, id2label = benchmark_data(root='r')

    probe_network = BERT()
    embedding =Task2VecNLP(probe_network, max_samples=1000, skip_layers=1).embed(train_data)
    print(f'embed {embedding}')


if __name__ == '__main__':
    benchmark()
