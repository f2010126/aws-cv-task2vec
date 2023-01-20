import torch
import wandb
import datetime
from pathlib import Path

print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

# Local
try:
    from nlp.nlp_model import BERT
    import task_similarity
    import task2vec_datasets
    from task2vec_datasets_nlp import benchmark_data, tenKGNAD, sb_10k, amazon_reviews_multi, cardiffnlp
    from task2vec import Task2Vec
    from task2vec_nlp import Task2VecNLP
    from models import get_model
except ImportError:
    from task2vec_datasets_nlp import benchmark_data, tenKGNAD, sb_10k, amazon_reviews_multi, cardiffnlp


def small_data(projectname="Task2VecVision"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    dataset_names = ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    # Change `root` with the directory you want to use to download the datasets
    dataset_list = [task2vec_datasets.__dict__[name](root='./data')[0] for name in dataset_names]

    embeddings = []
    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        wandb.init(
            # set the wandb project where this run will be logged
            project=projectname,
            job_type=name,
            config={
                "model": 'resnet34',
                "dataset": name,
                "device": device,
            }
        )
        probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets) + 1))  # .cuda()
        embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset))
        wandb.finish()
    wandb.init(
        # set the wandb project where this run will be logged
        project=projectname,
    )
    task_similarity.plot_distance_matrix(embeddings, dataset_names,filename='./vision_sets.png')
    wandb.finish()


def meta_nlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    project_name = f"Task2VecNLP_CrossEntropy"
    group_name = "Task2VecNLP"
    notes = "Changed some Params to match BERT's finetune"

    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        notes=notes,
        group=group_name,
        job_type='benchmark_nlp',
        config={
            "model": 'BERT',
            "dataset": 'benchmark_nlp',
            "device": device,
        }
    )
    train_data, val_data, label_map, _ = benchmark_data(root='./')
    probe_network = BERT(classes=len(label_map))
    embedding_eng = Task2VecNLP(probe_network, max_samples=1000, skip_layers=1).embed(train_data)
    wandb.finish()
    embeddings.append(embedding_eng)

    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        group=group_name,
        job_type='tenKGNAD',
        config={
            "model": 'BERT',
            "dataset": 'tenKGNAD',
            "device": device,
        }
    )
    train_data, val_data, label_map = tenKGNAD(root='./')
    probe_network = BERT(classes=len(label_map))
    embedding_10kgnad = Task2VecNLP(probe_network, max_samples=1000, skip_layers=1).embed(train_data)
    wandb.finish()
    embeddings.append(embedding_10kgnad)

    # Sentiment Corpus Deutsch SB-10k
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        group=group_name,
        job_type='sb_10k',
        config={
            "model": 'BERT',
            "dataset": 'sb_10k',
            "device": device,
        }
    )
    train_data, val_data, label_map = sb_10k()
    probe_network = BERT(classes=len(label_map))
    embedding_sb10k = Task2VecNLP(probe_network, max_samples=1000, skip_layers=1).embed(train_data)
    wandb.finish()
    embeddings.append(embedding_sb10k)

    # Amazon Multi Reviews (might be Usage?)
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        group=group_name,
        job_type='amazon_reviews_multi',
        config={
            "model": 'BERT',
            "dataset": 'amazon_reviews_multi',
            "device": device,
        }
    )
    train_data, val_data, label_map = amazon_reviews_multi()
    probe_network = BERT(classes=len(label_map))
    embedding_amazon = Task2VecNLP(probe_network, max_samples=1000, skip_layers=1).embed(train_data)
    wandb.finish()
    embeddings.append(embedding_amazon)

    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        group=group_name,
        job_type='cardiffnlp',
        config={
            "model": 'BERT',
            "dataset": 'cardiffnlp',
            "device": device,
        }
    )
    train_data, val_data, label_map = cardiffnlp(root='./')
    probe_network = BERT(classes=len(label_map))
    embedding_cardiff = Task2VecNLP(probe_network, max_samples=1000, skip_layers=1).embed(train_data)
    embeddings.append(embedding_cardiff)

    task_similarity.plot_distance_matrix(embeddings,
                                         labels =('benchmark', '10kGNAD', 'sb_10k', 'amazon', 'cardiffnlp'),
                                         filename='./5_texts_cross.png')
    wandb.finish()


if __name__ == '__main__':
    #meta_nlp()
    small_data(projectname="Task2VecVisionAnnot")