import torch
import wandb
import datetime
from pathlib import Path
import argparse



print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

# Local
try:
    from nlp.nlp_model import BERT, T2VBertArch
    import task_similarity
    import task2vec_datasets
    from task2vec_datasets_nlp import benchmark_data, tenKGNAD, sb_10k, amazon_reviews_multi, cardiffnlp
    from task2vec import Task2Vec
    from task2vec_nlp import Task2VecNLP
    from models import get_model
except ImportError:
    from task2vec_datasets_nlp import benchmark_data, tenKGNAD, sb_10k, amazon_reviews_multi, cardiffnlp


def small_data(projectname="Task2VecVision", notes='',skip=6):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ('stl10', 'mnist', 'cifar10', 'cifar100', 'letters', 'kmnist')
    dataset_names = ('mnist', 'cifar10', 'cifar100', 'letters', 'kmnist','stl10',)
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
        embeddings.append(Task2Vec(probe_network, max_samples=1000, skip_layers=skip).embed(dataset))
        wandb.finish()
    wandb.init(
        # set the wandb project where this run will be logged
        project=projectname,
        notes=notes,
    )
    task_similarity.plot_distance_matrix(embeddings, dataset_names,filename='./vision_sets.png')
    wandb.finish()


def meta_nlp(projectname="T2VSanity", notes='',skip=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    project_name = projectname
    group_name = "Task2VecNLP"
    notes = notes
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
    probe_network = T2VBertArch.from_pretrained("bert-base-german-cased") #BERT(classes=len(label_map))
    probe_network.replace_head(num_labels=len(label_map))
    probe_network.set_layers()

    embedding_eng = Task2VecNLP(probe_network, max_samples=1000, skip_layers=skip).embed(train_data)
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
    probe_network = T2VBertArch.from_pretrained("bert-base-german-cased") #BERT(classes=len(label_map))
    probe_network.replace_head(num_labels=len(label_map))
    probe_network.set_layers()

    embedding_10kgnad = Task2VecNLP(probe_network, max_samples=1000, skip_layers=skip).embed(train_data)
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
    probe_network = T2VBertArch.from_pretrained("bert-base-german-cased") #BERT(classes=len(label_map))
    probe_network.replace_head(num_labels=len(label_map))
    probe_network.set_layers()

    embedding_sb10k = Task2VecNLP(probe_network, max_samples=1000, skip_layers=skip).embed(train_data)
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
    probe_network = T2VBertArch.from_pretrained("bert-base-german-cased")#BERT(classes=len(label_map))
    probe_network.replace_head(num_labels=len(label_map))
    probe_network.set_layers()

    embedding_amazon = Task2VecNLP(probe_network, max_samples=1000, skip_layers=skip).embed(train_data)
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
    probe_network = T2VBertArch.from_pretrained("bert-base-german-cased") #BERT(classes=len(label_map))
    probe_network.replace_head(num_labels=len(label_map))
    probe_network.set_layers()

    embedding_cardiff = Task2VecNLP(probe_network, max_samples=1000, skip_layers=skip).embed(train_data)
    embeddings.append(embedding_cardiff)

    task_similarity.plot_distance_matrix(embeddings,
                                         labels =('benchmark', '10kGNAD', 'sb_10k', 'amazon', 'cardiffnlp'),
                                         filename='./5_texts_cross.png')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='T2V app description')
    parser.add_argument('--exp_name', type=str,
                        help='Experiment Name', default="Task2VecSanity")
    parser.add_argument('--exp_notes', type=str,
                        help='Experiment Notes', default="Debugging Workflow")
    parser.add_argument('--skip_layer', type=int,
                        help='Skip Layer of Model', default=1)

    args = parser.parse_args()
    meta_nlp(projectname=args.exp_name, notes=args.exp_notes,skip=args.skip_layer)
    #small_data(projectname=args.exp_name,notes=args.exp_notes, skip=args.skip_layer)