import argparse
import torch
import random
import wandb

from pathlib import Path
import os

import tf_idf
from tf_idf import run_tf_idf

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter,\
    UniformFloatHyperparameter
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario



X_train, y_train = np.random.randint(2, size=(20, 2)), np.random.randint(2, size=20)
X_val, y_val = np.random.randint(2, size=(5, 2)), np.random.randint(2, size=5)


def train_random_forest(config):
    model = RandomForestClassifier(max_depth=config["depth"])
    model.fit(X_train, y_train)

    # Define the evaluation metric as return
    return 1 - model.score(X_val, y_val)


# local
try:
    from tf_idf import run_tf_idf
except ImportError:
    from tf_idf import run_tf_idf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TF-IDF BOHB')
    parser.add_argument('--exp_seed', type=int,
                        help='Experiment Seed', default=9)
    args = parser.parse_args()
    torch.manual_seed(args.exp_seed)
    random.seed(args.exp_seed)
    np.random.seed(args.exp_seed)
    g = torch.Generator()
    g.manual_seed(args.exp_seed)

    configspace = ConfigurationSpace()
    # Define your hyperparameters

    lr = UniformFloatHyperparameter('lr', lower=1e-5, upper= 1e-2, default_value=1e-3, log=True)
    # feature_size = CategoricalHyperparameter('feature_size', choices=[1024, 512, 7000, 256],default_value=512)
    batch = CategoricalHyperparameter('batch', choices=[32, 512, 64, 128, 256], default_value=32)
    epochs = UniformIntegerHyperparameter('epochs', lower=3,upper=10, default_value=10)
    configspace.add_hyperparameters([lr, batch, epochs])

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 5,  # Max number of function evaluations (the more the better)
        "cs": configspace,
        "output_dir": "./smac_logs",
        "deterministic": True,

    })

    smac = SMAC4MF(scenario=scenario, tae_runner=run_tf_idf, intensifier_kwargs= {"initial_budget": 1, "max_budget": 25})
    best_found_config = smac.optimize()
    print(f"BEST-->{best_found_config}")













