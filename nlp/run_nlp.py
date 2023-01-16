import numpy as np  # linear algebra
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import nltk
import warnings
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

# local import
from data_processing import LoadingData
from nlp_model import BERT
from train_nlp_model import training_loop

nltk.download("all", quiet=False)
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # specify GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")

    ld = LoadingData()
    train_df = ld.train_data_frame
    label_map, id2label = ld.intent_to_cat, ld.cat_to_intent

    train_text, val_text, train_labels, val_labels = train_test_split(train_df['query'], train_df['category'],
                                                                      random_state=2018,
                                                                      test_size=0.2,
                                                                      stratify=train_df['category'])

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", )
    bert = BertModel.from_pretrained("bert-base-uncased", force_download=True)

    seq_len = [len(i.split()) for i in train_text]
    max_seq_len = max(seq_len)
    print(max_seq_len)

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
    batch_size = 16

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    # model = BERT_Arch(bert, label_map)
    model = BERT(classes=len(label_map))

    # push the model to GPU
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # compute the class weights
    class_wts = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(train_labels),
                                     y=train_labels)
    class_wts = dict(zip(np.unique(train_labels), class_wts))
    print(list(class_wts.values()))

    # convert class weights to tensor
    weights = torch.tensor(list(class_wts.values()), dtype=torch.float)
    weights = weights.to(device)

    # loss function
    # optional argument weight should be a 1D Tensor assigning weight to each of the classes
    # useful when you have an unbalanced training set.
    # loss_fn = nn.NLLLoss(weight=weights)
    loss_fn = nn.NLLLoss()
    # number of training epochs
    epochs = 2
    training_loop(model, optimizer, label_map, id2label, epochs, train_dataloader, loss_fn, val_dataloader, device)
