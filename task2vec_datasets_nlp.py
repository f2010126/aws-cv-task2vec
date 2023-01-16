import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer
from nlp.data_processing import LoadingData
from datasets import load_dataset
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())


def tenKGNAD(root):
    df = pd.read_csv("https://raw.githubusercontent.com/tblock/10kGNAD/master/articles.csv",
                     encoding="utf-8",
                     delimiter=";",
                     quotechar="'", names=["label", "text"])

    texts = df.text.values
    label_cats = df.label.astype('category').cat  # cast as a categorical variable

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
    return train_dataset, val_dataset, label_names


def makedataset_hf(hf_dataset, drop_columns, label_column, data_column):
    df = pd.DataFrame(hf_dataset)
    df = df.drop(drop_columns, axis=1)
    texts = df[data_column].values
    label_cats = df[label_column].astype('category').cat
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

    return dataset, label_names


def sb_10k(root='./'):
    dataset_train = load_dataset('tyqiangz/multilingual-sentiments', 'german', split='train')
    train_dataset, _ = makedataset_hf(dataset_train, drop_columns=['source'], label_column='label', data_column='text')

    dataset_val = load_dataset('tyqiangz/multilingual-sentiments', 'german', split='validation')
    val_dataset, label_names = makedataset_hf(dataset_val, drop_columns=['source'], label_column='label',
                                              data_column='text')

    return train_dataset, val_dataset, label_names


def amazon_reviews_multi(root='./'):
    dataset_train = load_dataset("amazon_reviews_multi", 'de', split='train')
    train_dataset, _ = makedataset_hf(dataset_train,
                                      drop_columns=['language', 'reviewer_id', 'product_id', 'review_id'],
                                      label_column='product_category', data_column='review_body')

    dataset_val = load_dataset("amazon_reviews_multi", 'de', split='validation')
    val_dataset, label_names = makedataset_hf(dataset_val,
                                              drop_columns=['language', 'reviewer_id', 'product_id', 'review_id'],
                                              label_column='product_category', data_column='review_body')

    return train_dataset, val_dataset, label_names


def cardiffnlp(root='./'):
    dataset_train = load_dataset("cardiffnlp/tweet_sentiment_multilingual", 'german', split='train')
    train_dataset, _ = makedataset_hf(dataset_train,
                                      drop_columns=[],
                                      label_column='label', data_column='text')

    dataset_val = load_dataset("cardiffnlp/tweet_sentiment_multilingual", 'german', split='validation')
    val_dataset, label_names = makedataset_hf(dataset_val,
                                              drop_columns=[],
                                              label_column='label', data_column='text')

    return train_dataset, val_dataset, label_names


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

    return train_data, val_data, label_map, id2label
