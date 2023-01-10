from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch.nn as nn


class BERT_Arch(nn.Module):
    def __init__(self, bert, label_map):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, len(label_map))

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

        # define the forward pass

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        # returns pooler_output and last_hidden_state
        outputs = self.bert(
            input_ids=sent_id,
            attention_mask=mask
        )
        x = self.fc1(outputs['pooler_output'])

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)
        return x