from transformers import BertModel
import torch.nn as nn
from task2vec_nlp import ProbeNetwork


class BERTArch(ProbeNetwork):
    def __init__(self, bert, label_map):
        super(BERTArch, self).__init__()
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
        self.layers = [
            self.bert,
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2,
            self.softmax
        ]

    @property
    def classifier(self):
        return self.softmax

    def forward(self, sent_id, mask):
        # pass the inputs to the model
        # returns pooler_output and last_hidden_state
        outputs = self.bert(
            input_ids=sent_id,
            attention_mask=mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        pooler_output = outputs['pooler_output']
        x = self.fc1(pooler_output)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)
        return x


class BERT(ProbeNetwork):
    def __init__(self,classes):
        super(BERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(768, 512)
        self.out = nn.Linear(512, classes)

        # define the forward pass
        self.layers = [self.bert_model, self.fc1, self.out]

    @property
    def classifier(self):
        return self.out

    def forward(self, **kwargs):
        calculate_fim = kwargs['enable_fim']
        if calculate_fim:
            """Replaces the default forward so that we can forward features starting from any intermediate layer."""
            x = kwargs['x']
            x = self.fc1(x)
            x = self.out(x)
            return x
        else:
            input_ids = kwargs['input_ids']
            mask = kwargs['mask']
            outputs = self.bert_model(input_ids=input_ids, attention_mask=mask)
            pooler_output = outputs['pooler_output']
            out = self.fc1(pooler_output)
            out = self.out(out)
            return out
