from transformers import BertModel
import torch.nn as nn
from task2vec_nlp import ProbeNetwork
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

class BERT(ProbeNetwork):
    def __init__(self, classes):
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
