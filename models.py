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


import torch.utils.model_zoo as model_zoo

import torchvision.models.resnet as resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from task2vec import ProbeNetwork

_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn


class ResNet(resnet.ResNet, ProbeNetwork):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        # Saves the ordered list of layers. We need this to forward from an arbitrary intermediate layer.
        self.layers = [
            self.conv1, self.bn1, self.relu,
            self.maxpool, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
            lambda z: torch.flatten(z, 1), self.fc
        ]

    @property
    def classifier(self):
        return self.fc

    # @ProbeNetwork.classifier.setter
    # def classifier(self, val):
    #     self.fc = val

    # Modified forward method that allows to start feeding the cached activations from an intermediate
    # layer of the network
    def forward(self, x, start_from=0):
        """Replaces the default forward so that we can forward features starting from any intermediate layer."""
        for layer in self.layers[start_from:]:
            x = layer(x)
        return x


@_add_model
def resnet18(pretrained=False, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model: ProbeNetwork = ResNet(block=resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(resnet.model_urls['resnet18'])
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


@_add_model
def resnet34(pretrained=False, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        state_dict = model_zoo.load_url(resnet.model_urls['resnet34'])
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model.load_state_dict(state_dict, strict=False)
    return model


def get_model(model_name, pretrained=False, num_classes=1000, **kwargs):
    try:
        return _MODELS[model_name](pretrained=pretrained, num_classes=num_classes, **kwargs)
    except KeyError:
        raise ValueError(f"Architecture {model_name} not implemented.")


class CNN_NLP(ProbeNetwork):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)
        # Conv Network

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.conv1 =  nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[0],
                      kernel_size=filter_sizes[0])
        self.conv2 =  nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[1],
                      kernel_size=filter_sizes[1])
        self.conv3 =  nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[2],
                      kernel_size=filter_sizes[2])

        self.id1= nn.Identity() # needed to maintain the feature input
        self.id2 = nn.Identity()
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        # layers 2,6 have the input features cached.
        self.layers = [
            # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
            lambda input_ids: self.embedding(input_ids).float(),
            # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
            # Output shape: (b, embed_dim, max_len)
            lambda reshape: reshape.permute(0, 2, 1),
            self.id1,
            # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
            lambda x_conv_list:[F.relu(conv1d(x_conv_list)) for conv1d in self.conv1d_list],
            # Max pooling. Output shape: (b, num_filters[i], 1)
            lambda x_pool_list: [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                 for x_conv in x_pool_list],
            # Concatenate x_pool_list to feed the fully connected layer.
            # Output shape: (b, sum(num_filters))
            lambda x_fc: torch.cat([x_pool.squeeze(dim=2) for x_pool in x_fc],
                                   dim=1),
            self.id2,
            # Compute logits. Output shape: (b, n_classes)
            self.dropout,
            self.fc
        ]

    @property
    def classifier(self):
        return self.fc

    def forward(self, input_ids,start_from=0):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """
        for layer in self.layers[start_from:]:
            input_ids = layer(input_ids)
        return input_ids
@_add_model
def cnn_text(pretrained=False, num_classes=1000, **kwargs):
    cnn_model = CNN_NLP(pretrained_embedding=kwargs['pretrained_embedding'],
                        freeze_embedding=kwargs['freeze_embedding'],
                        vocab_size=kwargs['vocab_size'],
                        embed_dim=kwargs['embed_dim'],
                        filter_sizes=kwargs['filter_sizes'],
                        num_filters=kwargs['num_filters'],
                        num_classes=2,
                        dropout=0.5)

    return cnn_model
