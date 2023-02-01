from typing import Union, Tuple, Optional, List
from transformers import BertModel, BertForSequenceClassification, BertConfig, BertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertEncoder

import torch
import torch.nn as nn
from task2vec_nlp import ProbeNetwork
from pathlib import Path

print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
logger = logging.get_logger(__name__)

def _bert_classifier_hook(layer, inputs):
    if not hasattr(layer, 'input_features'):
        layer.input_features = []
    layer.input_features.append(inputs[0].data.cpu().clone())

#The input contains only the positional arguments
# for encoder, store all the inputs so when it goes through the loop, use that. See size.
def _bert_hook(layer, inputs):
    if not hasattr(layer, 'input_features'):
        # init a dict of empty arrays.
        layer.input_features = {key: [] for key in range(len(inputs))}

    # append to arrays
    for i, item in enumerate(inputs):
        # copy any tensor values
        if isinstance(item, torch.Tensor):
            layer.input_features[i].append(item.data.cpu().clone())
        else:
            layer.input_features[i].append(item)


class BERT(ProbeNetwork):
    def __init__(self, classes):
        super(BERT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(768, 512)
        self.out = nn.Linear(512, classes)

    @property
    def classifier(self):
        return self.out

    def replace_head(self, num_labels):
        self.out = nn.Linear(512, num_labels)

    def set_layers(self):
        # define the forward pass
        self.layers = [self.bert_model, self.fc1, self.out]

    def store_input_size(self , input_ids):
        self.input_shape = input_ids.size()
    def forward(self, input_ids, attention_mask, enable_fim=False,**kwargs):
        calculate_fim = enable_fim
        if calculate_fim:
            """Replaces the default forward so that we can forward features starting from any intermediate layer."""
            x = kwargs['x']
            x = self.fc1(x)
            x = self.out(x)
            return x
        else:
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooler_output = outputs['pooler_output']
            out = self.fc1(pooler_output)
            out = self.out(out)
            return out


"""
Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
Because encoder, pooler and embeddings aren't available to import
"""


class T2VBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            start_from=0,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer[start_from:]):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,# not needed
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,# not needed
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,# not needed
                    output_attentions, # not needed
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class T2VBertArch(BertModel, ProbeNetwork):
    def __init__(self, config=BertConfig(), add_pooling_layer=True):
        super(BertModel, self).__init__(config=config)
        self.config = config
        self.num_labels = config.num_labels

        self.embeddings = BertEmbeddings(config)
        self.encoder = T2VBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self._classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        # after this, init the last layer and set up the layer list for the forward pass

    @property
    def classifier(self):
        return self._classifier

    def replace_head(self, num_labels):
        self._classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.post_init()

    def store_input_size(self , input_ids):
        self.input_shape = input_ids.size()
    def set_layers(self):
        self.layers = [layer for layer in self.encoder.layer]
        self.layers.append(self._classifier)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            start_from=0,
            enable_fim=False,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if enable_fim:
            # TODO: if it's FIM, use the saved value of input_ids.size() and encoder_extended_attention_mask
            batch_size, seq_length = self.input_shape
            input_shape = self.input_shape
            embedding_output = input_ids
            extended_attention_mask = attention_mask
            # Won't work if there' something needed. Store this as batches.
            encoder_extended_attention_mask = self.encoder_extended_attention_mask
        else:
            batch_size, seq_length = input_shape

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if not enable_fim:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

            # store the encoder_extended_attention_mask, then use later
            self.encoder_extended_attention_mask = encoder_extended_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
                )


        #######
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            start_from=start_from,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            # don't return here
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]

        # don't return here
        outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        bert_output = outputs[1]
        bert_output = self.dropout(bert_output)
        logits = self.classifier(bert_output)
        return logits


def test_model():
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = T2VBertArch.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs, start_from=7)
    return outputs


if __name__ == "__main__":
    model1 = BertModel.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model1(**inputs)
    custom_out = test_model()
    print()
