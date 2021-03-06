"""
Adapter with transformers and PyTorch.

Parameter-Efficient Transfer Learning for NLPParameter-Efficient Transfer Learning for NLP
https://arxiv.org/abs/1902.00751
https://github.com/google-research/adapter-bert
"""

from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.functional import linear

from transformers.models.bert.modeling_bert import BertModel


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


def batch_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    y = y + b.unsqueeze(1)
    return y


class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_size, external_param=False):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size
        self.act_fn = nn.GELU()

        if external_param:
            self.params = [None, None, None, None]
        else:
            self.params = nn.ParameterList([
                nn.Parameter(torch.Tensor(bottleneck_size, in_features)),
                nn.Parameter(torch.zeros(bottleneck_size)),
                nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                nn.Parameter(torch.zeros(in_features))
            ])
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.params[0], std=1e-3)
        nn.init.normal_(self.params[2], std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_forward = batch_linear if self.params[0].dim() == 3 else linear
        x = linear_forward(hidden_states, self.params[0], self.params[1])
        x = self.act_fn(x)
        x = linear_forward(x, self.params[2], self.params[3])
        x = x + hidden_states
        return x


class AdapterBertOutput(nn.Module):
    """
    替代BertOutput和BertSelfOutput
    """
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterBertModel(nn.Module):
    def __init__(self,
                 name_or_path_or_model: Union[str, BertModel],
                 adapter_size: int = 128,
                 external_param: Union[bool, List[bool]] = False,
                 **kwargs):
        super().__init__()
        if isinstance(name_or_path_or_model, str):
            self.bert = BertModel.from_pretrained(name_or_path_or_model)
        else:
            self.bert = name_or_path_or_model

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(
                self.bert.config.num_hidden_layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(
                self.bert.config.num_hidden_layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ValueError("wrong type of external_param!")

        self.adapters = nn.ModuleList([nn.ModuleList([
                Adapter(self.bert.config.hidden_size, adapter_size, e),
                Adapter(self.bert.config.hidden_size, adapter_size, e)
            ]) for e in param_place
        ])

        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[i][0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[i][1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size

    def forward(self,
                input_ids: torch.Tensor,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        column_size = (input_ids.size(0), 1)
        cls_ids = input_ids.new_full(column_size, 101)
        sep_ids = input_ids.new_zeros(column_size)
        if mask is None:
            sep_ids.fill_(102)

        input_ids = torch.cat((cls_ids, input_ids, sep_ids), dim=-1)
        if mask is not None:
            batch_length_add_one = (mask.sum(-1) + 1).tolist()
            ones, zeros = mask.new_ones(column_size), mask.new_zeros(column_size)
            mask = torch.cat((ones, mask, zeros), dim=-1).float()
            for i, index in enumerate(batch_length_add_one):
                input_ids[i, index], mask[i, index] = 102, True

        bert_output = self.bert(input_ids, mask)
        return bert_output[0][:, 1:-1, :]
