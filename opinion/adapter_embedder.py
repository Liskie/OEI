"""
Adapter transformers for AllenNLP.
Parameter-Efficient Transfer Learning for NLP. https://arxiv.org/abs/1902.00751
"""

from typing import Optional, Dict, Any, Union, List

from overrides import overrides

import torch
import torch.nn as nn
from torch.nn.functional import linear

from transformers import BertModel
from transformers.models.bert.modeling_bert import BertOutput, BertSelfOutput

from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import TokenEmbedder, PretrainedTransformerEmbedder
from allennlp.nn import util, Activation


@TokenEmbedder.register("adapter_transformer")
class AdapterTransformerEmbedder(PretrainedTransformerEmbedder):
    """
    目前只针对bert结构，插入adapter.
    """
    def __init__(
        self,
        model_name: str,
        *,
        adapter_layers: int = 12,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        external_param: Union[bool, List[bool]] = False,
        max_length: int = None,
        last_layer_only: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=False,
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )
        self.adapters = insert_adapters(
            adapter_layers, adapter_kwargs, external_param, self.transformer_model)
        self.adapter_layers = adapter_layers
        self.adapter_kwargs = adapter_kwargs


@TokenEmbedder.register("adapter_transformer_mismatched")
class AdapterTransformerMismatchedEmbedder(TokenEmbedder):
    """
    The adapter version of `PretrainedTransformerMismatchedEmbedder`.
    Just replaced `self._matched_embedder`.
    """

    def __init__(
        self,
        model_name: str,
        adapter_layers: int = 12,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        external_param: Union[bool, List[bool]] = False,
        max_length: int = None,
        last_layer_only: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = AdapterTransformerEmbedder(
            model_name,
            adapter_layers=adapter_layers,
            adapter_kwargs=adapter_kwargs,
            external_param=external_param,
            max_length=max_length,
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters
        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.
        # Returns
        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings


def insert_adapters(
    adapter_layers: int, adapter_kwargs: Dict[str, Any],
    external_param: Union[bool, List[bool]], transformer_model: BertModel
) -> nn.ModuleList:
    """
    初始化 adapters, 插入到 BERT, 并返回 adapters. 目前只支持bert结构!
    # Parameters
    adapter_layers : `int`, required.
        从 BERT 最后一层开始，多少层插入adapter。
    adapter_kwargs : `Dict`, required.
        初始化 `Adapter` 的参数。
    external_param : `Union[bool, List[bool]]`
        adapter 的参数是否留空以便外部注入。
    transformer_model : `BertModel`
        预训练模型。
    # Returns
    adapters_groups : `nn.ModuleList`, required.
        所插入的所有 adapter, 用于绑定到模型中。
    """
    if not isinstance(transformer_model, BertModel):
        raise ConfigurationError("目前只支持bert结构")

    if isinstance(external_param, bool):
        param_place = [external_param for _ in range(adapter_layers)]
    elif isinstance(external_param, list):
        param_place = [False for _ in range(adapter_layers)]
        for i, e in enumerate(external_param, 1):
            param_place[-i] = e
    else:
        raise ConfigurationError("wrong type of external_param!")

    adapter_kwargs.update(in_features=transformer_model.config.hidden_size)
    adapters_groups = nn.ModuleList([
        nn.ModuleList([
            Adapter(external_param=param_place[i], **adapter_kwargs),
            Adapter(external_param=param_place[i], **adapter_kwargs)
        ]) for i in range(adapter_layers)
    ])

    for i, adapters in enumerate(adapters_groups, 1):
        layer = transformer_model.encoder.layer[-i]
        layer.output = AdapterBertOutput(layer.output, adapters[0])
        layer.attention.output = AdapterBertOutput(layer.attention.output, adapters[1])

    return adapters_groups


class Adapter(nn.Module):
    """
    Adapter module.
    """
    def __init__(self, in_features: int, adapter_size: int = 64, bias: bool = True,
                 activation: str = 'gelu', external_param: bool = False,
                 train_layer_norm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.adapter_size = adapter_size
        self.bias = bias
        self.train_layer_norm = train_layer_norm
        self.act_fn = Activation.by_name(activation)()  # GELU is the best one.

        if external_param:
            self.weight_down, self.weight_up = None, None
        else:
            self.weight_down = nn.Parameter(torch.Tensor(adapter_size, in_features))
            self.weight_up = nn.Parameter(torch.Tensor(in_features, adapter_size))
            self.reset_parameters()

        if external_param or not bias:
            self.bias_down, self.bias_up = None, None
        else:
            self.bias_down = nn.Parameter(torch.zeros(adapter_size))
            self.bias_up = nn.Parameter(torch.zeros(in_features))

    def reset_parameters(self):
        nn.init.normal_(self.weight_down, std=1e-3)
        nn.init.normal_(self.weight_up, std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_func = batched_linear if self.weight_down.dim() == 3 else linear
        x = linear_func(hidden_states, self.weight_down, self.bias_down)
        x = self.act_fn(x)
        x = linear_func(x, self.weight_up, self.bias_up)
        x = x + hidden_states
        return x


class AdapterBertOutput(nn.Module):
    """
    替代 BertOutput 和 BertSelfOutput
    """
    def __init__(self, base: Union[BertOutput, BertSelfOutput], adapter: Adapter):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter.forward
        for param in base.LayerNorm.parameters():
            param.requires_grad = adapter.train_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def batched_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    if b is not None:
        y = y + b.unsqueeze(1)
    return y
