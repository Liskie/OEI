from typing import Dict, Set, Tuple, List, Any, cast, Union

import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList, init
from torch.distributions.beta import Beta

from nmnlp.common import Vocabulary
from nmnlp.common.trainer import format_metric, output
from nmnlp.embedding import build_word_embedding
from nmnlp.modules.dropout import WordDropout
from nmnlp.modules.encoder import LstmEncoder

from adapter import AdapterBertModel
from metric import ExactMatch, Binary, Proportional
from conditional_random_field import ConditionalRandomField


def build_model(name, **kwargs):
    m = {
        'tagger': Tagger,
        'ad': AdapterModel,
        'pga': PGAdapterModel,
        'mixmodel': MixupModel,
        'lstm': LSTMCrowd,
        'ft': Finetune
    }
    return m[name](**kwargs)


def tensor_like(data, t: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros_like(t)
    for i, l in enumerate(data):
        tensor[i, :len(l)] = torch.tensor(l, dtype=t.dtype, device=t.device)
    return tensor


class CRF(nn.Module):
    """ CRF classifier."""
    def __init__(
        self,
        num_tags: int,
        input_dim: int = 0,
        top_k: int = 1,
        reduction='sum',
        constraints: List[Tuple[int, int]] = None,
        include_start_end_transitions: bool = True,
        begin_tag_ids: Set = None,
        end_tag_ids: Set = None,
    ) -> None:
        super().__init__()
        if input_dim > 0:
            self.tag_projection = nn.Linear(input_dim, num_tags)
        else:
            self.tag_projection = None

        self.base = ConditionalRandomField(
            num_tags, reduction, constraints, include_start_end_transitions)
        self.top_k = top_k
        if include_start_end_transitions and not (begin_tag_ids or end_tag_ids):
            raise Exception("需指定 start, end !")
        else:
            for i in range(num_tags):
                if i not in begin_tag_ids:
                    self.base.start_transitions.data[i] = -10000
                if i not in end_tag_ids:
                    self.base.end_transitions.data[i] = -10000

    def forward(
        self, inputs: torch.FloatTensor, mask: torch.LongTensor,
        labels: torch.LongTensor = None, reduction: str = None,
    ) -> Dict[str, Any]:
        bool_mask = mask.bool()
        if self.tag_projection:
            inputs = self.tag_projection(inputs)
        scores = inputs * mask.unsqueeze(-1)

        best_paths = self.base.viterbi_tags(scores, bool_mask, top_k=self.top_k)
        # Just get the top tags and ignore the scores.
        tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        if labels is None:
            loss = torch.tensor(0, dtype=torch.float, device=inputs.device)
        else:
            # Add negative log-likelihood as loss
            loss = self.base(scores, labels, bool_mask, reduction)

        return dict(scores=scores, predicted_tags=tags, loss=loss)


class Tagger(nn.Module):
    """ a """
    def __init__(
        self,
        vocab: Vocabulary,
        word_embedding: Dict[str, Any],
        lstm_size: int = 400,
        input_namespace: str = 'words',
        label_namespace: str = 'labels',
        reduction: str = 'mean',
        save_embedding: bool = False,
        allowed: List[Tuple[int, int]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.word_embedding = build_word_embedding(num_embeddings=vocab.size_of(input_namespace),
                                                   vocab=vocab,
                                                   **word_embedding)

        if lstm_size > 0:
            self.lstm = LstmEncoder(
                self.word_embedding.output_dim, lstm_size,
                num_layers=1, bidirectional=True)
            feat_dim = self.lstm.output_dim
        else:
            self.lstm = lambda *args: args[0]
            feat_dim = self.word_embedding.output_dim

        num_tags = vocab.size_of(label_namespace)
        self.tag_projection = nn.Linear(feat_dim, num_tags)

        self.word_dropout = WordDropout(0.20)
        o_id = vocab.index_of('O', label_namespace)
        tag_to_id = vocab.token_to_index[label_namespace]
        begin_tag_ids = [o_id] + [i for t, i in tag_to_id.items() if t.startswith('B')]
        end_tag_ids = [o_id] + [i for t, i in tag_to_id.items() if t.startswith('I')]

        self.crf = CRF(num_tags, reduction=reduction, constraints=allowed,
                       begin_tag_ids=begin_tag_ids, end_tag_ids=end_tag_ids)
        args = (o_id, tag_to_id, True, True)
        self.metric = ExactMatch(*args)
        self.binary = Binary(*args)
        self.prop = Proportional(*args)
        self.save_embedding = save_embedding
        self.id_to_label = vocab.index_to_token[label_namespace]
        self.epoch = 0

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor,
        mask: torch.Tensor = None, labels: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        embedding = self.word_embedding(words, mask=mask, **kwargs)
        embedding = self.word_dropout(embedding)
        feature = self.lstm(embedding, lengths)
        scores = self.tag_projection(feature)
        output_dict = self.decode(scores, mask, labels, lengths)
        return output_dict

    def decode(self, scores, mask, labels, lengths):
        output_dict = self.crf(scores, mask, labels)
        if labels is not None:
            output_dict = self.add_metric(output_dict, labels, lengths)
        return output_dict

    def add_metric(self, output_dict, labels, lengths, prefix=''):
        prediction = tensor_like(output_dict['predicted_tags'], labels)
        for k in ('metric', 'binary', 'prop'):
            output_dict[k] = getattr(self, prefix + k)(prediction, labels, lengths)
        return output_dict

    def before_train_once(self, kwargs):
        self.epoch = kwargs['epoch']

    def after_process_one(self, metric, kwargs):
        self.print_detail(metric)

    def after_epoch_end(self, kwargs):
        self.print_detail(kwargs['metric'])

    def print_detail(self, metric):
        binary = self.binary.get_metric(reset=True)
        prop = self.prop.get_metric(reset=True)
        output(f"< Binary       > {format_metric(binary)}")
        output(f"< Proportional > {format_metric(prop)}")
        output(f"< Exact Match  > {format_metric(metric)}")

    def save(self, path):
        state_dict = self.state_dict()
        if not self.save_embedding:
            state_dict = {k: v for k, v in state_dict.items() if not self.drop_param(k)}
        torch.save(state_dict, path)

    def load(self, path_or_state, device):
        if isinstance(path_or_state, str):
            path_or_state = torch.load(path_or_state, map_location=device)
        info = self.load_state_dict(path_or_state, strict=False)
        missd = [i for i in info[0] if not self.drop_param(i)]
        if missd:
            print(missd)
        # print("model loaded.")

    def drop_param(_, name: str):
        return name.startswith('word_embedding.bert')


class AdapterModel(Tagger):
    def __init__(self,
                 vocab: Vocabulary,
                 adapter_size: int = 128,
                 external_param: Union[bool, List[bool]] = False,
                 self_eval: bool = False,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        self.word_embedding = AdapterBertModel(
            self.word_embedding.bert, adapter_size, external_param)
        self.self_eval = self_eval
        # self.not_output = False
        # self.out_dir = 'dev/out/ada-pg-self/'

    def drop_param(_, name: str):
        return super().drop_param(name) and 'LayerNorm' not in name

    # def before_time_start(self, _, trainer, kwargs):
    #     import os
    #     import shutil
    #     if self.not_output:
    #         return
    #     out_dir = f"dev/out/{trainer.prefix}/"
    #     if os.path.exists(out_dir):
    #         shutil.rmtree(out_dir)
    #     os.mkdir(out_dir)
    #     setattr(self, 'out_dir', out_dir)

    # def before_next_batch(self, kwargs):
    #     if self.training or self.not_output:
    #         return
    #     import json
    #     epoch, batch, out = kwargs['epoch'], kwargs['batch'], kwargs['output_dict']
    #     name = f"{'test' if epoch is None else 'dev'}-{self.epoch}"
    #     if hasattr(self, 'ann_id'):
    #         name += f"-ann{self.ann_id}"
    #     with open(f"{self.out_dir}/{name}.json", mode='a') as file:
    #         for ins, pred in zip(batch, out['predicted_tags']):
    #             one = dict(id=ins['id'], user=ins['user'],
    #                        prediction=[self.id_to_label[i] for i in pred])
    #             line = json.dumps(one) + '\n'
    #             file.write(line)
    #     return


class PGAdapterModel(AdapterModel):
    def __init__(self,
                 vocab: Vocabulary,
                 annotator_dim: int = 8,
                 annotator_num: int = 70,
                 adapter_size: int = 128,
                 num_adapters: int = 6,
                 batched_param: bool = False,
                 share_param: bool = False,
                 same_embedding: bool = False,
                 **kwargs):
        super().__init__(vocab, adapter_size, [True] * num_adapters, **kwargs)
        if same_embedding:
            w = torch.randn(annotator_dim).expand(annotator_num, -1).contiguous()
        else:
            w = torch.randn(annotator_num, annotator_dim)
        self.annotator_embedding = nn.Embedding(
            annotator_num, annotator_dim, _weight=w)  # max_norm=1.0

        dim = self.word_embedding.output_dim
        size = [2] if share_param else [num_adapters, 2]
        self.weight = ParameterList([
            Parameter(torch.Tensor(*size, adapter_size, dim, annotator_dim)),
            Parameter(torch.zeros(*size, adapter_size, annotator_dim)),
            Parameter(torch.Tensor(*size, dim, adapter_size, annotator_dim)),
            Parameter(torch.zeros(*size, dim, annotator_dim)),
        ])
        self.reset_parameters()
        self.adapter_size = adapter_size
        self.num_adapters = num_adapters
        self.batched_param = batched_param
        self.share_param = share_param

    def reset_parameters(self):
        # bound = 1e-2
        # init.uniform_(self.weight[0], -bound, bound)
        # init.uniform_(self.weight[2], -bound, bound)
        init.normal_(self.weight[0], std=1e-3)
        init.normal_(self.weight[2], std=1e-3)

    def set_annotator(self, user: torch.LongTensor):
        if (self.training or self.self_eval) and user is not None and user[0].item() != -1:  # expert = -1
            ann_emb = self.annotator_embedding(user[0] if self.batched_param else user)
        elif hasattr(self, 'scores'):
            weight = self.scores.softmax(0).unsqueeze(-1)
            ann_emb = self.annotator_embedding.weight.mul(weight).sum(0)
        elif hasattr(self, 'ann_id'):
            ann_emb = self.annotator_embedding.weight[self.ann_id]
        else:
            ann_emb = self.annotator_embedding.weight.mean(0)
        self.set_adapter_parameter(ann_emb)

    def set_adapter_parameter(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        param_list = [matmul(w, embedding) for w in self.weight]

        for i, adapters in enumerate(self.word_embedding.adapters[-self.num_adapters:]):
            for j, adapter in enumerate(adapters):
                params: List[torch.Tensor] = [p[j] if self.share_param else p[i, j] for p in param_list]
                setattr(adapter, 'params', params)

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        labels: torch.Tensor = None, user: torch.Tensor = None,
        embedding: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        if embedding is None:
            self.set_annotator(user)
        else:
            self.set_adapter_parameter(embedding)
        return super().forward(words, lengths, mask, labels, **kwargs)


class MixupModel(PGAdapterModel):
    """
    """
    def __init__(self, vocab, mode=0, timing=20, alpha=0.5, **kwargs):
        super().__init__(vocab, **kwargs)
        self.beta = Beta(alpha, alpha)
        self.pretraining = True
        self.mode = mode  # 1-ann mix, 2-text mix, 3-all mix, 4-mixup
        self.timing = timing

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        labels: torch.Tensor = None, user: torch.Tensor = None,
        l_tilde: torch.Tensor = None, u_tilde: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        if self.pretraining or not self.training:
            return super().forward(words, lengths, mask, labels, user, **kwargs)

        lam = self.beta.sample().to(words.device)
        index = torch.randperm(words.size(0))

        e = self.annotator_embedding(user)
        if self.mode in (1, 3):
            e_tilde = self.annotator_embedding(u_tilde) if self.mode == 0 else e[index]
            e = e * lam + e_tilde * (1 - lam)
        self.set_adapter_parameter(e)

        x = self.word_embedding(words, mask=mask)
        if self.mode in (2, 3, 4):
            x = x * lam + x[index] * (1 - lam)
            mask_tilde, lengths_tilde, l_tilde = mask[index], lengths[index], labels[index]
        else:
            mask_tilde, lengths_tilde = mask, lengths

        x = self.word_dropout(x)
        feature = self.lstm(x)
        scores = self.tag_projection(feature)
        out = self.decode(scores, mask, labels, lengths)
        out_tilde = self.decode(scores, mask_tilde, l_tilde, lengths_tilde)
        loss = out['loss'] * lam + out_tilde['loss'] * (1 - lam)
        return dict(loss=loss)

    def before_train_once(self, kwargs):
        super().before_train_once(kwargs)
        if self.epoch > self.timing and self.pretraining:
            self.pretraining = False
            trainer = kwargs['self']
            if self.mode == 1:
                trainer.dataset.train = trainer.dataset.mixup
                output(f"Epoch {self.epoch} switch to Ann mix.")
            elif self.mode == 2:
                trainer.dataset.train = trainer.dataset.ann
                trainer.batch_size = 1
                output(f"Epoch {self.epoch} switch to Text mix.")
            elif self.mode == 3:
                trainer.dataset.train = trainer.dataset.raw
                output(f"Epoch {self.epoch} switch to All mix.")
            elif self.mode == 4:
                trainer.dataset.train = trainer.dataset.raw
                output(f"Epoch {self.epoch} switch to mixup.")
            else:
                self.pretraining = True

    def before_time_start(self, _, trainer, kwargs):
        trainer.load()


class LSTMCrowd(AdapterModel):
    def __init__(self,
                 vocab: Vocabulary,
                 annotator_num: int = 70,
                 cat=False,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        self.cat = cat
        if cat:
            self.tag_projection = nn.Linear(self.lstm.output_dim * 2, 6)
            self.woker_matrix = nn.Embedding(
                annotator_num, self.lstm.output_dim,
                _weight=torch.zeros(annotator_num, self.lstm.output_dim))
        else:
            self.woker_matrix = nn.Embedding(
                annotator_num, 6, _weight=torch.zeros(annotator_num, 6))

    def forward(
        self, words: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
        user: torch.Tensor = None, labels: torch.Tensor = None, **kwargs
    ) -> Dict[str, Any]:
        embedding = self.word_embedding(words, mask=mask, **kwargs)
        embedding = self.word_dropout(embedding)
        feature = self.lstm(embedding, lengths)

        if self.training or self.self_eval:
            vector = self.woker_matrix(user).unsqueeze(1).expand(-1, words.size(1), -1)
        elif hasattr(self, 'ann_id'):
            vector = self.woker_matrix.weight[self.ann_id]
            vector = vector.unsqueeze(0).unsqueeze(0).expand(words.size(0), words.size(1), -1)
        else:
            vector = torch.zeros_like(feature)

        if self.cat:
            feature = torch.cat([feature, vector], dim=-1)

        scores = self.tag_projection(feature)
        if not self.cat and (self.training or hasattr(self, 'ann_id')):
            scores += vector

        output_dict = self.decode(scores, mask, labels, lengths)
        return output_dict


class Finetune(Tagger):
    def __init__(self,
                 vocab: Vocabulary,
                 **kwargs):
        super().__init__(vocab, lstm_size=0, **kwargs)
        for param in self.word_embedding.parameters():
            param.requires_grad = True

    def drop_param(_, name: str):
        return False
