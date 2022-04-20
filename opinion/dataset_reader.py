from typing import Dict, Iterable, List, Optional, Union
from itertools import chain

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer


def read_lines(path, sep=None):
    with open(path, mode='r', encoding='UTF-8') as file:
        sentence = list()
        for line in chain(file, [""]):
            cols = line.split(sep)
            if len(cols) < 2:
                if sentence:
                    yield sentence
                    sentence = list()
                else:
                    continue
            else:
                sentence.append([c.strip() for c in cols])


@DatasetReader.register("mpqa2_exp")
class MPAQ2Reader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        max_span_width: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers
        self.max_span_width = max_span_width

    def text_to_instance(self, text: Union[List[List[str]], Dict]) -> Instance:  # type: ignore
        words = [row[1] for row in text]
        labels, i, length = list(), 0, len(words)
        while i < length:
            parts = text[i][2].split('-')  # B-label -> B, label
            if parts[0] == 'O' or 'SE' not in parts[1]:
                labels.append('O')
                i += 1
                continue
            start, end, label, polarity = None, None, None, None
            while parts[0] != 'O':
                if parts[0] in ('M', 'E'):
                    assert start is not None, "ME should have a B"
                    assert label == parts[1], "ME should have the same label with B"
                    end += 1
                    if polarity == 'O':  # sentiment
                        polarity = parts[3]
                else:
                    assert start is None, "start should not have value when BOS"
                    if parts[0] in ('B', 'S'):
                        start, end, label, polarity = i, i + 1, parts[1], text[i][3]  # 直接切片
                    else:
                        raise Exception('Wrong tag : ', parts)
                i += 1
                if parts[0] in ('E', 'S'):
                    if polarity == 'positive':
                        label = "POS"
                    elif polarity == 'negative':
                        label = "NEG"
                    else:
                        raise Exception('Expression have no polarity.')
                    labels.append("B-" + label)
                    for _ in range(start + 1, end):
                        labels.append("I-" + label)
                    break
                parts = text[i][2].split('-')

        if len(set(labels)) <= 1:
            return None

        text_field = TextField([Token(t) for t in words], self.token_indexers)
        label_field = SequenceLabelField(labels, text_field)
        metadata_field = MetadataField(dict(words=words))
        fields = dict(tokens=text_field, tags=label_field, metadata=metadata_field)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        if file_path.endswith('.txt'):
            for s in ('dse', 'ese'):
                path = file_path.format(s)
                for lines in read_lines(path):
                    # if polarities are not all `O`, try to yield an instance.
                    if len(set([i[3] for i in lines])) > 1:
                        ins = self.text_to_instance(lines)
                        if ins is not None:
                            yield ins
