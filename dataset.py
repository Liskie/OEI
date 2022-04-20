import json
import random
from typing import Dict, Any
from itertools import combinations, chain
from collections import defaultdict, Counter

from nmnlp.common import DataSet


def json_load(path):
    file = open(path, mode='r', encoding='UTF-8')
    try:
        data = json.load(file)
    except json.decoder.JSONDecodeError:
        file.seek(0)
        data = [json.loads(line) for line in file]
    finally:
        file.close()
    print("--- read {} lines to ".format(len(data)) + path)
    return data


class OpinionDataset(DataSet):
    index_fields = ('words', 'labels')

    def collate_fn(self, batch) -> Dict[str, Any]:
        if hasattr(self, 'batch'):
            batch = batch[0]
        return super().collate_fn(batch)

    @classmethod
    def build(cls, data_dir, crowd=False, voting=False, seg=False) -> Dict[str, 'OpinionDataset']:
        train_data = cls.read_crowd(data_dir + "/train.json", crowd, voting, seg)
        dev_data, test_data = cls.read_expert(data_dir + "/dev.json"), cls.read_expert(data_dir + "/test.json")
        return dict(train=cls(train_data), dev=cls(dev_data), test=cls(test_data))

    @classmethod
    def new_crowd(cls, data_dir, name: str = "test-crowd"):
        return cls(cls.read_crowd(f"{data_dir}/{name}.json", True))

    @staticmethod
    def read_expert(path):
        # print('\n\n')
        raw, data = json_load(path), list()
        for r in raw:
            tags = ['O'] * len(r['text'])
            for ann in r['annotations']:
                start, end, label = ann['start_offset'], ann['end_offset'], ann['label']
                if start == end == -1:
                    continue
                if tags[start] != 'O':
                    print(1)
                tags[start] = 'B-' + label
                for i in range(start + 1, end):
                    if tags[i] != 'O':
                        print(1)
                    tags[i] = 'I-' + ann['label']
            data.append(_row_to_dict(r, tags))
        return data

    @staticmethod
    def read_crowd(path, crowd=False, voting=False, seg=False):
        train_raw, train_data = json_load(path), list()
        for row in train_raw:
            tags = None
            if crowd:
                ins = row_to_instances(row)
                train_data.extend(ins)
                continue
            elif voting:
                tags = vote_seg(row) if seg else vote_to_one(row)
            else:
                if len(row['bestUsers']) == 0:  # silver
                    users = set()
                    for ann in row['annotations']:
                        if ann['start_offset'] == ann['end_offset'] == -1:
                            continue
                        users.add(ann['user'])
                    best_user = list(users)[random.randint(0, len(users) - 1)]
                else:
                    best_user = row['bestUsers'][0]
                tags = tags_by_user(row, best_user)
            ins = _row_to_dict(row, tags)
            train_data.append(ins)
        # print('\n\n')
        return train_data


def row_to_instances(row):
    users = set(ann['user'] for ann in row['annotations'])
    user_tags = {u: tags_by_user(row, u) for u in users}
    data = [_row_to_dict(row, v, k) for k, v in user_tags.items()]
    return data


def tags_by_user(row, user):
    tags = ['O'] * len(row['text'])
    for ann in row['annotations']:
        if ann['user'] != user:
            continue
        start, end, label = ann['start_offset'], ann['end_offset'], ann['label']
        if start == end == -1:
            continue
        if tags[start] != 'O':
            raise Exception("标注了重叠的")
        tags[start] = 'B-' + label
        for i in range(start + 1, end):
            if tags[i] != 'O':
                raise Exception("标注了重叠的")
            tags[i] = 'I-' + ann['label']
    return tags


def vote_to_one(row):
    tags = ['O'] * len(row['text'])
    array = [0 for _ in row['text']]
    for ann in row['annotations']:
        value = 1 if ann['label'] == 'POS' else -1
        for i in range(ann['start_offset'], ann['end_offset']):
            array[i] += value
    label = None
    for i, a in enumerate(array):
        if abs(a) >= 2:
            if label is None:
                label = "POS" if a > 0 else "NEG"
                tags[i] = 'B-' + label
            elif (a > 0 and array[i - 1] > 0) or (a < 0 and array[i - 1] < 0):
                tags[i] = 'I-' + label
            else:
                label = None
        else:
            label = None

    return tags


def vote_seg(row):
    tags = ['O'] * len(row['text'])
    array = [0 for _ in row['text']]
    seg = [0 for _ in row['text']]
    for ann in row['annotations']:
        value = 1 if ann['label'] == 'POS' else -1
        for i in range(ann['start_offset'], ann['end_offset']):
            array[i] += value
            seg[i] += 1

    j = None
    for i, s in enumerate(seg + [0]):
        if s >= 2:
            if j is None:
                j = i
            else:
                continue
        elif j is not None:
            clazz = sum(array[j: i - 1])
            label = 'POS' if clazz > 0 else 'NEG'
            tags[j] = 'B-' + label
            for k in range(j + 1, i - 1):
                tags[k] = 'I-' + label
            j = None
        else:
            continue

    return tags


def _row_to_dict(row, tags, user=None):
    text, words = list(row['text']), list(row['text'])
    # tags = ['O'] + tags + ['O']
    # text = ['[CLS]'] + text + ['[SEP]']
    # words = ['[CLS]'] + words + ['[SEP]']
    ins = {
        'text': text,
        'words': words,
        'labels': tags,
        'id': row['id'],
        'annotations': row['annotations']
    }
    if user is not None:
        ins.update(user=user)
    return ins


def group_by_ins(data):
    def merge_instances(array):
        labels = [i['labels'] for i in array]
        users = [i['user'] for i in array]
        array[0].update(labels=labels, users=users)
        return array[0]

    id_to_data = defaultdict(list)
    for ins in data:
        id_to_data[ins['id']].append(ins)
    id_to_data = {k: merge_instances(v) for k, v in id_to_data.items()}
    return list(id_to_data.values())


def group_by_ann(data, batch_size=64):
    grouped = list()
    a2i = defaultdict(list)

    def add_ins(ins, u):
        if len(a2i[u]) >= batch_size:
            grouped.append(a2i[u])
            a2i[u] = list()
        a2i[u].append(ins)

    for ins in data:
        add_ins(ins, ins['user'])

    for v in a2i.values():
        grouped.append(v)

    return grouped


def group_data(dataset, by_ins=False):
    if by_ins:
        dataset.data = group_by_ins(dataset.data)
        return

    dataset.data = group_by_ann(dataset.data)
    setattr(dataset, 'batch', True)


def prob(p):
    return random.random(0, 1) > p


def group_mixup(dataset):
    by_ins, mixup = group_by_ins(dataset.data), list()
    for ins in by_ins:
        for (i, ui), (j, uj) in combinations(enumerate(ins['users']), 2):
            mixup.append(dict(
                words=ins['words'], user=ui, u_tilde=uj,
                labels=ins['labels'][i], l_tilde=ins['labels'][j]))

    dataset.data = mixup
    dataset.vec_fields = ['words', 'labels', 'l_tilde']
    dataset.int_fields = ['user', 'u_tilde']


def main():
    from stats import bio_to_span

    dataset = OpinionDataset.build('data/', False)
    for k in ('dev', 'test', 'train'):
        data = dataset[k].data
        kind, ki, span_len = dict(POS=0, NEG=0), dict(POS=0, NEG=0), dict(POS=list(), NEG=list())
        for ins in data:
            for i, l in enumerate(ins['labels']):
                if l.startswith("B-"):
                    ki[l[2:]] += 1
            spans = bio_to_span(ins['labels'])
            for s in spans:
                kind[s[2]] += 1
                span_len[s[2]].append(s[1] - s[0] + 1)
        print(k, " span kind num: ", kind)
        print(k, " span kind num: ", ki)
        c = Counter(chain(*span_len.values()))
        print(k, " span len: ", c)
        num = sum(kind.values())
        tl = sum(k * v for k, v in c.items())
        print(k, " avg span len: ", tl / num)
    return


if __name__ == "__main__":
    main()
