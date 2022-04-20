"""
"""

import os
import csv
import copy
import json
from typing import Dict, List, Any
from collections import defaultdict
import random

import torch
from torch.utils.data import DataLoader

from nmnlp.common import DataSet
from nmnlp.common.util import output, to_device

random.seed(123)
USER_NUM = 70


def compute_user_scores(model, dataset: DataSet, device, norm=False) -> torch.Tensor:
    if model is None:
        scores = torch.ones(USER_NUM)
        scores[-1] = 0
        return scores
    model.eval()

    three = (copy.deepcopy(model.metric), copy.deepcopy(model.prop), copy.deepcopy(model.binary))
    for m in three:
        m.get_metric(reset=True)

    metrics = [dict(
            exact=copy.deepcopy(three[0]),
            prop=copy.deepcopy(three[1]),
            binary=copy.deepcopy(three[2])
        ) for _ in range(USER_NUM)]
    loader = DataLoader(dataset, 64, collate_fn=dataset.collate_fn)

    for i, (input_dict, batch) in enumerate(loader):
        input_dict = to_device(input_dict, device)
        prediction = model(**input_dict)['predicted_tags']
        # for ins in batch:
        for ins, tags in zip(batch, prediction):
            # index = ins['users'].index(ins['best_users'][0])  # 还有不在users里的？？？？？？？？？？？
            # tags = torch.tensor([ins['labels'][index]])
            tags = torch.tensor([tags])
            for user, ann in zip(ins['users'], ins['labels']):
                len_ann = [len(ann)]
                ann = torch.tensor([ann])
                for k in ('exact', 'binary', 'prop'):
                    metrics[user][k](ann, tags, len_ann)

    user_scores = torch.zeros(USER_NUM, 3)
    for u in range(USER_NUM):
        for i, k in enumerate(('exact', 'binary', 'prop')):
            m = metrics[u][k].get_metric(reset=True)
            user_scores[u, i] = m['main_F1']
    expect = user_scores.mean(dim=-1)
    if norm:
        expect = norm_to_a_b(expect, -1.0, 1.0)
    return expect


def norm_to_a_b(data: torch.Tensor, a=0.0, b=1.0) -> torch.Tensor:
    _max, _min = data.max(), data.min()
    r = (b - a) / (_max - _min)
    norm = (data - _min) * r + a
    return norm


def inference_and_vote(model: torch.nn.Module, dataset: DataSet, device) -> Dict[str, Any]:
    model.eval()
    ins_to_labels = defaultdict(dict)
    loader = DataLoader(dataset, 64, collate_fn=dataset.collate_fn)

    # 先预测
    for ann_id in range(70):
        model.set_adapter_parameter(model.annotator_embedding.weight[ann_id])
        for i, (input_dict, batch) in enumerate(loader):
            input_dict = to_device(input_dict, device)
            predictions = model(**input_dict)['predicted_tags']
            for ins, tags in zip(batch, predictions):
                ins_to_labels[ins['id']][ann_id] = tags

    # 把人工标注的补回去
    for ins in dataset.data:
        for ann_id, tags in zip(ins['users'], ins['labels']):
            ins_to_labels[ins['id']][ann_id] = tags

    voted = {k: vote_all(v) for k, v in ins_to_labels.items()}
    voted = {k: dict(labels=v) for k, v in voted.items() if v}
    return voted


def vote_all(ann_tag: Dict[int, List], thresh=0.6):
    B_POS, I_POS, B_NEG, I_NEG = 2, 3, 4, 5
    thresh *= len(ann_tag)
    tags = [1 for _ in ann_tag[1]]
    array = [0 for _ in ann_tag[1]]

    for k, v in ann_tag.items():
        for i, t in enumerate(v):
            if t in (B_POS, I_POS):
                array[i] += 1
            elif t in (B_NEG, I_NEG):
                array[i] += -1

    label = None
    for i, a in enumerate(array):
        if abs(a) >= thresh:
            if label is None:
                label = B_POS if a > 0 else B_NEG
                tags[i] = label
            elif a ^ array[i - 1] >= 0:  # 同号
                tags[i] = label + 1
            else:
                label = None
        else:
            label = None

    if len(set(tags)) == 1:
        return None

    return tags


def mixup_data(_, dataset: DataSet, *args) -> Dict[str, Any]:
    data = {
        ins['id']: dict(anns=ins['labels'], users=ins['users'])
        for ins in dataset.data
    }
    return data


def save_metrics(metrics, path='dev/metrics.csv'):
    head = ['ann_id', 'main_F1']
    rows = [head] + [[i, m['main_F1']] for i, m in enumerate(metrics)]
    with open(path, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    output(f"saved at <{path}>")


def dump_json_line(path: str, data: List):
    with open(path, mode='w', encoding='utf8') as file:
        for i in data:
            line = json.dumps(i, ensure_ascii=False) + "\n"
            file.write(line)
    print("--- write {} lines to ".format(len(data)) + path)


def read_json_line(path: str) -> List[Dict]:
    data = list()
    with open(path, mode='r', encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line))
    print("--- read {} lines to ".format(len(data)) + path)
    return data


def data_split(path: str, num: int = 10):
    prefix = path.split('.')[0]
    data = read_json_line(path)
    num = len(data) // num
    parts: List[List] = list()
    while len(data) > num:
        parts.append(data[:num])
        data = data[num:]
    parts[-1].extend(data)
    os.mkdir(prefix)
    for i, part in enumerate(parts):
        dump_json_line("{}/{}.json".format(prefix, i), part)
    return


def replace(spans, tid, uid, ins, counter):
    def confilct_ann(a, b):
        if a['user'] != b['user']:
            return False
        if a['end_offset'] <= b['start_offset']:  # a.end < b.start, a is in front of b
            return False
        if b['end_offset'] <= a['start_offset']:  # b.end < a.start, b is in front of a
            return False
        return True  # they are overlaped

    if uid in ins['bestUsers']:
        return

    if tid == 13615 and uid == 69:
        print(1)

    for start, end, label in spans:
        if random.uniform(0, 1) < 0.3:
            ann = dict(label=label, start_offset=start, end_offset=end, user=uid, r=2)
            annotations = [a for a in ins['annotations'] if not confilct_ann(ann, a)]
            annotations.append(ann)
            ins['annotations'] = annotations
            counter[uid] += 1
    return


def refine_crowd():
    # with open('data/test-crowd.json', mode='r') as file:
    #     test_crowd = json.load(file)
    test_crowd = read_json_line("data/test-crowd-r1.json")

    refined = {i['id']: i for i in test_crowd}
    assert len(refined) == len(test_crowd)
    counter = defaultdict(int)

    predictions = read_json_line('dev/out/self-test-6.json')

    from stats import bio_to_span

    for pred in predictions:
        try:
            spans = bio_to_span(pred['prediction'])
            spans = list((s[0], s[1] + 1, s[2]) for s in spans)  # from inclusive to exclusive end offset
            replace(spans, pred['id'], pred['user'], refined[pred['id']], counter)
        except Exception:
            print(0)

    print("replaced {} annotations".format(sum(counter.values())))
    dump_json_line('data/test-crowd-r11.json', refined.values())

    return


def main():
    refine_crowd()


if __name__ == "__main__":
    main()
