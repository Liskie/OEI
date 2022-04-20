"""
"""

import glob
import json
from typing import List
from difflib import SequenceMatcher
from itertools import chain, combinations
from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from metric import ExactMatch, Proportional, Binary
from constant import REVIEWER_MAP, TEST_ID, DEV_ID, LABEL_MAP, USER_TO_ID


def json_load(path):
    with open(path, mode='r', encoding='UTF-8') as file:
        return json.load(file)


def write_formate(name, data):
    with open(f'dev/data/{name}.json', mode='w', encoding='UTF-8') as file:
        file.write('[')
        lines = ",\n".join(json.dumps(ins, ensure_ascii=False) for ins in data)
        file.write(lines)
        file.write(']\n')
    print(f'dev/data/{name}.json')


def read_lines(path, sep="\t"):
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
                sentence.append([cols[0]] + [c.strip() for c in cols[1:]])


def bio_to_span(seq: List[str]):
    spans = list()
    one = None  # [i, j, label]  如果切片要+1
    for i, t in enumerate(chain(seq, ['O'])):
        if t.startswith("B-"):
            if one:
                spans.append(tuple(one))
            one = [i, i, t[2:]]
        elif t.startswith("I-"):
            if one is None:
                raise Exception("有I前面无B")
            elif one[2] != t[2:]:
                raise Exception("BI不一致")
            else:
                one[1] = i
        elif one:
            spans.append(tuple(one))
            one = None

    return spans


def clean_ins(one, reviewer, select=False):
    for ann in one['annotations']:
        ann['user'] = USER_TO_ID[ann['user']]
        label = LABEL_MAP[ann['label']]
        ann['label'] = "POS" if label == "正面" else "NEG"

    if one['bestUsers'] == '-1':
        best = []
    else:
        best = [USER_TO_ID[int(u)] for u in one['bestUsers'].split('-')]

    if select:
        assert len(best) > 0
        annotations = []
        for ann in one['annotations']:
            if ann['user'] == best[0]:
                ann.pop('user')
                annotations.append(ann)
    else:
        annotations = one['annotations']

    ins = dict(id=one['id'], text=one['text'], annotations=annotations, reviewer=reviewer)
    if not select:
        ins.update(bestUsers=best)
    return ins


def crowd_test():
    crowd = list()
    path_list = glob.glob("dev/data/task1/*.json")
    with open('data/dev.json', mode='r', encoding='UTF-8') as file:
        testset = json.load(file)
    test_ids = {v: list() for v in REVIEWER_MAP.values()}
    for ins in testset:
        test_ids[ins['reviewer']].append(ins['id'])

    for path in path_list:
        reviewer = REVIEWER_MAP[path.split('/')[-1]]
        with open(path, mode='r', encoding='UTF-8') as file:
            data = json.load(file)
            for row in data:
                if row['id'] in test_ids[reviewer]:
                    ins = clean_ins(row, reviewer)
                    crowd.append(ins)

    write_formate('dev-crowd', crowd)
    return


def clean_data():
    trainset, devset, testset, matched = list(), list(), list(), set()
    path_list = glob.glob("dev/data/task1/*.json")
    test_data = read_lines('dev/data/task1/test.txt')
    test_data = [one for one in test_data if len(bio_to_span([i[1] for i in one])) > 0]
    abandon = list()
    for path in path_list:
        reviewer = REVIEWER_MAP[path.split('/')[-1]]
        with open(path, mode='r', encoding='UTF-8') as file:
            data = json.load(file)
            for row in data:
                if row['id'] in TEST_ID:
                    # if '中科院科学家率先揭示武汉新型冠状病毒的进化来源' in row['text']:
                    #     print(0)
                    ins = clean_ins(row, reviewer, True)
                    for i, one in enumerate(test_data):
                        if i in matched:
                            continue
                        text = ''.join([i[0] for i in one])
                        sim = SequenceMatcher(None, text, ins['text']).quick_ratio()
                        if sim > 0.99:
                            matched.add(i)
                            break
                    else:
                        abandon.append(ins)
                        continue
                    if len(text) != len(ins['text']):
                        print(ins['id'])
                        print(text)
                        print(ins['text'])
                        print('\n')
                    gold_spans = bio_to_span([i[1] for i in one])
                    if len(gold_spans) < 1:
                        continue
                    select_spans = [(
                        a['start_offset'], a['end_offset'] - 1,
                        "正面" if a['label'] == "POS" else "负面")
                        for a in ins['annotations']]
                    _, prop, _ = compute_correct(gold_spans, select_spans)
                    # if prop / len(gold_spans) < 0.2:
                    #     print(text)
                    #     print(gold_spans)
                    #     print(select_spans)
                    #     print('\n')
                    annotations = []
                    for span in gold_spans:
                        label = "POS" if span[-1] == "正面" else "NEG"
                        annotations.append(dict(
                            label=label, start_offset=span[0], end_offset=span[1] + 1
                        ))
                    ins.update(text=text, annotations=annotations)
                    testset.append(ins)
                elif row['id'] in DEV_ID:
                    ins = clean_ins(row, reviewer, True)
                    devset.append(ins)
                else:
                    ins = clean_ins(row, reviewer)
                    trainset.append(ins)

    unmatch = [a for a in range(len(test_data)) if a not in matched]
    for index in unmatch:
        print(''.join(i[0] for i in test_data[index]))
        print(bio_to_span([i[1] for i in test_data[index]]))
    # 4 7598 '中科院科学家率先揭示武汉新型冠状病毒的进化来源，以及传染人的分子作用机制和传染性风险丨科学大发现'
    # [(6, 7, '正面'), (45, 47, '正面')]
    # {"id": 7598, "text": "中科院科学家率先揭示武汉新型冠状病毒的进化来源，以及传染人的分子作用机制和传染性风险丨科学大发现 ​​​",
    #  "annotations": [{"label": "POS", "start_offset": 6, "end_offset": 8},
    #                  {"label": "POS", "start_offset": 45, "end_offset": 48}],
    #  "reviewer": 4},
    # 1 6167  '我还是想回武汉工作，家里医院给了offer，我们这春招肯定是完了。万幸，身边的朋友和我，没有感染上'
    # [(30, 31, '负面'), (33, 34, '正面')]
    # {"id": 6167, "text": "我还是想回武汉工作，家里医院给了offer，我们这春招肯定是完了。万幸，身边的朋友和我，没有感染上。",
    #  "annotations": [{"label": "NEG", "start_offset": 27, "end_offset": 32},
    #                  {"label": "POS", "start_offset": 33, "end_offset": 35}],
    #  "reviewer": 1},

    write_formate('train-', trainset)
    write_formate('dev-', devset)
    write_formate('test-', testset)
    return


def compute_correct(gold, pred):
    gold, pred = set(gold), set(pred)
    e = ExactMatch.get_correct(pred, gold)
    p = Proportional.get_correct(pred, gold)
    b = Binary.get_correct(pred, gold)
    return dict(exact=len(e), prop=sum([i[-1] for i in p]), binary=len(b))


def kappa(ins) -> float:
    user_tags = defaultdict(lambda: ['O'] * len(ins['text']))
    for ann in ins['annotations']:
        if ann['end_offset'] == ann['start_offset'] == -1:
            continue
        for i in range(ann['start_offset'], ann['end_offset']):
            user_tags[ann['user']][i] = ann['label']

    cohen_kappas = list()
    for i, j in combinations(user_tags.keys(), 2):
        ti, tj = user_tags[i], user_tags[j]
        if ti == tj:
            k = 1
        else:
            k = cohen_kappa_score(ti, tj)
        if -1 < k <= 1:
            cohen_kappas.append(k)
        else:
            print('000')

    return sum(cohen_kappas) / len(cohen_kappas) if cohen_kappas else 1


NUM = {'POS': 0, 'NEG': 0, 'l': 0}


def measure_datail(data, span_level=False):
    info = list()
    # [['i', 'span_num', 'kind_num', 'pred_num', 'exact_c', 'prop_c', 'binary_c']]
    # [['i', 'span_num', 'kind_num', 'len', 'pred_len', 'match?']]

    for i, ins in enumerate(data):
        gold_spans = bio_to_span([i[1] for i in ins])
        label_set = set(s[2] for s in gold_spans)
        if len(gold_spans) == 0:
            continue
        else:
            for span in gold_spans:
                NUM[span[2]] += 1
                NUM['l'] += span[1] - span[0] + 1
        row = dict(i=i, span_num=len(gold_spans), kind_num=len(label_set))
        if len(ins[0]) > 2:
            pred_spans = bio_to_span([i[2] for i in ins])
            if span_level:
                for gold in gold_spans:
                    gl = gold[1] - gold[0] + 1
                    for pred in pred_spans:
                        rj = row.copy()
                        match = 1 if pred == gold else 0
                        rj.update(len=gl, pred_len=pred[1] - pred[0] + 1, match=match)
                        info.append(rj)
                else:
                    continue
            else:
                correct = compute_correct(gold_spans, pred_spans)
                row.update(pred_num=len(pred_spans), **correct)
        info.append(row)
    return info


def stat_out(path):
    def score(frame, key='exact'):
        total = frame['span_num'].sum()
        pred = frame['pred_num'].sum()
        correct = frame[key].sum()
        p = correct / pred
        r = correct / total
        f = 2 * p * r / (p + r)
        return p, r, f

    def span_score(frame, length, comp='=='):
        len_frame = frame.query(f"len{comp}{length}")
        total = len(len_frame)
        correct = len_frame['match'].sum()
        pred = len(frame.query(f"pred_len{comp}{length}"))
        p = correct / pred
        r = correct / total
        f = 2 * p * r / (p + r)
        return p, r, f

    data = list(read_lines(path))
    print('\n')
    ins_level = measure_datail(data)
    opinion_counter = Counter([i['span_num'] for i in ins_level])
    kind_counter = Counter([i['kind_num'] for i in ins_level])
    print(NUM)
    print(opinion_counter)
    print(kind_counter)

    ins_frame = pd.DataFrame(ins_level)

    one_opinion = ins_frame.query("span_num==1 & kind_num==1")
    print(len(one_opinion))
    print("ins one opinion exact p r f: ", score(one_opinion))
    print("ins one opinion prop p r f: ", score(one_opinion, 'prop'))
    print("ins one opinion binary p r f: ", score(one_opinion, 'binary'))
    print('\n')
    multi_opinion_one_kind = ins_frame.query("span_num>1 & kind_num==1")
    print(len(multi_opinion_one_kind))
    print("ins multi_opinion_one_kind exact p r f: ", score(multi_opinion_one_kind))
    print("ins multi_opinion_one_kind prop p r f: ", score(multi_opinion_one_kind, 'prop'))
    print("ins multi_opinion_one_kind binary p r f: ", score(multi_opinion_one_kind, 'binary'))
    print('\n')
    multi_opinion_two_kind = ins_frame.query("span_num>1 & kind_num>1")
    print(len(multi_opinion_two_kind))
    print("ins multi_opinion_two_kind exact p r f: ", score(multi_opinion_two_kind))
    print("ins multi_opinion_two_kind prop p r f: ", score(multi_opinion_two_kind, 'prop'))
    print("ins multi_opinion_two_kind binary p r f: ", score(multi_opinion_two_kind, 'binary'))
    print('\n')

    span_level = measure_datail(data, True)
    span_frame = pd.DataFrame(span_level)

    for k in ("exact",):  # , "prop", "binary"
        print("\n", k)
        print(f"(1-2, {span_score(span_frame, 2, '<=')[-1] * 100:.2f})")
        for i in range(3, 8):
            print(f"({i}, {span_score(span_frame, i)[-1] * 100:.2f})")
        else:
            print(f"(8+, {span_score(span_frame, 8, '>=')[-1] * 100:.2f})")

    return span_level


def data_stat():
    total, kappas = 0, list()
    for k in ('dev-crowd', 'test-crowd', 'train'):
        data = json_load(f'data/{k}.json')
        for ins in data:
            for ann in ins['annotations']:
                if ann['end_offset'] == ann['start_offset'] == -1:
                    continue
                total += 1
            kappas.append(kappa(ins))

    print('total span: ', total)
    print('cohen kappa: ', sum(kappas) / len(kappas))

    for k in ('dev', 'test', 'train'):
        data = json_load(f'data/{k}.json')
        kind, span_len = dict(POS=0, NEG=0), dict(POS=list(), NEG=list())
        for ins in data:
            for ann in ins['annotations']:
                if ann['end_offset'] == ann['start_offset'] == -1:
                    continue
                length = ann['end_offset'] - ann['start_offset']
                span_len[ann['label']].append(length)
                kind[ann['label']] += 1
        print(k, " span kind num: ", kind)
        c = Counter(chain(*span_len.values()))
        print(k, " span len: ", c)
        num = sum(kind.values())
        tl = sum(k * v for k, v in c.items())
        print(k, " avg span len: ", tl / num)
    return


def from_log(path):
    with open(path, mode='r') as file:
        lines, one = list(), list()
        for line in file:
            line = line.strip()
            if line:
                one.append(line)
            elif one:
                lines.append(one)
                one = list()

    for rows in lines:
        if 'ann: ' not in rows[0]:
            print('\n')
            continue
        # numbers = list()
        numbers = [
            rows[0].split(': ')[-2].split(', ')[0],
            rows[0].split(': ')[-1],
            rows[1].split('0.')[-1],
            rows[2].split('0.')[-1],
            rows[3].split('0.')[-1],
            rows[5].split(': ')[-1],
            rows[6].split('0.')[-1],
            rows[7].split('0.')[-1],
            rows[8].split('0.')[-1]]
        for i in (2, 3, 4, 6, 7, 8):
            numbers[i] = numbers[i][:2] + '.' + numbers[i][2:]
        print(', '.join(numbers))  # ' & \\\\'

    return lines


def main():
    # clean_data()
    # with open('dev/data/train.json', mode='r', encoding='utf8') as file:
    #     train = json.load(file)
    data_stat()
    # crowd_test()
    from_log('dev/ada-mix-1-tt.log')
    siler = stat_out('dev/out/ada_123/test-0.txt')
    our = stat_out('dev/out/ada-pg_123/test-0.txt')
    mix = stat_out('dev/out/ada-mix_123/test-0.txt')
    mv = stat_out('dev/out/ada-vote_123/test-0.txt')
    lstm = stat_out('dev/out/lstm-crowd_123/test-0.txt')
    return siler, our, mix, mv, lstm


if __name__ == "__main__":
    main()
