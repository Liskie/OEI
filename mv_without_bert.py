import json


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


from seqeval.metrics import f1_score, recall_score, precision_score


def get_sliver_dict(data):
    bio_dict = {}
    for item in data:
        if item['bestUsers']:
            best_user = item['bestUsers'][0]
        else:
            best_user = item['annotations'][0]['user']
        bio_dict[item['id']] = []
        tags = ['O'] * len(item['text'])
        for ann in item['annotations']:
            if ann['user'] == best_user:
                for j in range(len(item['text'])):
                    if j == ann['start_offset']:
                        tags[j] = "B-" + str(ann['label'])
                    elif ann['start_offset'] < j < ann['end_offset']:
                        tags[j] = "I-" + str(ann['label'])
                    else:
                        continue
        bio_dict[item['id']] = tags
    return bio_dict


def get_f1_with_dict(worker_data, silver_dict):
    y_true = []
    y_pred = []
    for pred_item in worker_data:
        y_pred.append(pred_item['tags'])
        y_true.append(silver_dict[pred_item['id']])
    return f1_score(y_true, y_pred, average='macro')


def get_recall_with_dict(worker_data, silver_dict):
    y_true = []
    y_pred = []
    for pred_item in worker_data:
        y_pred.append(pred_item['tags'])
        y_true.append(silver_dict[pred_item['id']])
    return recall_score(y_true, y_pred, average='macro')


def get_precision_with_dict(worker_data, silver_dict):
    y_true = []
    y_pred = []
    for pred_item in worker_data:
        y_pred.append(pred_item['tags'])
        y_true.append(silver_dict[pred_item['id']])
    return precision_score(y_true, y_pred, average='macro')


if __name__ == '__main__':
    with open('data/train.json', 'r') as data_file:
        data = json.load(data_file)

    worker_data = []
    for line in data:
        worker_data.append({
            'id': line['id'],
            'tags': vote_to_one(line)
        })

    silver_dict = get_sliver_dict(data)

    print(f'{get_precision_with_dict(worker_data, silver_dict): .02%}')
    print(f'{get_recall_with_dict(worker_data, silver_dict): .02%}')
    print(f'{get_f1_with_dict(worker_data, silver_dict): .02%}')

    # Calculate average annotation lengths
    mv_tags = [line['tags'] for line in worker_data]
    mv_annotation_lengths = {
        'O': 0,
        'POS': 0,
        'NEG': 0,
    }
    for tags in mv_tags:
        for tag in tags:
            if tag == 'O':
                mv_annotation_lengths['O'] += 1
            elif tag.endswith('POS'):
                mv_annotation_lengths['POS'] += 1
            elif tag.endswith('NEG'):
                mv_annotation_lengths['NEG'] += 1
    mv_annotation_lengths = {k: v / len(mv_tags) for k, v in mv_annotation_lengths.items()}
    print(mv_annotation_lengths)
