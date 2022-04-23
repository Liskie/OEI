"""
"""

import os
import copy
import argparse

import wandb

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y', type=str, default='ada-pg', help='configuration file path.')
_ARG_PARSER.add_argument('--cuda', '-c', type=str, default='0', help='gpu ids, like: 1,2,3')
# _ARG_PARSER.add_argument('--test', '-t', type=bool, default=False, help='进行测试输出')
_ARG_PARSER.add_argument('--test', '-t', action='store_true', help='进行测试输出')
_ARG_PARSER.add_argument('--name', '-n', type=str, default=None, help='save name.')
_ARG_PARSER.add_argument('--seed', '-s', type=int, default=123, help='random seed')
# _ARG_PARSER.add_argument('--all', '-a', type=bool, default=False, help='all seed?')
_ARG_PARSER.add_argument('--all', '-a', action='store_true', help='all seed?')
# _ARG_PARSER.add_argument('--debug', '-d', default=False, action="store_true")
_ARG_PARSER.add_argument('--debug', '-d', action="store_true")
# _ARG_PARSER.add_argument('--em', type=bool, default=False)
_ARG_PARSER.add_argument('--em', action='store_true')
_ARG_PARSER.add_argument('--timing', type=int, default=None)
_ARG_PARSER.add_argument('--mode', type=int, default=None)
# _ARG_PARSER.add_argument('--case', type=bool, default=False, help='case study')
_ARG_PARSER.add_argument('--case', action='store_true', help='case study')
# _ARG_PARSER.add_argument('--self', type=bool, default=True, help='crowd annotators self eval')
_ARG_PARSER.add_argument('--self', action='store_true', help='crowd annotators self eval')

_ARG_PARSER.add_argument('--adapter_size', type=int, default=None)
_ARG_PARSER.add_argument('--lstm_size', type=int, default=None)
_ARG_PARSER.add_argument('--num_adapters', type=int, default=None)
# _ARG_PARSER.add_argument('--wandb_name', type=str, required=True)
_ARGS = _ARG_PARSER.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = _ARGS.cuda

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

if _ARGS:
    import random

    import numpy as np
    import torch

    from transformers import BertTokenizer

    import nmnlp
    from nmnlp.common.config import load_yaml
    from nmnlp.common.util import output, merge_dicts
    # from nmnlp.common.util import cache_path, load_cache, dump_cache
    from nmnlp.common.writer import Writer
    from nmnlp.common import Trainer, Vocabulary
    from nmnlp.common.optim import build_optimizer

    from em import em_once
    # from util import save_metrics
    from dataset import OpinionDataset, group_data, group_mixup
    from models import build_model  # , PGAdapterModel, LSTMCrowd
else:
    raise Exception('Argument error.')

SEEDS = (123, 456, 789, 686, 666, 233, 1024, 2080, 3080, 3090)
SEEDS = SEEDS[5:]

nmnlp.common.trainer.EARLY_STOP_THRESHOLD = 5


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)  # CPU占用过高，但训练速度没快，还没找到问题所在


def case_study(pg, vocab, device):
    all = argparse.Namespace(**load_yaml("./dev/config/ada-all.yml"))
    mv = argparse.Namespace(**load_yaml("./dev/config/ada-vote.yml"))
    silver = argparse.Namespace(**load_yaml("./dev/config/ada.yml"))
    mix = argparse.Namespace(**load_yaml("./dev/config/ada-mix.yml"))
    lstm = argparse.Namespace(**load_yaml("./dev/config/lstm-crowd.yml"))
    cfgs = dict(silver=silver, vanila=pg, annmix=mix, lstm_c=lstm, all___=all, mv____=mv)
    for v in cfgs.values():
        v.model['allowed'] = allowed_transition(vocab)

    models = dict()
    for k, v in cfgs.items():
        models[k] = build_model(vocab=vocab, **v.model)
        models[k].load(v.trainer['pre_train_path'], device)
        models[k].to(device)
        models[k].eval()
        print(f"loaded <{v.trainer['pre_train_path']}>")

    def text_to_batch(text):
        input_ids = vocab.indices_of(list(text), 'words')
        input_ids = [101] + input_ids + [102]
        words = torch.tensor(input_ids, device=device).unsqueeze(0)
        mask = torch.ones_like(words)
        # lengths = torch.tensor([words.size(1)], device=device).unsqueeze(0)
        return dict(words=words, mask=mask, lengths=None)

    def format(tags, text):
        string = ""
        for la, t in zip(tags, text):
            if la in (4, 5):
                ft = f"\033[34m{t}\033[0m"
            elif la in (2, 3):
                ft = f"\033[31m{t}\033[0m"
            else:
                ft = t
            string += ft
        return string

    with torch.no_grad():
        text = input("\n测试句子: ")
        while text.strip():
            batch = text_to_batch(text)
            for k, m in models.items():
                pred = m.forward(**batch)['predicted_tags']
                print(k, ": ", format(pred[0][1:-1], text))
            else:
                text = input("\n测试句子: ")

    return models


def run_once(cfg, dataset, vocab, device, writer=None, seed=123):
    model = build_model(vocab=vocab, **cfg.model)
    setattr(model, 'seed', seed)
    para_num = sum([p.numel() for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    optimizer = build_optimizer(model, **cfg.optim)
    trainer = Trainer(vars(cfg), dataset, vocab, model, optimizer, None, None,
                      writer, device, **cfg.trainer)

    if _ARGS.yaml in ('pg-ada', 'ada-pg-batch', 'ada-pgo'):
        group_data(dataset.train, by_ins=False)
    elif 'mix' in cfg.model['name']:
        mixup, ann = copy.deepcopy(dataset.train), copy.deepcopy(dataset.train)
        group_mixup(mixup)
        group_data(ann)
        setattr(dataset, "mixup", mixup)
        setattr(dataset, "ann", ann)
        setattr(dataset, "raw", dataset.train)

    if _ARGS.self:
        # Before :
        # dataset.train : train.json
        # dataset.dev   : dev.json
        # dataset.test  : test.json
        dataset.dev = OpinionDataset.new_crowd(cfg.data['data_dir'], 'dev-crowd')
        dataset.dev.index_with(vocab)
        dataset.test_exp = dataset.test
        dataset.test = OpinionDataset.new_crowd(cfg.data['data_dir'], 'test-crowd')
        dataset.test.index_with(vocab)
        for i in range(10):
            name = 'test-crowd-r{}'.format(i + 1)
            test_i = OpinionDataset.new_crowd('data', name)
            test_i.index_with(vocab)
            setattr(dataset, name, test_i)
        # After :
        # dataset.train    : train.json
        # dataset.dev      : dev-crowd.json
        # dataset.test_exp : test.json
        # dataset.test     : test-crowd.json
        # dataset.test-crowd-r{i} : test-crowd-r{i}.json

    if not _ARGS.test:
        # 训练过程
        trainer.train()
        output(model.metric.data_info)

    trainer.load()

    if _ARGS.self:
        for i in range(10):
            name = 'test-crowd-r{}'.format(i + 1)
            output(name)
            test_i = getattr(dataset, name)
            trainer.test(test_i)
            print('\n')

    # if isinstance(model, (PGAdapterModel, LSTMCrowd)):
    #     # metrics = list()
    #     # for i in range(70):
    #     #     setattr(model, 'ann_id', i)
    #     #     output("ann: ", i)
    #     #     metrics.append(trainer.test(dataset.test))
    #     #     print('\n')
    #     # else:
    #     #     save_metrics(metrics)
    #     #     return
    #     # test_metric = trainer.test(dataset.test)
    #     test_crowd = OpinionDataset.new_crowd('data')
    #     # test_crowd.index_with(vocab)
    #     # test_metric = trainer.test(test_crowd)
    #     # print('\n')
    #     for a in range(70):
    #         setattr(model, 'ann_id', a)
    #         train_a = copy.copy(dataset.train)
    #         train_a.data = [i for i in dataset.train.data if i['user'] == a]
    #         output("ann: ", a, ", train num: ", len(train_a))
    #         trainer.test(train_a)
    #         test_a = OpinionDataset([i for i in test_crowd if i['user'] == a])
    #         test_a.index_with(vocab)
    #         output("ann: ", a, ", test num: ", len(test_a))
    #         trainer.test(test_a)
    #         print('\n')
    #     delattr(model, 'ann_id')

    test_metric = trainer.test(dataset.test)
    return model.metric.best, test_metric


def valdata(s1, s2):
    d1, d2 = s1.data, s2.data
    for i1, i2 in zip(d1, d2):
        for k in i1:
            if i1[k] != i2[k]:
                print(i1[k])

    return


def allowed_transition(vocab):
    def idx(token: str) -> int:
        return vocab.index_of(token, 'labels')

    allowed = [
        (idx('O'), idx('O')),
        (idx('O'), idx('B-POS')),
        (idx('O'), idx('B-NEG')),
        (idx('B-POS'), idx('O')),
        (idx('B-POS'), idx('I-POS')),
        (idx('B-POS'), idx('B-NEG')),
        (idx('B-NEG'), idx('O')),
        (idx('B-NEG'), idx('I-NEG')),
        (idx('B-NEG'), idx('B-POS')),
        (idx('I-POS'), idx('O')),
        (idx('I-POS'), idx('I-POS')),
        (idx('I-POS'), idx('B-NEG')),
        (idx('I-NEG'), idx('O')),
        (idx('I-NEG'), idx('I-NEG')),
        (idx('I-NEG'), idx('B-POS')),
    ]
    return allowed


def main():
    cfg = argparse.Namespace(**load_yaml(f"./dev/config/{_ARGS.yaml}.yml"))

    print(f'args:\n{_ARGS}\n')

    device = torch.device("cuda:0")
    data_kwargs, vocab_kwargs = dict(cfg.data), dict(cfg.vocab)
    use_bert = 'bert' in cfg.model['word_embedding']['name_or_path']

    # 如果用了BERT，要加载tokenizer
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(
            cfg.model['word_embedding']['name_or_path'],
            do_lower_case=False)
        print("I'm batman!  ",
              tokenizer.tokenize("I'm batman!"))  # [CLS] [SEP]
        data_kwargs['tokenizer'] = tokenizer
        vocab_kwargs['oov_token'] = tokenizer.unk_token
        vocab_kwargs['padding_token'] = tokenizer.pad_token
    else:
        tokenizer = None

    # cache_name = _ARGS.yaml
    # if not os.path.exists(cache_path(cache_name)):
    # cfg.data['data_dir'] = 'data/'

    dataset = argparse.Namespace(**OpinionDataset.build(**cfg.data))
    # dataset.train : train.json
    # dataset.dev   : dev.json
    # dataset.test  : test.json

    vocab = Vocabulary.from_data(dataset, **vocab_kwargs)
    vocab.set_field(['[PAD]', 'O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG'], 'labels')

    if use_bert:
        # 若用BERT，则把words词表替换为BERT的
        vocab.token_to_index['words'] = tokenizer.vocab
        vocab.index_to_token['words'] = tokenizer.ids_to_tokens
        # dump_cache((dataset, vocab), cache_name)
    # else:
    #     dataset, vocab = load_cache(cache_name)
        # bd, _ = load_cache('task1-bimix')
        # valdata(dataset.train, bd.train)
        # valdata(dataset.dev, bd.dev)
        # valdata(dataset.test, bd.test)

    dataset.train.index_with(vocab)
    dataset.dev.index_with(vocab)
    dataset.test.index_with(vocab)

    cfg.model['allowed'] = allowed_transition(vocab)

    if _ARGS.case:
        set_seed(_ARGS.seed)
        case_study(cfg, vocab, device)
        return

    if _ARGS.debug:
        _ARGS.name = "debug"

    if _ARGS.timing is not None:
        cfg.model['timing'] = _ARGS.timing
    if _ARGS.mode is not None:
        cfg.model['mode'] = _ARGS.mode
        prefix = _ARGS.name if _ARGS.name else f"{_ARGS.yaml}-m{_ARGS.mode}"
    else:
        prefix = _ARGS.name if _ARGS.name else _ARGS.yaml

    if _ARGS.self:
        cfg.model['self_eval'] = True
        prefix += '_self-eval'

    if isinstance(_ARGS.lstm_size, int):
        cfg.model['lstm_size'] = _ARGS.lstm_size
        prefix += f'_l{_ARGS.lstm_size}'
    if isinstance(_ARGS.adapter_size, int):
        cfg.model['adapter_size'] = _ARGS.adapter_size
        prefix += f'_a{_ARGS.adapter_size}'
    if isinstance(_ARGS.num_adapters, int):
        cfg.model['num_adapters'] = _ARGS.num_adapters
        prefix += f'_n{_ARGS.num_adapters}'
    print(cfg.model)

    info = list()
    if _ARGS.debug:
        log_dir = None
        cfg.trainer['save_strategy'] = 'no'
    else:
        # log_dir = f"./dev/tblog/{prefix}"
        # if not os.path.exists(log_dir):
        #     os.mkdir(log_dir)
        log_dir = None

    if _ARGS.em:
        set_seed(_ARGS.seed)
        em_once(cfg, dataset, vocab, device, _ARGS.seed, prefix, log_dir)
        return

    seeds = SEEDS if _ARGS.all else [_ARGS.seed]
    for seed in seeds:
        print('\n')
        print(f'seed = {seed}')
        print()
        set_seed(seed)
        cfg.trainer['prefix'] = f"{prefix}_{seed}"
        wandb_run = wandb.init(project="OEI",
                               name=cfg.trainer['prefix'],
                               entity="icall-oei",
                               group=prefix,
                               reinit=True)
        if 'pre_train_path' not in cfg.trainer:
            cfg.trainer['pre_train_path'] = os.path.normpath(
                f"./dev/model/{cfg.trainer['prefix']}_best.pth")
        writer = Writer(log_dir, str(seed), 'tensorboard') if log_dir else None
        info.append(run_once(cfg, dataset, vocab, device, writer, seed))
        wandb_run.finish()

    print("\ntotal info:")
    print(info)
    # print('\nAVG DEV: ', merge_dicts([i[0] for i in info], avg=True))
    # print('AVG TEST: ', merge_dicts([i[1] for i in info], avg=True))


if __name__ == "__main__":
    # OpinionDataset.new_crowd('data', 'test-crowd-r2')
    main()
