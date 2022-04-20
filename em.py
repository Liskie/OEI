# from typing import Dict, Callable
import copy
import time

import torch

from nmnlp.common.util import sec_to_time, output as printf
from nmnlp.common.writer import Writer
from nmnlp.common import Trainer
from nmnlp.common.optim import build_optimizer
from nmnlp.common.trainer import format_metric
from nmnlp.common.metrics import a_better_than_b

from util import compute_user_scores
from models import build_model
from dataset import OpinionDataset, group_by_ins, group_data

USER_NUM = 70


def em_once(cfg, dataset, vocab, device, seed: int, prefix, log_dir):
    group_set = OpinionDataset(group_by_ins(copy.deepcopy(dataset.train.data)))
    group_set.vec_fields = ['words']
    if 'prob' in prefix:
        group_data(dataset.train, by_ins=True)

    metrics = dict(dev=[], test=[])
    prefix = f"{prefix}_{seed}_"
    model = None
    last_e = []

    # cfg.model['name'] = 'tagger'
    # model = build_model(vocab=vocab, **cfg.model)
    # model.to(device=device)
    # model.load(f'dev/model/select_{seed}_best.pth', device)

    # with torch.no_grad():
    #     expect = compute_user_scores(model, group_set, device)
    # print('\n')
    # printf(f'E step [{seed}] finished: mean user scores ', expect[:-1].mean())
    # print(expect.tolist())

    # return

    for i in range(30):
        # E step
        with torch.no_grad():
            expect = compute_user_scores(model, group_set, device)
        print('\n')
        printf(f'E step [{i}] finished: mean user scores ', expect.mean())
        print(expect.tolist())
        print('\n')

        if last_e:
            if all(last_e[-1] == expect) or (len(last_e) > 1 and all(last_e[-2] == expect)):
                print('收敛')
                break
        else:
            last_e.append(expect)

        del model
        torch.cuda.empty_cache()
        time.sleep(5)

        # M step
        model = build_model(vocab=vocab, **cfg.model)

        dev_metric, test_metric = bimix_step(
            cfg, model, dataset, vocab, device, prefix, expect.to(device))
        # setattr(model, 'scores', expect.to(device=device))
        # # writer = Writer(log_dir, f"{seed}_{i}", backend='tensorboard') if log_dir else None
        # dev_metric, test_metric = x_step(
        #     cfg, model, dataset, vocab, device, None, prefix + str(i))
        metrics['dev'].append(dev_metric)
        metrics['test'].append(test_metric)

        print('\n')
        printf(f'M step [{i}] finished.')

    return metrics


def x_step(cfg, model, dataset, vocab, device, writer: Writer, prefix):
    model.to(device=device)
    optimizer = build_optimizer(model, **cfg.optim)
    trainer = Trainer(vars(cfg), dataset, vocab, model, optimizer, None, None,
                      writer, device, prefix=prefix, **cfg.trainer)

    trainer.train()
    printf(model.metric.data_info)

    trainer.load()
    test_metric = trainer.test(dataset.test)
    return model.metric.best, test_metric


def bimix_step(cfg, model, dataset, vocab, device, prefix, expect):
    def load_and_set(epoch):
        trainer.pre_train_path = f"dev/model/{prefix}rande.pth"
        trainer.load()
        # avg_emb = expect[:-1].softmax(0).unsqueeze(-1).to(device=device)
        # avg_emb = model.user_embedding.weight.data[:-1].mul(avg_emb).sum(0)
        # setattr(model, 'avg_embedding', avg_emb)
        # model.set_expert(expect)
        setattr(model, 'scores', expect)

    model.to(device=device)
    best_epoch, stop_counter, batch_size = -1, 0, 128
    best_metrics, last_metric = None, None
    trainer = Trainer(
        vars(cfg), dataset, vocab, model, None, device=device, prefix=prefix,
        **cfg.trainer
    )
    printf("Training started ...")
    time_start = time.time()
    for epoch in range(26):
        load_and_set(epoch)

        dev_metric = trainer.test(
            dataset.dev, batch_size, device, f'                       {epoch} DEV')
        test_metric = trainer.test(dataset.test, batch_size, device)
        if model.metric.is_best(dev_metric):
            best_epoch, stop_counter = epoch, 0
            best_metrics = dev_metric, test_metric
        elif last_metric and a_better_than_b(last_metric, dev_metric):
            stop_counter += 1
        last_metric = dev_metric
        if stop_counter > 5:
            break
        break

    time_train = time.time() - time_start
    printf(f'training compete, time: {sec_to_time(time_train)} .')
    printf(f"Best epoch: {best_epoch}, {format_metric(model.metric.best)}")
    load_and_set(best_epoch)
    return best_metrics
