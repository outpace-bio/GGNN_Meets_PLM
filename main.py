import faulthandler
faulthandler.enable()

import argparse
from collections import defaultdict
from functools import partial

import numpy as np
import os
import sklearn.metrics as sk_metrics
import time
import torch
import torch.nn as nn
import torch_geometric
import tqdm
from atom3d.datasets import LMDBDataset
from atom3d.splits.splits import split_randomly
from atom3d.util import metrics
from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace

import gvp
import gvp.atom3d
from gvp import set_seed, Logger
from egnn import egnn_clean as eg


import torch
# import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    """
     Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


print = partial(print, flush=True)
models_dir = 'models'

parser = argparse.ArgumentParser()
parser.add_argument('task', metavar='TASK', default='PSR', choices=['PPBind', 'PPI'])
parser.add_argument('--toy', metavar='TOY TARGET', type=str, default='id', choices=['id', 'dist'])
parser.add_argument('--connect', metavar='CONNECTION', type=str, default='rball', choices=['rball', 'knn'])
parser.add_argument('--model', metavar='MODEL', type=str, default='gvp', choices=['egnn', 'gvp', 'molformer'])  # metavar will show in help information
parser.add_argument('--plm', metavar='PLM', type=int, default=0, help='whether use PLM features')
parser.add_argument('--num-workers', metavar='N', type=int, default=4, help='number of threads for loading data, default=4')
parser.add_argument('--lba-split', metavar='SPLIT', type=int, choices=[30, 60], help='identity cutoff for LBA, 30 (default) or 60', default=30)
parser.add_argument('--batch', metavar='SIZE', type=int, default=32, help='batch size, default=32 for gvp-gnn')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120, help='maximum time between evaluations on valset, default=120 minutes')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20, help='maximum time per evaluation on valset, default=20 minutes')
parser.add_argument('--epochs', metavar='N', type=int, default=200, help='training epochs, default=50')
parser.add_argument('--test', metavar='PATH', default=None, help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-4, type=float, help='learning rate')
parser.add_argument('--load', metavar='PATH', default=None, help='initialize first 2 GNN layers with pretrained weights')
parser.add_argument('--gpu', metavar='GPU', type=str, default='0')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
task_tag = args.task + str(args.lba_split) if args.task == 'LBA' else args.task
log = Logger(f'./', f'training_{task_tag}_{args.model}_{args.plm}.log')
# device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and not args.debug else 'cpu'
set_seed(0)


def collate(samples):
    if args.plm:
        nodes, coords, label, token_reps = zip(*samples)
    else:
        nodes, coords, label = zip(*samples)
    if args.task == 'PPI':
        nodes1, nodes2 = zip(*nodes)
        coords1, coords2 = zip(*coords)
        label1, label2 = zip(*label)
        nodes = nodes1 + nodes2
        coords = coords1 + coords2
        label = label1 + label2
        if args.plm:
            token_reps1, token_reps2 = zip(*token_reps)
            token_reps = token_reps1 + token_reps2
    nodes = pad_sequence(nodes, batch_first=True, padding_value=21)         # 21 is the token id of [UNK]
    coords = pad_sequence(coords, batch_first=True, padding_value=0.0)
    if args.task in ['TOY', 'PPI']:
        label = pad_sequence(label, batch_first=True, padding_value=-1)
    elif args.task == 'PSR':
        label, id = zip(*label)
        label = torch.stack(label)
    else:
        label = torch.stack(label)
    batch = SimpleNamespace(label=label, nodes=nodes, coords=coords)
    if args.plm:
        token_reps = pad_sequence(token_reps, batch_first=True, padding_value=0.0)
        batch.token_reps = token_reps
    if args.task == 'PSR':
        batch.id = id
    return batch


def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    gpu_id = rank

    datasets = get_datasets(args.task, args.lba_split, device=gpu_id)
    # if args.plm: args.num_workers = 0  # https://stackoverflow.com/questions/59081290/not-using-multiprocessing-but-get-cuda-error-on-google-colab-while-using-pytorch
    
    # There is an issue serializing LMDB data in multiprocessing mode so we need to set num_workers to 0
    args.num_workers = 0
    
    log.logger.info(f'{"=" * 40} Configuration {"=" * 40}\nModel: {args.model}; Task: {args.task}; PLM: {args.plm}; Graph: {args.connect}; Epochs: {args.epochs};'
                    f' Batch Szie: {args.batch}; GPU: {args.gpu}; Worker: {args.num_workers}\n{"=" * 40} Start Training {"=" * 40}')


    trainset = torch_geometric.loader.DataLoader(dataset=datasets[0],
                                                 num_workers=args.num_workers,
                                                 batch_size=args.batch,
                                                 shuffle=False,
                                                 sampler=DistributedSampler(datasets[0]))
    
    valset = torch_geometric.loader.DataLoader(dataset=datasets[1],
                                                num_workers=args.num_workers,
                                                batch_size=args.batch,
                                                shuffle=False,
                                                sampler=DistributedSampler(datasets[1]))

    testset = torch_geometric.loader.DataLoader(dataset=datasets[2],
                                                num_workers=args.num_workers,
                                                batch_size=args.batch,
                                                shuffle=False,
                                                sampler=DistributedSampler(datasets[2]))

    # model = get_model(args.task, args.model).to(device)
    model = DDP(get_model(args.task, args.model).to(gpu_id), device_ids=[gpu_id], find_unused_parameters=True)

    if args.test:
        test(model, testset, args.test)
    else:
        model_path = train(model, trainset, valset, patience=8)
        # test(model, testset, model_path.split('/')[-1])


def test(model, testset, model_path):
    model.load_state_dict(torch.load('models/' + model_path))
    print('Loading model weight successfully! Start to test. ')
    model.eval()
    t = tqdm.tqdm(testset)
    metrics = get_metrics(args.task)
    targets, predicts, ids = [], [], []
    with torch.no_grad():
        for batch in t:
            pred = forward(model, batch, model.device)
            label = get_label(batch)
            if args.model == 'molformer' and args.task in ['TOY', 'PPI']:
                mask = (batch.nodes != 21)
                label, pred = label[mask], pred[mask]

            if args.task == 'RES' or (args.task == 'TOY' and args.toy == 'id'): pred = pred.argmax(dim=-1)
            if args.task == 'PSR': ids.extend(batch.id)
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))

    for name, func in metrics.items():
        if args.task == 'PSR': func = partial(func, ids=ids)
        value = func(targets, predicts)
        log.logger.info(f"{name}: {value}")


def train(model, trainset, valset, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=5, min_lr=5e-7)

    best_path, best_val, wait = None, np.inf, 0
    if not os.path.exists(models_dir): os.makedirs(models_dir)

    for epoch in range(args.epochs):
        trainset.sampler.set_epoch(epoch)
        
        model.train()
        
        train_loss = loop(trainset, model, optimizer=optimizer, max_time=args.train_time)

        model.eval()
        with torch.no_grad():
            val_loss = loop(valset, model, max_time=args.val_time)
        log.logger.info(f'[Epoch {epoch}] Train loss: {train_loss:.8f} Val loss: {val_loss:.8f}')

        if '0' in str(model.device):
            if val_loss < best_val:
                path = f"{models_dir}/{args.task}_centered_coords_{args.model}_plm{args.plm}_epoch{epoch}_val_loss_{val_loss}.chkpt"
                print(f'Saving to: {path}')
                torch.save(model.state_dict(), path)

                best_path, best_val = path, val_loss
            else:
                wait += 1

        log.logger.info(f'Best {best_path} Val loss: {best_val:.8f}\n')
        
        if wait >= patience: break                                          # early stop
        lr_scheduler.step(val_loss)                                         # based on validation loss
    return best_path


def loop(dataset, model, optimizer=None, max_time=None):
    start = time.time()
    loss_fn = get_loss(args.task)
    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0

    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            out = forward(model, batch, model.device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise e
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
        
        label = get_label(batch)
        loss_value = loss_fn(out, label.to(model.device))
        total_loss += float(loss_value)
        total_count += 1

        if optimizer:
            try:
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise e
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
        t.set_description(f"Loss: {total_loss / total_count:.5f}")  # tdqm的description

    return total_loss / total_count


def load(model, path):
    params = torch.load(path)
    state_dict = model.state_dict()
    for name, p in params.items():
        if name in state_dict and name[:8] in ['layers.0', 'layers.1'] and state_dict[name].shape == p.shape:
            print("Loading", name)
            model.state_dict()[name].copy_(p)


def get_label(batch, task=args.task):
    if task == 'PPBind':
        return batch[0].label.float()
    elif type(batch) in [list, tuple]:
        return torch.cat([i.label for i in batch])

    return batch.label


def get_metrics(task):
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])

    correlations = {'pearson': partial(_correlation, metrics.pearson), 'kendall': partial(_correlation, metrics.kendall), 'spearman': partial(_correlation, metrics.spearman)}
    mean_correlations = {f'mean {k}': partial(v, glob=False) for k, v in correlations.items()}

    if task == 'TOY':
        return {'rmse': partial(sk_metrics.mean_squared_error, squared=False)} if args.toy == 'dist' else {'accuracy': metrics.accuracy}
    else:
        return {'PSR': {**correlations, **mean_correlations}, 'PPI': {'auroc': metrics.auroc}, 'RES': {'accuracy': metrics.accuracy},
                'MSP': {'auroc': metrics.auroc, 'auprc': metrics.auprc}, 'LEP': {'auroc': metrics.auroc, 'auprc': metrics.auprc},
                'LBA': {**correlations, 'rmse': partial(sk_metrics.mean_squared_error, squared=False)}}[task]


def get_loss(task):
    if task == 'PPBind':
        return nn.BCEWithLogitsLoss()
    elif task in ['PPI']:
        return nn.BCELoss()                      # binary classification



def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = [x.to(device) for x in batch]      # PPI two graphs
    elif type(batch) == SimpleNamespace:
        for k in batch.__dict__:
            if k != 'id':
                batch.__dict__[k] = batch.__dict__[k].to(device)
    else:
        batch = batch.to(device)
    return model(batch)


def get_datasets(task, lba_split=30, device='cpu'):
    data_path = {'PPI': 'data/PPI/DIPS-split/data/', 'PPBind': 'data/paul_PPbind/test/'}[task]      # TOY use the test dataset of RES

    if task == 'PPI':
        # Paul edit -- they took only the test data and split it into train, val, test. Seems like the train data would
        # leak into the test data. 
        # After checking Atom3D, it seems that a seed is set so the splits should always be the same.
        dataset = LMDBDataset(data_path + 'test', transform=gvp.atom3d.PPITransform(plm=args.plm, device=device))
        trainset, valset, testset = split_randomly(dataset)
        # Let's load the predefined train, val, test splits (Paul)
        # trainset = LMDBDataset(data_path + 'train', transform=gvp.atom3d.PPITransform(plm=args.plm, device=device))
        # valset = LMDBDataset(data_path + 'val', transform=gvp.atom3d.PPITransform(plm=args.plm, device=device))
        # testset = LMDBDataset(data_path + 'test', transform=gvp.atom3d.PPITransform(plm=args.plm, device=device))
    elif task == 'PPBind':
        dataset = LMDBDataset(data_path, transform=gvp.atom3d.PPBindingTransform(plm=args.plm, device=device))
        trainset, valset, testset = split_randomly(dataset)

    print(len(trainset), len(valset), len(testset))
    return trainset, valset, testset


def get_model(task, model):
    if model == 'egnn':
        if task == 'PPBind':
            return eg.PPBindingModel(plm=args.plm)
        elif task == 'PPI':
            return eg.PPIModel(plm=args.plm)
        return {}[task]()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print('World size:', world_size)
    save_every = 1
    mp.spawn(main, args=(world_size, args.epochs, save_every,), nprocs=world_size)
