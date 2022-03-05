import copy
import os
import torch
import torchaudio
import torchvision
import numpy as np
import models
from config import cfg
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['SpeechCommandsV1', 'SpeechCommandsV2']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
        cfg['data_length'] = 1 * dataset['train'].sr
        cfg['n_fft'] = round(0.04 * dataset['train'].sr)
        cfg['hop_length'] = round(0.02 * dataset['train'].sr)
        cfg['background_noise'] = dataset['train'].background_noise
        if cfg['aug'] == 'plain':
            train_transform = make_plain_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
        elif cfg['aug'] == 'basic':
            train_transform = make_basic_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'],
                                                   cfg['background_noise'])
        elif cfg['aug'] == 'basic-spec':
            train_transform = make_basic_spec_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'],
                                                        cfg['background_noise'])
        elif cfg['aug'] == 'basic-spec-ps':
            train_transform = make_basic_spec_ps_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'],
                                                           cfg['background_noise'])
        elif cfg['aug'] == 'basic-spec-ps-rand':
            train_transform = make_basic_spec_ps_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'],
                                                           cfg['background_noise'])
        else:
            raise ValueError('Not valid aug')
        plain_transform = make_plain_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
        test_transform = plain_transform
        dataset['train'].transform = datasets.Compose(
            [*train_transform, torchvision.transforms.Normalize(*cfg['stats'][data_name])])
        dataset['test'].transform = datasets.Compose(
            [*test_transform, torchvision.transforms.Normalize(*cfg['stats'][data_name])])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=input_collate, worker_init_fn=np.random.seed(cfg['seed']))
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=input_collate, worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=input_collate, worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = iid(dataset['train'], num_users)
        data_split['test'] = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'] = non_iid(dataset['train'], num_users)
        data_split['test'] = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    return data_split


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset, num_users):
    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
        for target_i in range(cfg['target_size']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['target_size']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split


def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    separated_dataset.id = list(range(len(separated_dataset.data)))
    return separated_dataset


def separate_dataset_semi(dataset, supervised_idx=None):
    if supervised_idx is None:
        if cfg['num_supervised'] == -1:
            supervised_idx = list(range(len(dataset)))
        else:
            target = torch.tensor(dataset.target)
            num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
            supervised_idx = []
            for i in range(cfg['target_size']):
                idx = torch.where(target == i)[0]
                idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                supervised_idx.extend(idx)
    idx = list(range(len(dataset)))
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    sup_dataset = separate_dataset(dataset, supervised_idx)
    unsup_dataset = separate_dataset(dataset, unsupervised_idx)
    return sup_dataset, unsup_dataset, supervised_idx

def make_batchnorm_dataset_su(server_dataset, client_dataset):
    batchnorm_dataset = copy.deepcopy(server_dataset)
    batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data
    batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target
    batchnorm_dataset.id = batchnorm_dataset.id + client_dataset.id
    return batchnorm_dataset


def make_dataset_normal(dataset):
    _transform = dataset.transform
    plain_transform = make_plain_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    dataset.transform = plain_transform
    return dataset, _transform


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        dataset, _transform = make_dataset_normal(dataset)
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
        dataset.transform = _transform
    return test_model


def make_plain_transform(data_length, n_fft, hop_length):
    import datasets
    n_stft = n_fft // 2 + 1
    plain_transform = [datasets.transforms.CenterCropPad(data_length),
                       torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
                       datasets.transforms.ComplextoPower(),
                       datasets.transforms.SpectoMFCC(n_mfcc=40, melkwargs={'n_stft': n_stft}),
                       torchaudio.transforms.AmplitudeToDB('power', 80),
                       datasets.transforms.SpectoImage(),
                       torchvision.transforms.ToTensor()]
    return plain_transform


def make_basic_transform(data_length, n_fft, hop_length, background_noise):
    import datasets
    n_stft = n_fft // 2 + 1
    basic_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                       datasets.transforms.CenterCropPad(data_length),
                       datasets.transforms.RandomTimeShift(0.1),
                       datasets.transforms.RandomBackgroundNoise(background_noise, 0.8, 0.1),
                       torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
                       datasets.transforms.ComplextoPower(),
                       datasets.transforms.SpectoMFCC(n_mfcc=40, melkwargs={'n_stft': n_stft}),
                       torchaudio.transforms.AmplitudeToDB('power', 80),
                       datasets.transforms.SpectoImage(),
                       torchvision.transforms.ToTensor()]
    return basic_transform


def make_basic_spec_transform(data_length, n_fft, hop_length, background_noise):
    import datasets
    n_stft = n_fft // 2 + 1
    basic_spec_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                            datasets.transforms.CenterCropPad(data_length),
                            datasets.transforms.RandomTimeShift(0.1),
                            datasets.transforms.RandomBackgroundNoise(background_noise, 0.8, 0.1),
                            torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
                            datasets.transforms.ComplextoPower(),
                            torchaudio.transforms.FrequencyMasking(42),
                            torchaudio.transforms.TimeMasking(12),
                            datasets.transforms.SpectoMFCC(n_mfcc=40, melkwargs={'n_stft': n_stft}),
                            torchaudio.transforms.AmplitudeToDB('power', 80),
                            datasets.transforms.SpectoImage(),
                            torchvision.transforms.ToTensor()]
    return basic_spec_transform


def make_basic_spec_ps_transform(data_length, n_fft, hop_length, background_noise):
    import datasets
    n_stft = n_fft // 2 + 1
    basic_spec_ps_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                               datasets.transforms.CenterCropPad(data_length),
                               datasets.transforms.RandomTimeShift(0.1),
                               datasets.transforms.RandomBackgroundNoise(background_noise, 0.8, 0.1),
                               torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
                               datasets.transforms.RandomPitchShift(4),
                               datasets.transforms.ComplextoPower(),
                               torchaudio.transforms.FrequencyMasking(42),
                               torchaudio.transforms.TimeMasking(12),
                               datasets.transforms.SpectoMFCC(n_mfcc=40, melkwargs={'n_stft': n_stft}),
                               torchaudio.transforms.AmplitudeToDB('power', 80),
                               datasets.transforms.SpectoImage(),
                               torchvision.transforms.ToTensor()]
    return basic_spec_ps_transform


def make_basic_spec_ps_rand_transform(data_length, n_fft, hop_length, background_noise):
    import datasets
    n_stft = n_fft // 2 + 1
    basic_spec_ps_rand_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                                    datasets.transforms.CenterCropPad(data_length),
                                    datasets.transforms.RandomTimeShift(0.1),
                                    datasets.transforms.RandomBackgroundNoise(background_noise, 0.8, 0.1),
                                    torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
                                    datasets.transforms.RandomPitchShift(4),
                                    datasets.transforms.ComplextoPower(),
                                    torchaudio.transforms.FrequencyMasking(42),
                                    torchaudio.transforms.TimeMasking(12),
                                    datasets.transforms.SpectoMFCC(n_mfcc=40, melkwargs={'n_stft': n_stft}),
                                    torchaudio.transforms.AmplitudeToDB('power', 80),
                                    datasets.transforms.SpectoImage(),
                                    datasets.randaugment.RandAugment(n=2, m=10),
                                    torchvision.transforms.ToTensor()]
    return basic_spec_ps_rand_transform


class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        input = {'data': input['data'], 'target': input['target']}
        return input

    def __len__(self):
        return self.size
