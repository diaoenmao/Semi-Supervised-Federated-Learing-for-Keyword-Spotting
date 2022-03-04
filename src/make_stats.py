import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_plain_transform
from utils import save, process_control, process_dataset, collate, Stats, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

stats_path = './res/stats'
dim = 1

if __name__ == "__main__":
    import datasets

    process_control()
    cfg['seed'] = 0
    data_names = ['SpeechCommandsV1', 'SpeechCommandsV2']
    with torch.no_grad():
        for data_name in data_names:
            cfg['data_name'] = data_name
            root = os.path.join('data', cfg['data_name'])
            dataset = eval('datasets.{}(root=root, split=\'train\')'.format(cfg['data_name']))
            cfg['data_length'] = 1 * dataset.sr
            cfg['n_fft'] = round(0.04 * dataset.sr)
            cfg['hop_length'] = round(0.02 * dataset.sr)
            cfg['background_noise'] = dataset.background_noise
            plain_transform = make_plain_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
            dataset.transform = datasets.Compose(plain_transform)
            data_loader = make_data_loader({'train': dataset}, cfg['model_name'])
            stats = Stats(dim=dim)
            for i, input in enumerate(data_loader['train']):
                input = collate(input)
                stats.update(input['data'])
            stats = (stats.mean.tolist(), stats.std.tolist())
            print(cfg['data_name'], stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}.pt'.format(cfg['data_name'])))
