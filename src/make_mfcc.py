import argparse
import copy
import datetime
import models
import os
import shutil
import datasets
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_transform
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

if __name__ == "__main__":
    cfg['seed'] = 0
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['control']['data_name'] = 'SpeechCommandsV1'
    process_control()
    augs = ['plain', 'basic-rands', 'basic-spec-rands']
    dataset = fetch_dataset(cfg['data_name'])
    for i in range(len(augs)):
        dataset['train'].transform = datasets.Compose([make_transform(augs[i])])
        data_loader = make_data_loader(dataset, cfg['model_name'], shuffle={'train': False, 'test': False},
                                       batch_size={'train': 1, 'test': 1})
        for i, input in enumerate(data_loader['train']):
            input = collate(input)
            print(i, input['data'].shape, input['target'].shape)
            torchvision.utils.save_image(input['data'], './output/train.png')
            # torchaudio.save('./output/temp.wav', input['data'][0], 16000)
            break
