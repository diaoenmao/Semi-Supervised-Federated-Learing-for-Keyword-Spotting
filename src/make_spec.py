import argparse
import copy
import datetime
import models
import os
import shutil
import datasets
import time
import torch
import torchvision
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, make_transform, input_collate
from metrics import Metric
from utils import save, to_device, process_control, makedir_exist_ok, collate
from logger import make_logger

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    makedir_exist_ok('./output/spec')
    cfg['seed'] = 1
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['control']['data_name'] = 'SpeechCommandsV1'
    process_control()
    augs = ['plain', 'basic', 'basic-spec', 'basic-rands', 'basic-spec-rands']
    dataset = fetch_dataset(cfg['data_name'])
    target_dict = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6, "off": 7, "stop": 8,
                   "go": 9, 'silence': 10, 'unknown': 11}
    target_list = list(target_dict.keys())
    label_0, label_1 = 0, 1
    idx_0 = torch.arange(len(dataset['train'].target))[torch.tensor(dataset['train'].target) == label_0][0]
    idx_1 = torch.arange(len(dataset['train'].target))[torch.tensor(dataset['train'].target) == label_1][0]
    beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
    for i in range(len(augs)):
        dataset['train'].transform = datasets.Compose([make_transform(augs[i])])
        input_0 = input_collate([dataset['train'][idx_0]])
        input_0 = collate(input_0)
        input_1 = input_collate([dataset['train'][idx_1]])
        input_1 = collate(input_1)
        plot_spectrogram_1(input_0['data'], './output/spec/{}_{}_0.png'.format(augs[i], target_list[label_0]))
        plot_spectrogram_1(input_1['data'], './output/spec/{}_{}_1.png'.format(augs[i], target_list[label_1]))
        lam = 0.7
        input_mix = lam * input_0['data'] + (1 - lam) * input_1['data']
        plot_spectrogram_1(input_mix, './output/spec/{}_mix_{}.png'.format(augs[i], lam))
    return


def plot_spectrogram_1(spec, path, aspect='auto', info=False):
    spec = spec.numpy()[0, 0]
    plt.figure()
    plt.imshow(spec, origin='lower', aspect=aspect)
    if info:
        plt.ylabel('Frequency Bins')
        plt.xlabel('Time Frames')
        plt.colorbar()
    else:
        plt.axis('off')
    plt.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0)
    return


def plot_spectrogram_2(spec, path):
    torchvision.utils.save_image(spec, path)
    return


if __name__ == "__main__":
    main()
