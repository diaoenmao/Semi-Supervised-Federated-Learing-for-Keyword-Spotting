import argparse
import copy
import datetime
import models
import os
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset_semi, make_transform, \
    make_fix_transform
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


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset)
    sup_dataset, unsup_dataset, supervised_idx = separate_dataset_semi(dataset['train'])
    sup_dataset.transform = make_transform(cfg['sup_aug'])
    unsup_dataset.transform = make_fix_transform(cfg['loss_mode'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, 'local')
    scheduler = make_scheduler(optimizer, 'global')
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    result = resume(cfg['model_tag'])
    if result is None:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = result['epoch']
        supervised_idx = result['supervised_idx']
        model.load_state_dict(result['model_state_dict'])
        optimizer.load_state_dict(result['optimizer_state_dict'])
        scheduler.load_state_dict(result['scheduler_state_dict'])
        logger = result['logger']
        sup_dataset, unsup_dataset, supervised_idx = separate_dataset_semi(dataset['train'], supervised_idx)
    sup_dataloader = make_data_loader({'train': sup_dataset}, cfg['model_name'])
    unsup_sampler = UnSupSampler(len(sup_dataloader), cfg[cfg['model_name']]['batch_size']['train'], cfg['sup_ratio'],
                                 len(unsup_dataset))
    unsup_dataloader = make_data_loader({'train': unsup_dataset}, cfg['model_name'],
                                        batch_sampler={'train': unsup_sampler})
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train(sup_dataloader['train'], unsup_dataloader['train'], model, optimizer, metric, logger, epoch)
        test(data_loader['test'], model, metric, logger, epoch)
        scheduler.step()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model.module.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train(sup_dataloader, unsup_dataloader, model, optimizer, metric, logger, epoch):
    logger.safe(True)
    model.train(True)
    start_time = time.time()
    for i, (sup_input, unsup_input) in enumerate(zip(sup_dataloader, unsup_dataloader)):
        sup_input = collate(sup_input)
        unsup_input = collate(unsup_input)
        sup_input = to_device(sup_input, cfg['device'])
        unsup_input = to_device(unsup_input, cfg['device'])
        with torch.no_grad():
            model.train(False)
            unsup_output_ = model(unsup_input)
            buffer = torch.softmax(unsup_output_['target'], dim=-1)
            new_target, mask = make_hard_pseudo_label(buffer)
            sup_input['target'] = new_target.detach()
        input_size = sup_input['data'].size(0)
        optimizer.zero_grad()
        output = model(sup_input)
        if torch.any(mask):
            unsup_input['loss_mode'] = cfg['loss_mode']
            output_ = model(unsup_input)
            output['loss'] += output_['loss']
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], sup_input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(sup_dataloader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(sup_dataloader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * len(sup_dataloader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(sup_dataloader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


class UnSupSampler(torch.utils.data.Sampler):
    def __init__(self, num_batches, sup_batch_size, sup_ratio, data_size):
        self.num_batches = num_batches
        self.sup_batch_size = sup_batch_size
        self.sup_ratio = sup_ratio
        self.data_size = data_size
        self.batch_size = int(sup_batch_size / sup_ratio)

    def __iter__(self):
        yield from torch.randperm(self.data_size)[:self.batch_size].tolist()

    def __len__(self):
        return self.num_batches


def make_hard_pseudo_label(soft_pseudo_label):
    max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
    mask = max_p.ge(cfg['threshold'])
    return hard_pseudo_label, mask


if __name__ == "__main__":
    main()
