from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import torchaudio
import models

if __name__ == "__main__":
    cfg['seed'] = 0
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['control']['data_name'] = 'SpeechCommandsV1'
    cfg['control']['model_name'] = 'conv'
    process_control()
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(len(dataset['train']), len(dataset['test']))
    print(len(data_loader['train']), len(data_loader['test']))
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        print(i, input['data'].shape, input['target'].shape)
        # torchaudio.save('./output/temp.wav', input['data'][0], 16000)
        # exit()
        break
    for i, input in enumerate(data_loader['test']):
        input = collate(input)
        print(i, input['data'].shape, input['target'].shape)
        break
