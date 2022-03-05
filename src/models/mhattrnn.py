import torch
import torch.nn as nn
from config import cfg
from .utils import init_param, loss_fn


class MHAttRNN(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.cnn = nn.Sequential(
            nn.Conv2d(data_shape[1], hidden_size, (1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, (1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(1, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.mhatt = nn.MultiheadAttention(embed_dim=2 * hidden_size, num_heads=num_heads, dropout=dropout,
                                           batch_first=True)
        self.linear = nn.Linear(hidden_size, target_size)

    def f(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1, x.size(-1)).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, [x.size(1) // 2], :]
        x, _ = self.mhatt(x, x, x)
        x = x.squeeze(1)
        x = self.linear(x)
        return x

    def forward(self, input):
        output = {}
        output['target'] = self.f(input['data'])
        if 'loss_mode' in input:
            if input['loss_mode'] == 'sup':
                output['loss'] = loss_fn(output['target'], input['target'])
            elif input['loss_mode'] == 'fix':
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif input['loss_mode'] == 'fix-mix':
                aug_output = self.f(input['aug'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
                mix_output = self.f(input['mix_data'])
                output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                        1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
            else:
                raise ValueError('Not valid loss mode')
        else:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def mhattrnn():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['mhattrnn']['hidden_size']
    num_heads = cfg['mhattrnn']['num_heads']
    dropout = cfg['mhattrnn']['dropout']
    model = MHAttRNN(data_shape, hidden_size, target_size, num_heads, dropout)
    model.apply(init_param)
    return model
