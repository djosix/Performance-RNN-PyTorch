import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import numpy as np
import sys, os, time, optparse

from tensorboardX import SummaryWriter

import config, utils
from model import PerformanceRNN
from sequence import NoteSeq, EventSeq, ControlSeq
from data import Dataset

# pylint: disable=E1102
# pylint: disable=E1101

#========================================================================

def get_options():
    parser = optparse.OptionParser()
    parser.add_option('-s',
        dest='save_path',
        type='string',
        default='train.sess')
    parser.add_option('-d',
        dest='data_path',
        type='string',
        default='dataset/processed/')
    parser.add_option('-i',
        dest='saving_interval',
        type='float',
        default=60.)
    return parser.parse_args()[0]

options = get_options()

#========================================================================

save_path = options.save_path
data_path = options.data_path
saving_interval = options.saving_interval

event_dim = EventSeq.dim()
control_dim = ControlSeq.dim()
model_config = config.model
learning_rate = config.train['learning_rate']
batch_size = config.train['batch_size']
window_size = config.train['window_size']
stride_size = config.train['stride_size']
use_transposition = config.train['use_transposition']
device = config.device

print('=' * 50)
print('Saving path:', options.save_path)
print('Dataset path:', options.data_path)
print('Saving interval:', options.saving_interval)
print('Event dimension:', event_dim)
print('Hyperparameters:', model_config)
print('Learning rate:', learning_rate)
print('Batch size:', batch_size)
print('Window size:', window_size)
print('Stride size:', stride_size)
print('Random transposition:', use_transposition)
print('=' * 50)

#========================================================================


def load_model():
    model = PerformanceRNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    try:
        sess = torch.load(save_path)
        model.load_state_dict(sess['model'])
        optimizer.load_state_dict(sess['optimizer'])
        print('Session loaded')
    except:
        pass
    print(model)
    return model, optimizer

def load_dataset():
    dataset = Dataset(data_path)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    print('Dataset size:', dataset_size)
    return dataset


print('Loading model')
model, optimizer = load_model()

print('Loading dataset')
dataset = load_dataset()

print('=' * 50)

#========================================================================

def save_model():
    global model, optimizer
    print('Saving to', save_path)
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, save_path)
    print('Done saving')

#========================================================================

def compute_gradient_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

#========================================================================

writer = SummaryWriter()
last_saving_time = time.time()
loss_function = nn.CrossEntropyLoss()

try:
    batch_gen = dataset.batches(batch_size, window_size, stride_size)

    for iteration, batch in enumerate(batch_gen):
        events, controls = batch # [steps, batch] [steps, batch, control_dim]

        if use_transposition:
            offset = np.random.choice(np.arange(-6, 6))
            events, controls = utils.transposition(events, controls, offset)

        events = torch.LongTensor(events).to(device)
        controls = torch.FloatTensor(controls).to(device)
        init = torch.randn(batch_size, model.init_dim).to(device)
        init.requires_grad_() # start tracking the graph
        
        assert window_size == events.shape[0] == controls.shape[0]
        outputs = model.generate(init, window_size, events[:-1], controls, output_type='logit')
        assert outputs.shape[:2] == events.shape[:2]
        loss = loss_function(outputs.view(-1, event_dim), events.view(-1))

        model.zero_grad()
        loss.backward()
        norm = compute_gradient_norm(model.parameters())

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        writer.add_scalar('loss', loss.item(), iteration)
        writer.add_scalar('norm', norm.item(), iteration)
        print('iter {}, loss: {}'.format(iteration, loss.item()))

        if time.time() - last_saving_time > saving_interval:
            save_model()
            last_saving_time = time.time()

except KeyboardInterrupt:
    save_model()
