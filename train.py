import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import numpy as np

import os
import sys
import time
import optparse

from tensorboardX import SummaryWriter

import utils
import config
from data import Dataset
from model import PerformanceRNN
from sequence import NoteSeq, EventSeq, ControlSeq

# pylint: disable=E1102
# pylint: disable=E1101

#========================================================================
# Settings
#========================================================================

def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='train.sess')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='dataset/processed/')

    parser.add_option('-i', '--saving-interval',
                      dest='saving_interval',
                      type='float',
                      default=60.)

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=config.train['batch_size'])

    parser.add_option('-l', '--learning-rate',
                      dest='learning_rate',
                      type='float',
                      default=config.train['learning_rate'])

    parser.add_option('-w', '--window-size',
                      dest='window_size',
                      type='int',
                      default=config.train['window_size'])

    parser.add_option('-z', '--stride-size',
                      dest='stride_size',
                      type='int',
                      default=config.train['stride_size'])

    parser.add_option('-c', '--control-ratio',
                      dest='control_ratio',
                      type='float',
                      default=config.train['control_ratio'])

    parser.add_option('-t', '--use-transposition',
                      dest='use_transposition',
                      action='store_true',
                      default=config.train['use_transposition'])

    parser.add_option('-p', '--model-params',
                      dest='model_params',
                      type='string',
                      default='')
                      
    parser.add_option('-r', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]

options = get_options()

#------------------------------------------------------------------------

sess_path = options.sess_path
data_path = options.data_path
saving_interval = options.saving_interval

learning_rate = options.learning_rate
batch_size = options.batch_size
window_size = options.window_size
stride_size = options.stride_size
use_transposition = options.use_transposition
control_ratio = options.control_ratio
reset_optimizer = options.reset_optimizer

event_dim = EventSeq.dim()
control_dim = ControlSeq.dim()
model_config = config.model
model_params = utils.params2dict(options.model_params)
model_config.update(model_params)
device = config.device

print('-' * 50)

print('Session path:', sess_path)
print('Dataset path:', data_path)
print('Saving interval:', saving_interval)
print('-' * 50)

print('Hyperparameters:', utils.dict2params(model_config))
print('Learning rate:', learning_rate)
print('Batch size:', batch_size)
print('Window size:', window_size)
print('Stride size:', stride_size)
print('Control ratio:', control_ratio)
print('Random transposition:', use_transposition)
print('Reset optimizer:', reset_optimizer)
print('Device:', device)
print('-' * 50)


#========================================================================
# Load session and dataset
#========================================================================

def load_session():
    model = PerformanceRNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    try:
        sess = torch.load(sess_path)
        model.load_state_dict(sess['model'])
        if not reset_optimizer:
            optimizer.load_state_dict(sess['optimizer'])
        print('Session is loaded from', sess_path)
    except:
        print('New session')
        pass
    return model, optimizer

def load_dataset():
    dataset = Dataset(data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset


print('Loading session')
model, optimizer = load_session()
print(model)

print('-' * 50)

print('Loading dataset')
dataset = load_dataset()
print(dataset)

print('-' * 50)

#------------------------------------------------------------------------

def save_model():
    global model, optimizer
    print('Saving to', sess_path)
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, sess_path)
    print('Done saving')


#========================================================================
# Training
#========================================================================

writer = SummaryWriter()
last_saving_time = time.time()
loss_function = nn.CrossEntropyLoss()

try:
    batch_gen = dataset.batches(batch_size, window_size, stride_size)

    for iteration, (events, controls) in enumerate(batch_gen):
        if use_transposition:
            offset = np.random.choice(np.arange(-6, 6))
            events, controls = utils.transposition(events, controls, offset)

        events = torch.LongTensor(events).to(device)
        assert events.shape[0] == window_size

        if np.random.random() < control_ratio:
            controls = torch.FloatTensor(controls).to(device)
            assert controls.shape[0] == window_size
        else:
            controls = None

        init = torch.randn(batch_size, model.init_dim).to(device)
        outputs = model.generate(init, window_size, events[:-1], controls, output_type='logit')
        assert outputs.shape[:2] == events.shape[:2]

        loss = loss_function(outputs.view(-1, event_dim), events.view(-1))
        model.zero_grad()
        loss.backward()
        writer.add_scalar('loss', loss.item(), iteration)

        # norm = utils.compute_gradient_norm(model.parameters())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # writer.add_scalar('norm', norm.item(), iteration)
        
        optimizer.step()

        print(f'iter {iteration}, loss: {loss.item()}')

        if time.time() - last_saving_time > saving_interval:
            save_model()
            last_saving_time = time.time()

except KeyboardInterrupt:
    save_model()
