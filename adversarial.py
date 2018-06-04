# Adversarial learning for event-based music generation with SeqGAN
# Reference:
# "SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient."
# (Yu, Lantao, et al.).

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import os, sys, time, argparse
from progress.bar import Bar

import config, utils
from config import device
from data import Dataset
from model import PerformanceRNN
from sequence import EventSeq, ControlSeq
from data import Dataset

# pylint: disable=E1101


#========================================================================
# Discriminator
#========================================================================

discriminator_config = {
    'event_dim': EventSeq.dim(),
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0.3
}

class EventSequenceEncoder(nn.Module):
    def __init__(self, event_dim=EventSeq.dim(), hidden_dim=512,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()
        self.event_embedding = nn.Embedding(event_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.output_fc = nn.Linear(hidden_dim, 1)
        self.output_fc_activation = nn.Sigmoid()

    def forward(self, events, hidden=None, output_logits=False):
        # events: [steps, batch_size]
        events = self.event_embedding(events)
        outputs, _ = self.gru(events, hidden)
        output = outputs.mean(0) # [batch_size, hidden_dim]
        output = self.output_fc(output).squeeze(-1) # [batch_size]
        if output_logits:
            return output
        output = self.output_fc_activation(output)
        return output


#========================================================================
# Pretrain Discriminator
#========================================================================

def pretrain_discriminator(model_sess_path,         # load
                           discriminator_sess_path, # load + save
                           batch_data_generator,    # Dataset(...).batches(...)
                           discriminator_config_overwrite={},
                           control_ratio=1.0,
                           num_iter=-1,
                           save_interval=60.0,
                           discriminator_lr=0.001,
                           enable_logging=False):
    
    # Load generator
    model_sess = torch.load(model_sess_path)
    model_config = model_sess['model_config']
    model = PerformanceRNN(**model_config).to(device)
    model.load_state_dict(model_sess['model_state'])

    print('-' * 70)
    print(f'Generator from "{model_sess_path}"')
    print(model)
    print('-' * 70)

    # Load discriminator and optimizer
    global discriminator_config
    try:
        discriminator_sess = torch.load(discriminator_sess_path)
        discriminator_config = discriminator_sess['discriminator_config']
        discriminator_state = discriminator_sess['discriminator_state']
        discriminator_optimizer_state = discriminator_sess['discriminator_optimizer_state']
        print(f'Discriminator from "{discriminator_sess_path}"')
        discriminator_loaded = True
    except:
        print(f'New discriminator session at "{discriminator_sess_path}"')
        discriminator_config.update(discriminator_config_overwrite)
        discriminator_loaded = False

    discriminator = EventSequenceEncoder(**discriminator_config).to(device)
    optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_lr)
    if discriminator_loaded:
        discriminator.load_state_dict(discriminator_state)
        optimizer.load_state_dict(discriminator_optimizer_state)

    print(discriminator)
    print('-' * 70)

    def save_discriminator():
        print(f'Saving to "{discriminator_sess_path}"')
        torch.save({
            'discriminator_config': discriminator_config,
            'discriminator_state': discriminator.state_dict(),
            'discriminator_optimizer_state': optimizer.state_dict()
        }, discriminator_sess_path)
        print('Done saving')

    # Disable gradient for generator
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()
    discriminator.train()

    loss_func = nn.BCEWithLogitsLoss()
    last_save_time = time.time()

    if enable_logging:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    try:
        for i, (events, controls) in enumerate(batch_data_generator):
            if i == num_iter:
                break
            
            steps, batch_size = events.shape
            # Prepare inputs
            events = torch.LongTensor(events).to(device)
            if np.random.random() <= control_ratio:
                controls = torch.FloatTensor(controls).to(device)
            else:
                controls = None
            init = torch.randn(batch_size, model.init_dim).to(device)
            # Predict for real event sequence
            real_events = events
            real_logit = discriminator(real_events, output_logits=True)
            real_target = torch.ones_like(real_logit).to(device)
            # Predict for fake event sequence from the generator
            fake_events = model.generate(init, steps, events=events[:-1],
                                             controls=controls, output_type='index')
            fake_logit = discriminator(fake_events, output_logits=True)
            fake_target = torch.zeros_like(fake_logit).to(device)
            # Compute loss
            loss = (loss_func(real_logit, real_target) +
                    loss_func(fake_logit, fake_target)) / 2
            # Backprop
            discriminator.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'{i} loss: {loss.item()}')

            if enable_logging:
                writer.add_scalar('pretrain/D-loss', loss.item(), i)

            if last_save_time + save_interval < time.time():
                last_save_time = time.time()
                save_discriminator()

    except KeyboardInterrupt:
        save_discriminator()


#========================================================================
# Adversarial Learning
#========================================================================

def train_adversarial(sess_path, batch_data_generator,
                      model_load_path, model_optimizer_class,
                      model_learning_rate, reset_model_optimizer,
                      discriminator_load_path, discriminator_optimizer_class,
                      discriminator_learning_rate, reset_discriminator_optimizer,
                      g_d_training_steps, mc_sample_size, mc_sample_factor,
                      save_interval, control_ratio, enable_logging):
    
    if enable_logging:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    if os.path.isfile(sess_path):
        adv_state = torch.load(sess_path)
        model_config = adv_state['model_config']
        model_state = adv_state['model_state']
        model_optimizer_state = adv_state['model_optimizer_state']
        discriminator_config = adv_state['discriminator_config']
        discriminator_state = adv_state['discriminator_state']
        discriminator_optimizer_state = adv_state['discriminator_optimizer_state']
        print('-' * 70)
        print('Session is loaded from', sess_path)
        loaded_from_session = True

    else:
        model_sess = torch.load(model_load_path)
        model_config = model_sess['model_config']
        model_state = model_sess['model_state']
        discriminator_sess = torch.load(discriminator_load_path)
        discriminator_config = discriminator_sess['discriminator_config']
        discriminator_state = discriminator_sess['discriminator_state']
        loaded_from_session = False

    model = PerformanceRNN(**model_config)
    model.load_state_dict(model_state)
    model.to(device).train()
    model_optimizer = model_optimizer_class(model.parameters(), lr=model_learning_rate)

    discriminator = EventSequenceEncoder(**discriminator_config)
    discriminator.load_state_dict(discriminator_state)
    discriminator.to(device).train()
    discriminator_optimizer = discriminator_optimizer_class(discriminator.parameters(),
                                                            lr=discriminator_learning_rate)

    if loaded_from_session:
        if not reset_model_optimizer:
            model_optimizer.load_state_dict(model_optimizer_state)
        if not reset_discriminator_optimizer:
            discriminator_optimizer.load_state_dict(discriminator_optimizer_state)
    
    def print_info():
        print('-' * 70)
        print('Options')
        print('sess_path:', sess_path)
        print('save_interval:', save_interval)
        print('batch_data_generator:', batch_data_generator)
        print('control_ratio:', control_ratio)
        print('g_d_training_steps:', g_d_training_steps)
        print('mc_sample_size:', mc_sample_size)
        print('mc_sample_factor:', mc_sample_factor)
        print('enable_logging:', enable_logging)
        print('model_load_path:', model_load_path)
        print('model_optimizer_class:', model_optimizer_class)
        print('model_learning_rate:', model_learning_rate)
        print('reset_model_optimizer:', reset_model_optimizer)
        print('discriminator_load_path:', discriminator_load_path)
        print('discriminator_optimizer_class:', discriminator_optimizer_class)
        print('discriminator_learning_rate:', discriminator_learning_rate)
        print('reset_discriminator_optimizer:', reset_discriminator_optimizer)
        print('-' * 70)
        print(f'Generator from "{sess_path if loaded_from_session else model_load_path}"')
        print(model)
        print('-' * 70)
        print(f'Discriminator from "{sess_path if loaded_from_session else discriminator_load_path}"')
        print(discriminator)
        print('-' * 70)
    
    print_info()
    
    def save():
        print(f'Saving to "{sess_path}"')
        torch.save({
            'model_config': model_config,
            'model_state': model.state_dict(),
            'model_optimizer_state': model_optimizer.state_dict(),
            'discriminator_config': discriminator_config,
            'discriminator_state': discriminator.state_dict(),
            'discriminator_optimizer_state': discriminator_optimizer.state_dict()
        }, sess_path)
        print('Done saving')
    
    def mc_rollout(generated, hidden, total_steps, controls=None):
        # generated: [t, batch_size]
        # hidden: [n_layers, batch_size, hidden_dim]
        # controls: [total_steps - t, batch_size, control_dim]
        generated = torch.cat(generated, 0)
        generated_steps, batch_size = generated.shape # t, b
        steps = total_steps - generated_steps # s

        generated = generated.unsqueeze(1) # [t, 1, b]
        generated = generated.repeat(1, mc_sample_size, 1) # [t, mcs, b]
        generated = generated.view(generated_steps, -1) # [t, mcs * b]

        hidden = hidden.unsqueeze(1).repeat(1, mc_sample_size, 1, 1)
        hidden = hidden.view(model.gru_layers, -1, model.hidden_dim)

        if controls is not None:
            assert controls.shape == (steps, batch_size, model.control_dim)
            controls = controls.unsqueeze(1) # [s, 1, b, c]
            controls = controls.repeat(1, mc_sample_size, 1, 1) # [s, mcs, b, c]
            controls = controls.view(steps, -1, model.control_dim) # [s, mcs * b, c]

        event = generated[-1].unsqueeze(0) # [1, mcs * b]
        control = None # default when controls is None
        outputs = []

        for i in range(steps):
            if controls is not None:
                control = controls[i].unsqueeze(0) # [1, mcs * b, c]

            output, hidden = model.forward(event, control=control, hidden=hidden)
            probs = model.output_fc_activation(output) / mc_sample_factor
            event = Categorical(probs).sample() # [1, mcs * b]
            outputs.append(event)

        sequences = torch.cat([generated, *outputs], 0)
        assert sequences.shape == (total_steps, mc_sample_size * batch_size)
        return sequences

    try:
        last_save_time = time.time()
        
        for i, (events, controls) in enumerate(batch_data_generator):
            steps, batch_size = events.shape
            init = torch.randn(batch_size, model.init_dim).to(device)
            events = torch.LongTensor(events).to(device)
            use_control = np.random.random() <= control_ratio
            controls = torch.FloatTensor(controls).to(device) if use_control else None

            if (i % sum(g_d_training_steps)) < g_d_training_steps[0]:
                # Generator step
                hidden = model.init_to_hidden(init)
                event = model.get_primary_event(batch_size)
                outputs = []
                generated = []
                q_values = []

                for step in Bar('MC Rollout').iter(range(steps)):
                    control = controls[step].unsqueeze(0) if use_control else None
                    output, hidden = model.forward(event, control=control, hidden=hidden)
                    outputs.append(output)
                    probs = model.output_fc_activation(output / mc_sample_factor)
                    generated.append(Categorical(probs).sample())

                    with torch.no_grad():
                        if step < steps - 1:
                            sequences = mc_rollout(generated, hidden, steps, controls[step+1:])
                            mc_score = discriminator(sequences) # [mcs * b]
                            mc_score = mc_score.view(mc_sample_size, batch_size) # [mcs, b]
                            q_value = mc_score.mean(0, keepdim=True) # [1, batch_size]
                        
                        else:
                            q_value = discriminator(torch.cat(generated, 0))
                            q_value = q_value.unsqueeze(0) # [1, batch_size]
                    
                        q_values.append(q_value)
                
                q_values = torch.cat(q_values, 0) # [steps, batch_size]
                q_values = q_values - q_values.mean().detach() # baseline
                generated = torch.cat(generated, 0) # [steps, batch_size]
                outputs = torch.cat(outputs, 0) # [steps, batch_size, event_dim]

                loss = (F.cross_entropy(outputs.view(-1, model.event_dim),
                                        generated.view(-1),
                                        reduce=False) * q_values.view(-1)).mean()
                loss.backward()
                model.zero_grad()

                oracle_loss = F.cross_entropy(
                                outputs.view(-1, model.event_dim), events.view(-1))
                print(f'{i} (G-step) loss: {oracle_loss.item()}')
                if enable_logging:
                    writer.add_scalar('adversarial/G-loss', oracle_loss.item(), i)

            else:
                # Discriminator step
                with torch.no_grad():
                    generated = model.generate(init, steps, None, controls)
                    
                fake_logit = discriminator(generated, output_logits=True)
                real_logit = discriminator(events, output_logits=True)

                concat_logit = torch.cat([-fake_logit, real_logit], 0)
                concat_target = torch.ones_like(concat_logit)
                loss = F.binary_cross_entropy_with_logits(concat_logit, concat_target)

                discriminator.zero_grad()
                loss.backward()
                discriminator_optimizer.step()

                print(f'{i} (D-step) loss: {loss.item()}')
                if enable_logging:
                    writer.add_scalar('adversarial/D-loss', loss.item(), i)
                
            if last_save_time + save_interval < time.time():
                last_save_time = time.time()
                save()

    except KeyboardInterrupt:
        save()



#========================================================================
# Script Arguments
#========================================================================

def batch_generator(args):
    print('-' * 70)
    dataset = Dataset(args.dataset_path, verbose=True)
    print(dataset)
    return dataset.batches(args.batch_size, args.window_size, args.stride_size)

def pretrain(args):
    pretrain_discriminator(model_sess_path=args.generator_session_path,
                           discriminator_sess_path=args.discriminator_session_path,
                           discriminator_config_overwrite=utils.params2dict(args.discriminator_parameters),
                           batch_data_generator=args.batch_generator(args),
                           control_ratio=args.control_ratio,
                           num_iter=args.stop_iteration,
                           save_interval=args.save_interval,
                           discriminator_lr=args.discriminator_learning_rate,
                           enable_logging=args.enable_logging)

def adversarial(args):
    train_adversarial(sess_path=args.session_path,
                      batch_data_generator=args.batch_generator(args),
                      model_load_path=args.generator_load_path,
                      model_optimizer_class=getattr(optim, args.generator_optimizer),
                      model_learning_rate=args.generator_learning_rate,
                      reset_model_optimizer=args.reset_generator_optimizer,
                      discriminator_load_path=args.discriminator_load_path,
                      discriminator_optimizer_class=getattr(optim, args.discriminator_optimizer),
                      discriminator_learning_rate=args.discriminator_learning_rate,
                      reset_discriminator_optimizer=args.reset_discriminator_optimizer,
                      g_d_training_steps=list(map(int, args.g_d_training_steps.split(','))),
                      mc_sample_size=args.monte_carlo_sample_size,
                      mc_sample_factor=args.monte_carlo_sample_factor,
                      control_ratio=args.control_ratio,
                      save_interval=args.save_interval,
                      enable_logging=args.enable_logging)

def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.add_argument('-d', '--dataset-path', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-w', '--window-size', type=int, default=200)
    parser.add_argument('-s', '--stride-size', type=int, default=10)
    parser.set_defaults(batch_generator=batch_generator)
    pre_parser = subparsers.add_parser('pretrain', aliases=['p', 'pre'])
    pre_parser.add_argument('-G', '--generator-session-path', type=str, default=True)
    pre_parser.add_argument('-D', '--discriminator-session-path', type=str, required=True)
    pre_parser.add_argument('-p', '--discriminator-parameters', type=str, default='')
    pre_parser.add_argument('-l', '--discriminator-learning-rate', type=float, default=0.001)
    pre_parser.add_argument('-c', '--control-ratio', type=float, default=1.0)
    pre_parser.add_argument('-n', '--stop-iteration', type=int, default=-1)
    pre_parser.add_argument('-i', '--save-interval', type=float, default=60.0)
    pre_parser.add_argument('-L', '--enable-logging', action='store_true', default=False)
    pre_parser.set_defaults(main=pretrain)
    adv_parser = subparsers.add_parser('adversarial', aliases=['a', 'adv'])
    adv_parser.add_argument('-S', '--session-path', type=str, required=True)
    adv_parser.add_argument('-Gp', '--generator-load-path', type=str)
    adv_parser.add_argument('-Go', '--generator-optimizer', type=str, default='Adam')
    adv_parser.add_argument('-Gl', '--generator-learning-rate', type=float, default=0.001)
    adv_parser.add_argument('-Gr', '--reset-generator-optimizer', action='store_true', default=False)
    adv_parser.add_argument('-Dp', '--discriminator-load-path', type=str)
    adv_parser.add_argument('-Do', '--discriminator-optimizer', type=str, default='RMSprop')
    adv_parser.add_argument('-Dl', '--discriminator-learning-rate', type=float, default=0.001)
    adv_parser.add_argument('-Dr', '--reset-discriminator-optimizer', action='store_true', default=False)
    adv_parser.add_argument('-GD', '--g-d-training-steps', type=str, default='15,5')
    adv_parser.add_argument('-ms', '--monte-carlo-sample-size', type=int, default=8)
    adv_parser.add_argument('-mf', '--monte-carlo-sample-factor', type=float, default=1.0)
    adv_parser.add_argument('-c', '--control-ratio', type=float, default=1.0)
    adv_parser.add_argument('-i', '--save-interval', type=float, default=60.0)
    adv_parser.add_argument('-L', '--enable-logging', action='store_true', default=False)
    adv_parser.set_defaults(main=adversarial)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.main(args)
