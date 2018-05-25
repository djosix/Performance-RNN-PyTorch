import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from progress.bar import Bar
from config import device

# pylint: disable=E1101,E1102


class PerformanceRNN(nn.Module):
    def __init__(self, event_dim, control_dim, init_dim, hidden_dim,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()

        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + 1 + control_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim

        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.output_fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.)

    def _sample_event(self, output, greedy=True):
        if greedy:
            return output.argmax(-1)
        else:
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, control=None, hidden=None):
        # One step forward

        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)

        if control is None:
            default = torch.ones(1, batch_size, 1).to(device)
            control = torch.zeros(1, batch_size, self.control_dim).to(device)
        else:
            default = torch.zeros(1, batch_size, 1).to(device)
            assert control.shape == (1, batch_size, self.control_dim)

        concat = torch.cat([event, default, control], -1)
        input = self.concat_input_fc(concat)
        input = self.concat_input_fc_activation(input)

        _, hidden = self.gru(input, hidden)
        output = hidden.sum(0).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden

    def generate_steps(self, init, steps, events=None, controls=None,
                       greedy=1.0, output_type='index'):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[:steps-1]

        use_control = controls is not None
        if use_control:
            assert len(controls.shape) == 3
            if controls.shape[0] > 1:
                assert controls.shape[0] >= steps
                controls = controls[:steps]
            else:
                controls = controls.repeat(steps, 1, 1)
        
        hidden = self.inithid_fc(init)
        hidden = self.inithid_fc_activation(hidden)
        hidden = hidden.view(self.gru_layers, batch_size, self.hidden_dim)

        primary_event_index = self.event_dim - 1
        event = torch.tensor([[primary_event_index] * batch_size]).to(device)

        for step in range(steps):
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forward(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy)

            if output_type == 'index':
                yield event
            elif output_type == 'softmax':
                yield self.output_fc_activation(output)
            elif output_type == 'logit':
                yield output
            elif output_type == 'index_and_logit':
                yield event, output
            else:
                assert False
            if use_teacher_forcing and step < steps - 1: # avoid last one
                event = events[step].unsqueeze(0)

    def generate(self, *kargs, verbose=False, **kwargs):
        steps = self.generate_steps(*kargs, **kwargs)
        if verbose:
            steps = Bar('Generating', max=kargs[1]).iter(steps)
        outputs = torch.cat(list(steps), 0)
        return outputs

