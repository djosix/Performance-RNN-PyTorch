import torch
import numpy as np
import os, sys

import config, utils
from config import device, model as model_config
from model import PerformanceRNN
from data import Dataset
from sequence import EventSeq, Control, ControlSeq

# pylint: disable=E1101,E1102


#========================================================================
# Settings
#========================================================================

event_dim = EventSeq.dim()
save_dir = sys.argv[1]
load_path = sys.argv[2]
batch_size = int(sys.argv[3])
steps = int(sys.argv[4])
greedy = float(sys.argv[5])

print('=' * 50)
print('Model:', load_path)
print('Batch size:', batch_size)
print('Steps:', steps)
print('Greedy:', greedy)

try:
    pitch_probs = [s for s in sys.argv[6].split(',') if s]
    
    if not pitch_probs:
        pitch_probs = [0.] * 12
    else:
        pitch_probs = list(map(float, pitch_probs))
        assert len(pitch_probs) == 12
    
    pitch_probs = np.array(pitch_probs)
    if pitch_probs.sum() != 0:
        pitch_probs = pitch_probs / pitch_probs.sum()
    
    density = int(sys.argv[7])
    assert density in range(len(ControlSeq.note_density_bins))

    print('Pitch histogram:', pitch_probs)
    print('Note density:', density)
    conditioning = True

except:
    conditioning = False

print('=' * 50)


#========================================================================
# Generating
#========================================================================

model = PerformanceRNN(**model_config).to(device)
model.load_state_dict(torch.load(load_path)['model'])

if conditioning:
    controls = utils.make_controls(batch_size, pitch_probs, density, steps=1)
    controls = torch.FloatTensor(controls).to(device)
else:
    controls = None

init = torch.randn(batch_size, model.init_dim).to(device)
outputs = model.generate(init, steps, controls=controls, greedy=greedy) # [steps, batch]
outputs = outputs.cpu().numpy().T # [batch, steps]


#========================================================================
# Saving
#========================================================================

os.makedirs(save_dir, exist_ok=True)

for i, output in enumerate(outputs):
    name = f'output-{i:03d}.mid'
    path = os.path.join(save_dir, name)
    n_notes = utils.event_indeces_to_midi_file(output, path)
    print(path, f'{n_notes} notes')
