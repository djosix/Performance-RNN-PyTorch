
def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    import os
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def event_indeces_to_midi_file(event_indeces, midi_file_name, velocity_scale=0.8):
    from sequence import EventSeq
    event_seq = EventSeq.from_array(event_indeces)
    note_seq = event_seq.to_note_seq()
    for note in note_seq.notes:
        note.velocity = int((note.velocity - 64) * velocity_scale + 64)
    note_seq.to_midi_file(midi_file_name)
    return len(note_seq.notes)

def make_controls(batch_size, pitch_probs, density, steps=1):
    from sequence import Control, ControlSeq
    import numpy as np
    assert pitch_probs.shape[0] == 12
    assert density in range(0, ControlSeq.feat_dims()['note_density'])
    control = Control(pitch_probs, density).to_array()
    controls = np.array([[control] * batch_size] * steps)
    return controls

def transposition(events, controls, offset=0):
    # events [steps, batch_size, event_dim]
    # return events, controls
    from sequence import EventSeq, ControlSeq
    import numpy as np

    events = np.array(events, dtype=np.int64)
    controls = np.array(controls, dtype=np.float32)
    event_feat_ranges = EventSeq.feat_ranges()

    on = event_feat_ranges['note_on']
    off = event_feat_ranges['note_off']

    if offset > 0:
        indeces0 = (((on.start <= events) & (events < on.stop - offset)) |
                    ((off.start <= events) & (events < off.stop - offset)))
        indeces1 = (((on.stop - offset  <= events) & (events < on.stop)) |
                    ((off.stop - offset <= events) & (events < off.stop)))
        events[indeces0] += offset
        events[indeces1] += offset - 12
    elif offset < 0:
        indeces0 = (((on.start - offset <= events) & (events < on.stop)) |
                    ((off.start - offset <= events) & (events < off.stop)))
        indeces1 = (((on.start <= events) & (events < on.start - offset)) |
                    ((off.start <= events) & (events < off.start - offset)))
        events[indeces0] += offset
        events[indeces1] += offset + 12

    assert ((0 <= events) & (events < EventSeq.dim())).all()
    histr = ControlSeq.feat_ranges()['pitch_histogram']
    controls[:, :, histr.start:histr.stop] = np.roll(
                    controls[:, :, histr.start:histr.stop], offset, -1)

    return events, controls
