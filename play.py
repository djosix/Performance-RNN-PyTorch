info = '''
------------------------------------
Simple MIDI Player & Note Visualizer
------------------------------------

Please install pygame and fluidsynth for Python 3:
    
    pip3 install pygame
    pip3 install git+https://github.com/txomon/pyfluidsynth.git

Usage:

    python3 play.py soundfont_path midi_files...

'''.lstrip()


import os, sys, time, threading
import pygame, fluidsynth
import sequence, pretty_midi
import numpy as np

# pylint: disable=E1101

entities = []
lock = threading.Lock()
done = False

width = 1024
height = 768

#====================================================================
# Display
#====================================================================

class NoteEntity:
    def __init__(self, key, velocity):
        self.done = False
        self.age = 255
        self.color = pygame.color.Color('black')
        self.radius = int(40 * velocity / 128)
        self.velocity = np.array([0., -velocity / 10], dtype=np.float32)
        pr = sequence.DEFAULT_PITCH_RANGE
        x = key * width / 128
        self.position = np.array([x, height], dtype=np.float32)
    
    def update(self):
        self.position += self.velocity
        self.velocity += np.random.randn(2) / 20
        self.velocity *= 0.99
        self.age -= 1
        if self.age == 0:
            self.done = True

    def get_color(self):
        return pygame.color.Color(self.age, self.age, self.age)

    def render(self, screen):
        x, y = self.position.astype(np.int32)
        color = self.get_color()
        pygame.draw.circle(screen, color, (x, y), self.radius)

def add_entitiy(key, velocity):
    global entities, lock
    lock.acquire()
    entities.append(NoteEntity(key, velocity))
    lock.release()    

def display():
    global entities, lock, width, height, done
    pygame.init()
    pygame.display.set_caption('generator')
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(pygame.color.Color('black'))

        lock.acquire()
        entities = [entity for entity in entities if not entity.done]
        for i, entity in enumerate(entities):
            if i + 1 < len(entities):
                next_ent = entities[i + 1]
                pygame.draw.line(screen, entity.get_color(), entity.position, next_ent.position)
        for entity in entities:
            entity.render(screen)
            entity.update()
        lock.release()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()



#====================================================================
# Synth
#====================================================================

def note_repr(key, velocity):
    octave = key // 12 - 1
    name = ['C', 'C#', 'D', 'D#', 'E', 'F',
            'F#', 'G', 'G#', 'A', 'A#', 'B'][key % 12]
    vel = int(10 * velocity / 128)
    return f'({name}{octave} {vel})'


def play(midi_files, sound_font_path):
    global done
    
    fs = fluidsynth.Synth(gain=5)
    fs.start()

    try:
        sfid = fs.sfload(sound_font_path)
        fs.program_select(0, sfid, 0, 0)
    except:
        print('Failed to load', sound_font_path)
        return

    for midi_file in midi_files:

        print(f'Playing {midi_file}')

        try:
            note_seq = sequence.NoteSeq.from_midi_file(midi_file)
            event_seq = sequence.EventSeq.from_note_seq(note_seq)
        except:
            print('Failed to load', midi_file)
            continue
        
        velocity = sequence.DEFAULT_VELOCITY
        velocity_bins = sequence.EventSeq.get_velocity_bins()
        time_shift_bins = sequence.EventSeq.time_shift_bins
        pitch_start = sequence.EventSeq.pitch_range.start

        for event in event_seq.events:
            if event.type == 'note_on':
                key = int(event.value + pitch_start)
                fs.noteon(0, key, velocity)
                print(f' {note_repr(key, velocity)} ', end='', flush=True)
                add_entitiy(key, velocity)
            elif event.type == 'note_off':
                key = int(event.value + pitch_start)
                fs.noteoff(0, key)
            elif event.type == 'time_shift':
                print('.', end='', flush=True)
                time.sleep(time_shift_bins[event.value])
            elif event.type == 'velocity':
                velocity = int(velocity_bins[
                    min(event.value, velocity_bins.size - 1)])

        print('Done')

    done = True


#====================================================================
# Main
#====================================================================


if __name__ == '__main__':

    try:
        sound_font_path = sys.argv[1]
        midi_files = sys.argv[2:]
    except:
        print(info)
        sys.exit()

    assert os.path.isfile(sound_font_path), sound_font_path
    for midi_path in midi_files:
        assert os.path.isfile(midi_path), midi_path

    threading.Thread(target=play,
                     args=(midi_files, sound_font_path),
                     daemon=True).start()

    display()
