![generated-sample-sheet-music](https://user-images.githubusercontent.com/17045050/42017029-3b4f7060-7ae0-11e8-829b-6d6b8b829759.png)

# Performance RNN - PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of Performance RNN, inspired by *Ian Simon and Sageev Oore. "Performance RNN: Generating Music with Expressive
Timing and Dynamics." Magenta Blog, 2017.*
[https://magenta.tensorflow.org/performance-rnn](https://magenta.tensorflow.org/performance-rnn).

This model is not implemented in the official way!

## Generated Samples

- [Random C Major MIDI](https://drive.google.com/open?id=1mZtkpsu1yA8oOkE_1b2jyFsvCW70FiKU)
- [Random C Major MP3](https://drive.google.com/open?id=1UqyJ9e58AOimFeY1xoCPyedTz-g2fUxv)
- [Random C Minor MIDI](https://drive.google.com/open?id=1lIVCIT7INuTa-HKrgPzewrgCbgwCRRa1)
- [Random C Minor MP3](https://drive.google.com/open?id=1pVg3Mg2pSq8VHJRJrgNUZybpsErjzpjF)

## Directory Structure

```
.
├── dataset/
│   ├── midi/
│   │   ├── dataset1/
│   │   │   └── *.mid
│   │   └── dataset2/
│   │       └── *.mid
│   ├── processed/
│   │   └── dataset1/
│   │       └── *.data (preprocess.py)
│   └── scripts/
│       └── *.sh (dataset download scripts)
├── output/
│   └── *.mid (generate.py)
├── save/
│   └── *.sess (train.py)
└── runs/ (tensorboard logdir)
```


## Instructions

- Download datasets

    ```shell
    cd dataset/
    bash scripts/NAME_scraper.sh midi/NAME
    ```

- Preprocessing

    ```shell
    # Preprocess all MIDI files under dataset/midi/NAME
    python3 preprocess.py dataset/midi/NAME dataset/processed/NAME
    ```

- Training

    ```shell
    # Train on .data files in dataset/processed/MYDATA, and save to save/myModel.sess every 10s
    python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -i 10

    # Or...
    python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -p hidden_dim=1024
    python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -b 128 -c 0.3
    python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -w 100 -S 10
    ```

- Generating

    ```shell
    # Generate with control sequence from test.data and model from save/test.sess
    python3 generate.py -s save/test.sess -c test.data

    # Generate with pitch histogram and note density (C major scale)
    python3 generate.py -s save/test.sess -l 1000 -c '1,0,1,0,1,1,0,1,0,1,0,1;3'

    # Or...
    python3 generate.py -s save/test.sess -l 1000 -c ';3' # uniform pitch histogram
    python3 generate.py -s save/test.sess -l 1000 # no control

    # Use control sequence from processed data
    python3 generate.py -s save/test.sess -c dataset/processed/some/processed.data
    ```
    
    ![generated-sample-1](https://user-images.githubusercontent.com/17045050/42017026-37dfd7b2-7ae0-11e8-99a9-75d27510f44b.png)
    
    ![generated-sample-2](https://user-images.githubusercontent.com/17045050/42017017-337ce0a2-7ae0-11e8-8193-12ea539af424.png)

## Pretrained Model

- [ecomp.sess - trained with MIDI files from International Piano-e-Competition](https://drive.google.com/open?id=1daT6XRQUTS6AQ5jyRPqzowXia-zVqg6m)

## Requirements

- pretty_midi
- numpy
- pytorch
- tensorboardX
- progress
