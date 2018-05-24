# Performance RNN - PyTorch

PyTorch implementation of Performance RNN, inspired by *Ian Simon and Sageev Oore. "Performance RNN: Generating Music with Expressive
Timing and Dynamics." Magenta Blog, 2017.*
[https://magenta.tensorflow.org/performance-rnn](https://magenta.tensorflow.org/performance-rnn).

This model is not implemented in the official way!


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
│   │       └── *.data (generated with preprocess.py)
│   └── scripts/
│       └── *.sh (dataset download scripts)
├── generated/
│   └── *.mid (generated with generate.py)
└── runs/ (tensorboard logdir)
```


## Getting Started

- Download datasets

    ```
    cd dataset/
    bash scripts/NAME_scraper.sh midi/NAME
    ```

- Preprocessing

    ```shell
    # Will preprocess all MIDI files under dataset/midi/NAME
    python3 preprocess.py dataset/midi/NAME dataset/processed/NAME
    ```

- Training

    ```shell
    # Train on .data files in dataset/processed/MYDATA,
    # and save to myModel.sess every 10s.
    python3 train.py -s myModel.sess -d dataset/processed/MYDATA -i 10
    ```

- Generating

    For usage, see `python3 generate.py -h`.

    ```shell
    Usage: generate.py [options]

    Options:
    -h, --help            show this help message and exit
    -c CONTROL, --control=CONTROL
                            control or a processed data file path, e.g.,
                            "PITCH_HISTOGRAM;NOTE_DENSITY" like
                            "2,0,1,1,0,1,0,1,1,0,0,1;4", or ";3" (which gives all
                            pitches the same probability), or
                            "/path/to/processed/midi/file.data" (uses control
                            sequence from the given processed data)
    -b BATCH_SIZE, --batch-size=BATCH_SIZE
    -s SESS_PATH, --session=SESS_PATH
                            session file containing the trained model
    -o OUTPUT_DIR, --output-dir=OUTPUT_DIR
    -l MAX_LEN, --max-length=MAX_LEN
    -g GREEDY_RATIO, --greedy-ratio=GREEDY_RATIO
    ```

    Generate with control sequence from test.data and model from test.sess:

    ```shell
    python3 generate.py -s test.sess -c test.data
    ```

    Generate with pitch histogram and note density (C major scale).

    ```shell
    python3 generate.py -s test.sess -l 1000 -c '1,0,1,0,1,1,0,1,0,1,0,1;3'
    ```


## Requirements

```
pretty_midi
numpy
pytorch
tensorboardX
progress
```
