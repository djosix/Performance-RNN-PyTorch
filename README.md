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
    ```shell
    python3 generate.py \
        myModel.sess \  # load trained model from myModel.sess
        generated/ \    # save to generated/
        10 \            # generate 10 event sequences
        2000 \          # generate 2000 event steps
        0.9 \           # 90% sampling with argmax and 10% multinomial
        '1,0,1,0,1,1,0,1,0,1,0,1' \ # pitch histogram ([12] or [0])
        3               # note density (0-5)
    ```

## Requirements

```
pretty_midi
numpy
pytorch
tensorboardX
progress
```
