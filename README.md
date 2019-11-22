# wav2Letter.pytorch

Implementation of Wav2Letter using [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc).
Creates a network based on the [Wav2Letter](https://arxiv.org/abs/1609.03193) architecture, trained with the CTC activation function.

Currently Tested on pytorch [1.3.0] with cuda10.1 and python3.7.

Branch trainableFrontEnd : contains the code in progress to train the model using the raw audio samples only.

Branch  python27 : contains the same code as of master but for python2.7 and pytorch0.4.1

## Features

* Train Wav2Letter.
* Language model support using kenlm.
* Noise injection for online training to improve noise robustness.
* Audio augmentation to improve noise robustness.
* Easy start/stop capabilities in the event of crash or hard stop during training.
* Visdom/Tensorboard support for visualizing training graphs.

# Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in
an Anaconda installation on Ubuntu.

Install [PyTorch](https://github.com/pytorch/pytorch#installation) if you haven't already.

Install this fork for Warp-CTC bindings:
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding
python setup.py install
```

Install pytorch audio:
```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
pip install cffi
python setup.py install
```

If you want decoding to support beam search with an optional language model, install ctcdecode:
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

Finally clone this repo and run this within the repo:
```
pip install -r requirements.txt
```

# Usage

### Custom Dataset

To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,transcription
/path/to/audio2.wav,transcription
...
```

The first path is to the audio file, and the second is the text containing the transcript on one line. This can then be used as stated below.

## Training

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualize training. Once a server has been started, to use:

```
python train.py --visdom
```

There is also [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) support to visualize training. Follow the instructions to set up. To use:

```
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```

For both visualisation tools, you can add your own name to the run by changing the `--id` parameter when training.

## Testing

For testing write all the file path into a csv and run
```
python test.py
```
PS : for speed improvements try to run test.py with the flag '--fuse-layers'. This option will fuse all the conv-bn operation and increase the model inference speed.

### Noise Augmentation/Injection

There is support for two different types of noise; noise augmentation and noise injection.

#### Noise Augmentation

Applies small changes to the tempo and gain when loading audio to increase robustness. To use, use the `--augment` flag when training.

#### Noise Injection

Dynamically adds noise into the training data to increase robustness. To use, first fill a directory up with all the noise files you want to sample from.
The dataloader will randomly pick samples from this directory.

To enable noise injection, use the `--noise-dir /path/to/noise/dir/` to specify where your noise files are. There are a few noise parameters to tweak, such as
`--noise_prob` to determine the probability that noise is added, and the `--noise-min`, `--noise-max` parameters to determine the minimum and maximum noise to add in training.

Included is a script to inject noise into an audio file to hear what different noise levels/files would sound like. Useful for curating the noise dataset.

```
python noise_inject.py --input-path /path/to/input.wav --noise-path /path/to/noise.wav --output-path /path/to/input_injected.wav --noise-level 0.5 # higher levels means more noise
```

### Checkpoints

Training supports saving checkpoints of the model to continue training from should an error occur or early termination. To enable epoch
checkpoints use:

```
python train.py --checkpoint
```

To enable checkpoints every N batches through the epoch as well as epoch saving:

```
python train.py --checkpoint --checkpoint-per-batch N # N is the number of batches to wait till saving a checkpoint at this batch.
```

Note for the batch checkpointing system to work, you cannot change the batch size when loading a checkpointed model from it's original training
run.

To continue from a checkpointed model that has been saved:

```
python train.py --continue-from models/wav2Letter_checkpoint_epoch_N_iter_N.pth.tar
```

This continues from the same training state as well as recreates the visdom graph to continue from if enabled.

If you would like to start from a previous checkpoint model but not continue training, add the `--finetune` flag to restart training
from the `--continue-from` weights.

### Choosing batch sizes

Included is a script that can be used to benchmark whether training can occur on your hardware, and the limits on the size of the model/batch
sizes you can use. To use:

```
python benchmark.py --batch-size 32
```

Use the flag `--help` to see other parameters that can be used with the script.

### Model details

Saved models contain the metadata of their training process. To see the metadata run the below command:

```
python model.py --model-path models/wav2Letter.pth.tar
```

To also note, there is no final softmax layer on the model as when trained, warp-ctc does this softmax internally. This will have to also be implemented in complex decoders if anything is built on top of the model, so take this into consideration!

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```
python test.py --model-path models/wav2Letter.pth --test-manifest /path/to/test_manifest.csv --cuda
```

### Alternate Decoders
By default, `test.py`  use a `GreedyDecoder` which picks the highest-likelihood output label at each timestep. Repeated and blank symbols are then filtered to give the final output.

A beam search decoder can optionally be used with the installation of the `ctcdecode` library as described in the Installation section. The `test` and `transcribe` scripts have a `--decoder` argument. To use the beam decoder, add `--decoder beam`. The beam decoder enables additional decoding parameters:
- **beam_width** how many beams to consider at each timestep
- **lm_path** optional binary KenLM language model to use for decoding
- **alpha** weight for language model
- **beta** bonus weight for words

### Time offsets

Use the `--offsets` flag to get positional information of each character in the transcription when using `transcribe.py` script. The offsets are based on the size
of the output tensor, which you need to convert into a format required.
For example, based on default parameters you could multiply the offsets by a scalar (duration of file in seconds / size of output) to get the offsets in seconds.

## Acknowledgements

This work is inspired from the [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) repository of [Sean Naren](https://github.com/SeanNaren). 
This work was done as a part of [Silversparro](https://www.silversparro.com) project work regarding speech to text. 
