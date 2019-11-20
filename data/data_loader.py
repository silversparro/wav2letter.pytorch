from __future__ import division
import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import scipy.signal
import torch
from scipy.io.wavfile import read
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import scipy.io.wavfile as wave

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def load_audio(path):
    sample_rate, sound = read(path)
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


def normalize_tf_data(signal):
    return signal / np.max(np.abs(signal))

def load_audioW2l2(path):
    # sound, _ = torchaudio.load(path)
    sample_freq, sound = wave.read(path)
    sound = (normalize_tf_data(sound.astype(np.float32)) * 32767.0).astype(
        np.int16)
    return sound

def pcen2(E, sr=22050, hop_length=512, t=0.395,eps=0.000001,alpha=0.98,delta=2.0,r=0.5):

    s = 1 - np.exp(- float(hop_length) / (t * sr))
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r
    # return M

def split_normalize_with_librosa(
        audio, top_db=50, frame_length=1024, hop_length=256,
        skip_idx=0):

    edges = librosa.effects.split(audio,
                                  top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    new_audio = np.zeros_like(audio)
    for idx, (start, end) in enumerate(edges[skip_idx:]):
        segment = audio[start:end]
        if start==end:
            print ("Warning: splitting in librosa resulted in an empty segment")
            continue
        new_audio[start:end] = librosa.util.normalize(segment)

    return new_audio


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        self.paths = path is not None and librosa.util.find_files(path)
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_src = load_audio(noise_path)
        noise_offset_fraction = np.random.rand()
        noise_dst = np.zeros_like(data)
        src_offset = int(len(noise_src) * noise_offset_fraction)
        src_left = len(noise_src) - src_offset
        dst_offset = 0
        dst_left = len(data)
        while dst_left > 0:
            copy_size = min(dst_left, src_left)
            np.copyto(noise_dst[dst_offset:dst_offset + copy_size],
                      noise_src[src_offset:src_offset + copy_size])
            if src_left > dst_left:
                dst_left = 0
            else:
                dst_left -= copy_size
                dst_offset += copy_size
                src_left = len(noise_src)
                src_offset = 0
        data += noise_level * noise_dst
        return data

class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, peak_normalization=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.peak_normalization = peak_normalization
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'],
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = (load_audio(audio_path) * 32767.0).astype(
                np.float32)
            # y = load_audio(audio_path).astype(np.float32)
        if self.peak_normalization:
            y = split_normalize_with_librosa(y)

        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)

        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        pcenResult = pcen2(E=spect,sr=self.sample_rate,hop_length=hop_length)

        spect = np.log1p(spect)
        # spect = torch.FloatTensor(spect)
        # pcenResult = torch.FloatTensor(pcenResult)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            # spect.add_(-mean)
            # spect.div_(std)
            spect = np.add(spect,-mean)
            spect = spect/std
            meanPcen = pcenResult.mean()
            stdPcen = pcenResult.std()
            # spect.add_(-mean)
            # spect.div_(std)
            pcenResult = np.add(pcenResult, -meanPcen)
            pcenResult = pcenResult / stdPcen

        return spect ,pcenResult

    def parse_audio_w2l2(self, audio_path):
        if self.augment:
            # y = load_randomly_augmented_audio(audio_path, self.sample_rate)
            y = (load_randomly_augmented_audio(audio_path, self.sample_rate,w2l2=True)).astype(np.float32)
        else:
            y = load_audio(audio_path)
        if self.peak_normalization:
            y = split_normalize_with_librosa(y)

        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)

        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        pcenResult = pcen2(E=spect, sr=self.sample_rate, hop_length=hop_length)

        spect = np.log1p(spect)
        mean = spect.mean()
        std = spect.std()
        spect = np.add(spect, -mean)
        spect = spect / std
        meanPcen = pcenResult.mean()
        stdPcen = pcenResult.std()
        # spect.add_(-mean)
        # spect.div_(std)
        pcenResult = np.add(pcenResult, -meanPcen)
        pcenResult = pcenResult / stdPcen
        return spect,pcenResult

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False,peak_normalization=False, augment=False,w2l2=True):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        # ids.sort(key=self.sortAccTosizeOfAudio)
        self.ids = ids
        self.size = len(ids)
        self.w2l2 = w2l2
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, peak_normalization, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcriptLoaded = sample[0], sample[-1]
        transcriptToUse=transcriptLoaded
        if self.w2l2:
            spect,magnitudeOfAudio = self.parse_audio_w2l2(audio_path)
        else:
            spect, magnitudeOfAudio = self.parse_audio(audio_path)
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcriptToUse)]))
        return spect, transcript, magnitudeOfAudio, audio_path, transcriptToUse

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size

    def sortAccToLengthOFTranscription(self,elem):
        return len(elem[-1].split(' '))

    def sortAccTosizeOfAudio(self,elem):
        return os.stat(elem[0]).st_size

def _collate_fn(batch):
    def func(p):
        return p[0].shape[1]

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.shape[0]
    minibatch_size = len(batch)
    max_seqlength = longest_sample.shape[1]
    inputs = torch.zeros(minibatch_size, freq_size,max_seqlength)
    # inputs = torch.zeros(minibatch_size, max_seqlength,freq_size)
    inputsMags = torch.zeros(minibatch_size, freq_size,max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    inputFilePathAndTranscription = []

    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        tensorMag = sample[2]
        tensorPath = sample[3]
        orignalTranscription = sample[4]
        seq_length = tensor.shape[1]
        tensorShape = tensor.shape
        # tensorMagShape = tensorMag.shape
        tensorNew = np.pad(tensor,((0,0),(0,abs(tensorShape[1]-max_seqlength))),'wrap')
        # tensorNew = np.pad(tensor,((0,abs(tensorShape[0]-max_seqlength)),(0,0)),'constant', constant_values=(0))
        if tensorMag is not None:
            tensorMagNew = np.pad(tensorMag,((0,0),(0,abs(tensorShape[1]-max_seqlength))),'wrap')
            inputsMags[x].copy_(torch.FloatTensor(tensorMagNew))

        # inputs[x].narrow(0, 1, max_seqlength).copy_(torch.FloatTensor(tensorNew))
        inputs[x].copy_(torch.FloatTensor(tensorNew))
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
        sumValueForInput = 0#sum(sum(tensor))
        sumValueForInputMag = 0#sum(sum(tensorMag))
        inputFilePathAndTranscription.append([tensorPath,orignalTranscription,sumValueForInput,sumValueForInputMag,tensorShape[0]])
    numChars = len(targets)
    targets = torch.IntTensor(targets)
    # inputFilePath = torch.IntTensor(inputFilePath)
    return inputs, targets, input_percentages, target_sizes, inputFilePathAndTranscription, inputsMags

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]


def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain,w2l2):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        if w2l2:
            y = load_audioW2l2(augmented_filename).astype(np.float32)
        else:
            y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8),w2l2=False):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value,w2l2=w2l2)
    return audio
