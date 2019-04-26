import math
from collections import OrderedDict
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from wavenet import WaveNet
from torch.nn.parameter import Parameter

class VAE(nn.Module):
    def __init__(self, n_in, n_out, n_sam_per_datapoint=1, bias=True):
        '''n_sam_per_datapoint is L from equation 7,
        https://arxiv.org/pdf/1312.6114.pdf'''
        super(VAE, self).__init__()
        self.linear = nn.Conv1d(n_in, n_out * 2, 1, bias=bias)
        self.n_sam_per_datapoint = n_sam_per_datapoint

        # Cache these values for later access by the objective function
        self.mu = None
        self.sigma = None

    def forward(self, x):
        # B, T, I, C: n_batch, n_timesteps, n_in_chan, n_out_chan
        # L: n_sam_per_datapoint
        # Input: (B, I, T)
        # Output: (B * L, C, T)
        mu_sigma = self.linear(x)
        n_out_chan = mu_sigma.size(1) // 2
        mu = mu_sigma[:,:n_out_chan,:]  # first half of the channels
        sigma = mu_sigma[:,n_out_chan:,:] # second half of the channels

        L = self.n_sam_per_datapoint
        sample_sz = (mu.size()[0] * L,) + mu.size()[1:]
        if L > 1:
            sigma = sigma.repeat(L, 1, 1)
            mu = mu.repeat(L, 1, 1)

        # epsilon is the randomness injected here
        epsilon = mu.new_empty(sample_sz).normal_()
        samples = sigma * epsilon + mu
        # Cache mu and sigma for objective function later
        self.mu, self.sigma = mu, sigma

        return samples


class Jitter(nn.Module):
    '''Time-jitter regularization.  With probability [p, (1-2p), p], replace
    element i with element [i-1, i, i+1] respectively.  Disallow a run of 3
    identical elements in the output.  Let p = replacement probability, s =
    "stay probability" = (1-2p).

    tmp[i][j] = Categorical(a, b, c)
    encodes P(x_t|x_(t-1), x_(t-2))
    a 2nd-order Markov chain which generates a sequence in alphabet {0, 1, 2}.

    The following meanings hold:

    0: replace element with previous
    1: do not replace
    2: replace element with following

    For instance, suppose you have:
    source sequence: ABCDEFGHIJKLM
    jitter sequence: 0112021012210
    output sequence: *BCEDGGGIKLLL

    The only triplet that is disallowed is 012, which causes use of the same source
    element three times in a row.  So, P(x_t=0|x_(t-2)=2, x_(t-1)=1) = 0 and is
    renormalized.  Otherwise, all conditional distributions have the same shape,
    [p, (1-2p), p].

    Jitter has a "receptive field" of 3, and it is unpadded.  Our index mask will be
    pre-constructed to have {0, ..., n_win

    '''

    def __init__(self, replace_prob):
        '''n_win gives number of
        '''
        super(Jitter, self).__init__()
        p, s = replace_prob, (1 - 2 * replace_prob)
        tmp = torch.Tensor([p, s, p]).repeat(3, 3, 1)
        tmp[2][1] = torch.Tensor([0, s / (p + s), p / (p + s)])
        self.cond2d = [[dist.Categorical(tmp[i][j]) for i in range(3)] for j in range(3)]
        self.mindex = None
        self.adjust = None

    def gen_mask(self):
        '''populates a tensor mask to be used for jitter, and sends it to GPU for
        next window'''
        n_batch = self.mindex.shape[0]
        n_time = self.mindex.shape[1] - 1
        self.mindex[:, 0:2] = 1
        for b in range(n_batch):
            # The Markov sampling process
            for t in range(2, n_time):
                self.mindex[b, t] = \
                    self.cond2d[self.mindex[b, t - 2]][self.mindex[b, t - 1]].sample()
            self.mindex[b, n_time] = 1

        # adjusts so that temporary value of mindex[i] = {0, 1, 2} imply {i-1,
        # i, i+1} also, first and last elements of mindex mean 'do not replace
        # the element with previous or next, but choose the existing element.
        # This prevents attempting to replace the first element of the input
        # with a non-existent 'previous' element, and likewise with the last
        # element.
        self.mindex += self.adjust

        # Will this play well with back-prop?

    def forward(self, x):
        '''Input: (B, I, T)'''
        n_batch = x.shape[0]
        if self.mindex is None:
            n_time = x.shape[2]
            self.mindex = x.new_empty(n_batch, n_time + 1, dtype=torch.long)
            self.adjust = torch.arange(n_time + 1, dtype=torch.long,
                                       device=x.device).repeat(n_batch, 1) - 2

        self.gen_mask()

        assert x.shape[2] == self.mindex.shape[1] - 1
        y = x.new_empty(x.shape)
        for b in range(n_batch):
            y[b] = torch.index_select(x[b], 1, self.mindex[b, 1:])
        return y

class VAEImpl(nn.Module):
    def __init__(self, n_in, n_out):
        '''n_sam_per_datapoint is L from equation 7,
        built from understanding of
        https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd'''
        super(VAEImpl, self).__init__()

        self.fc1 = nn.Conv1d(n_in, n_out*2, 1)
        self.fc_bn1 = nn.BatchNorm1d(n_out*2)
        self.fc21 = nn.Conv1d(n_out*2, n_out*2, 1)
        self.fc22 = nn.Conv1d(n_out*2, n_out*2, 1)

    def forward(self, x):

        fc1 = self.relu(self.fc_bn1(self.fc1(x)))

        mu = self.fc21(fc1)
        std = self.fc22(fc1)

        std = std.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        hiddenRepr = eps.mul(std).add_(mu)

        return hiddenRepr

class stft(nn.Module):
    def __init__(self, nfft=1024, hop_length=512, window="hanning"):
        super(stft, self).__init__()
        assert nfft % 2 == 0

        self.hop_length = hop_length
        self.n_freq = n_freq = nfft//2 + 1

        self.real_kernels, self.imag_kernels = _get_stft_kernels(nfft, window)
        self.real_kernels_size = self.real_kernels.size()
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.real_kernels_size[0], kernel_size=(self.real_kernels_size[2], self.real_kernels_size[3]), stride=(self.hop_length)),
            nn.BatchNorm2d(self.real_kernels_size[0]),
            nn.Hardtanh(0, 20, inplace=True)
        )
    def forward(self, sample):
        sample = sample.unsqueeze(1)
        magn = self.conv(sample)

        magn = magn.permute(0, 2, 1, 3)
        return magn

def _get_stft_kernels(nfft, window):
    nfft = int(nfft)
    assert nfft % 2 == 0

    def kernel_fn(freq, time):
        return np.exp(-1j * (2 * np.pi * time * freq) / float(nfft))

    kernels = np.fromfunction(kernel_fn, (nfft//2+1, nfft), dtype=np.float64)

    if window == "hanning":
        win_cof = scipy.signal.get_window("hanning", nfft)[np.newaxis, :]
    else:
        win_cof = np.ones((1, nfft), dtype=np.float64)

    kernels = kernels[:, np.newaxis, np.newaxis, :] * win_cof

    real_kernels = nn.Parameter(torch.from_numpy(np.real(kernels)).float())
    imag_kernels = nn.Parameter(torch.from_numpy(np.imag(kernels)).float())

    return real_kernels, imag_kernels

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_

class ConEncoder(nn.Module):
    def __init__(self, input_size, output_size, kernal_size, stride, drop_out_prob=-1.0,
                 bn=True,activationUse=True,residual=False):
        super(ConEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernal_size = kernal_size
        self.stride = stride
        self.residual = residual
        self.drop_out_prob = drop_out_prob
        self.activationUse = activationUse #(kernal_size[0]-stride)//2 if kernal_size[0]!=1 else
        '''using the below code for the padding calculation'''
        self.conv1 = nn.Sequential(

            nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernal_size,
                      stride=stride,padding=0),
        )
        self.batchNorm = nn.BatchNorm1d(num_features=output_size,momentum=0.90,eps=0.001) if bn else None
        self.drop_out_layer = nn.Dropout(drop_out_prob) if self.drop_out_prob != -1 else None

    def forward(self, xs, hid=None):
        output = self.conv1(xs)
        if self.batchNorm is not None:
            output = self.batchNorm(output)
        if self.activationUse:
            output = torch.clamp(input=output,min=0,max=20)
        if self.residual:
            output = xs+output
        if self.drop_out_layer is not None:
            output = self.drop_out_layer(output)

        return output

class AutoEncoder(nn.Module):
    def __init__(self,sample_rate,window_size, labels="abc",audio_conf=None,mixed_precision=False):
        super(AutoEncoder, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._sample_rate=sample_rate
        self._window_size=window_size
        self.mixed_precision=mixed_precision

        nfft = (self._sample_rate * self._window_size)
        input_size = 1+int((nfft/2))
        hop_length = sample_rate * self._audio_conf.get("window_stride", 0.01)
        kernal = [3, 3, 4, 3, 3, 1, 1, 1, 1]
        strides = [1, 1, 2, 1, 1, 1, 1, 1, 1]
        stack_residual = [False, True, False, True, True, True, True, True, True]
        self.frontEnd = stft(hop_length=int(hop_length), nfft=int(nfft))
        conv1ds = []

        for idx,kernal_size,stride,residual in enumerate(zip(kernal,strides,stack_residual)):

            convTemp = ConEncoder(input_size=input_size,output_size=768,kernal_size=(kernal_size,),stride=stride,drop_out_prob=0.2,residual=residual)
            conv1ds.append(('encoder_conv1d_{}'.format(idx),convTemp))

        self.conv1ds = nn.Sequential(OrderedDict(conv1ds))
        self.bottleNeck = VAEImpl(768,64)
        self.jitter = Jitter(0.12)
        self.decoder = WaveNet(num_time_samples=2,num_channels=64,num_blocks=2,num_layers=10,num_hidden=10)

    def forward(self, x):
        x = self.frontEnd(x)
        x = x.squeeze(1)
        x = self.conv1ds(x)
        x = self.bottleNeck(x)
        x = self.jitter(x)
        x = self.decoder(x)
        x = x.transpose(1,2)
        return x

    @classmethod
    def load_model(cls, path, cuda=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(labels=package['labels'], audio_conf=package['audio_conf'],sample_rate=package["sample_rate"]
                    ,window_size=package["window_size"],mixed_precision=package.get('mixed_precision',False))
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']

        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        # keyNames = package['state_dict'].keys()
        #
        # for keyname in keyNames:
        #     if "num_batches_tracked" in keyname:
        #         del package['state_dict'][keyname]
        model.load_state_dict(package['state_dict'])
        # for x in model.rnns:
        #     x.flatten_parameters()
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, cuda=False):
        model = cls(labels=package['labels'], audio_conf=package['audio_conf'],sample_rate=package["sample_rate"]
                    ,window_size=package["window_size"],mixed_precision=package.get('mixed_precision',False))
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        # model_is_cuda = next(model.parameters()).is_cuda
        # model = model.module if model_is_cuda else model
        package = {
            'version': model._version,
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'mixed_precision': model.mixed_precision
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels
    @staticmethod
    def get_sample_rate(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._sample_rate if model_is_cuda else model._sample_rate
    @staticmethod
    def get_window_size(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._window_size if model_is_cuda else model._window_size
    @staticmethod
    def setAudioConfKey(model,key,value):
        model._audio_conf[key] = value
        return model
    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._audio_conf if model_is_cuda else model._audio_conf

    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version
        }
        return meta

if __name__ == '__main__':
    pass
