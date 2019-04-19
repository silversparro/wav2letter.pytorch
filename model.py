import math
from collections import OrderedDict
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
        magn = magn[:,:,1:,:]
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

class PCEN(nn.Module):
    def __init__(self):
        super(PCEN,self).__init__()

        '''
        initialising the layer param with the best parametrised values i searched on web (scipy using theese values)
        alpha = 0.98
        delta=2
        r=0.5
        '''
        self.log_alpha = Parameter(torch.FloatTensor([0.98]))
        self.log_delta = Parameter(torch.FloatTensor([2]))
        self.log_r     = Parameter(torch.FloatTensor([0.5]))
        self.eps = 0.000001

    def forward(self,x,smoother):
        # t = x.size(0)
        # t = x.size(1)
        # t = x.size(2)
        # t = x.size(3)
        # alpha = self.log_alpha.exp().expand_as(x)
        # delta = self.log_delta.exp().expand_as(x)
        # r     = self.log_r.exp().expand_as(x)
        # print 'updated values are alpha={} , delta={} , r={}'.format(self.log_alpha,self.log_delta,self.log_r)
        smooth = (self.eps + smoother) ** (-(self.log_alpha))
        # pcen = (x/(self.eps + smoother)**alpha + delta)**r - delta**r
        pcen = (x * smooth + self.log_delta)**self.log_r - self.log_delta**self.log_r
        return pcen

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class Cov1dBlock(nn.Module):
    def __init__(self, input_size, output_size, kernal_size, stride, drop_out_prob=-1.0, dilation=1, padding='same',bn=True,activationUse=True):
        super(Cov1dBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernal_size = kernal_size
        self.stride = stride
        self.dilation = dilation
        self.drop_out_prob = drop_out_prob
        self.activationUse = activationUse
        self.padding = kernal_size[0] #(kernal_size[0]-stride)//2 if kernal_size[0]!=1 else
        '''using the below code for the padding calculation'''
        input_rows = input_size
        filter_rows = kernal_size[0]
        effective_filter_size_rows = (filter_rows - 1) * dilation + 1
        out_rows = (input_rows + stride - 1) // stride
        self.rows_odd = False
        if padding=='same':
            self.padding_needed =max(0, (out_rows - 1) * stride + effective_filter_size_rows -
                input_rows)

            self.padding_rows = max(0, (out_rows - 1) * stride +
                               (filter_rows - 1) * dilation + 1 - input_rows)

            self.rows_odd = (self.padding_rows % 2 != 0)

            self.addPaddings = self.padding_rows
        elif padding=='half':
            self.addPaddings = kernal_size[0]
        elif padding == 'invalid':
            self.addPaddings = 0

        self.paddingAdded = nn.ReflectionPad1d(self.addPaddings//2) if self.addPaddings >0 else None
        self.conv1 = nn.Sequential(

            nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernal_size,
                      stride=stride, padding=(0), dilation=dilation),
            # nn.ReLU6()
        )
        self.batchNorm = nn.BatchNorm1d(num_features=output_size,momentum=0.90,eps=0.001) if bn else None
        # self.activation = nn.Hardtanh(min_val=0,max_val=20) if activationUse else None
        # self.activation = nn.ReLU6() if activationUse else None
        # self.activation = True if activationUse else False
        self.drop_out_layer = nn.Dropout(drop_out_prob) if self.drop_out_prob != -1 else None
        # self.activation = nn.ReLU6() if activationUse else None

        # torch.nn.init.xavier_normal(self.conv1._modules['0'].weight)

    def forward(self, xs, hid=None):
        if self.paddingAdded is not None:
            xs = self.paddingAdded(xs)
        output = self.conv1(xs)
        if self.batchNorm is not None:
            output = self.batchNorm(output)
        if self.activationUse:
            output = torch.clamp(input=output,min=0,max=20)
            # output = self.activation(output)
        if self.drop_out_layer is not None:
            output = self.drop_out_layer(output)

        return output

class WaveToLetter(nn.Module):
    def __init__(self,sample_rate,window_size, labels="abc",audio_conf=None,mixed_precision=False):
        super(WaveToLetter, self).__init__()

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
        input_size = int((nfft/2))
        hop_length = sample_rate * self._audio_conf.get("window_stride", 0.01)

        # self.pcen = PCEN()
        self.frontEnd = stft(hop_length=int(hop_length), nfft=int(nfft))
        conv1 =  Cov1dBlock(input_size=input_size,output_size=256,kernal_size=(11,),stride=2,dilation=1,drop_out_prob=0.2,padding='same')
        conv2s = []
        conv2s.append(('conv1d_{}'.format(0),conv1))
        inputSize = 256
        for idx in range(15):
            layergroup = idx//3
            if (layergroup) == 0:
                convTemp = Cov1dBlock(input_size=inputSize,output_size=256,kernal_size=(11,),stride=1,dilation=1,drop_out_prob=0.2,padding='same')
                conv2s.append(('conv1d_{}'.format(idx+1),convTemp))
                inputSize = 256
            elif (layergroup) == 1:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=384, kernal_size=(13,), stride=1, dilation=1,
                                      drop_out_prob=0.2)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize=384
            elif (layergroup) ==2:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=512, kernal_size=(17,), stride=1, dilation=1,
                                      drop_out_prob=0.2)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 512

            elif (layergroup) ==3:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=640, kernal_size=(21,), stride=1, dilation=1,
                                      drop_out_prob=0.3)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 640

            elif (layergroup) ==4:
                convTemp = Cov1dBlock(input_size=inputSize, output_size=768, kernal_size=(25,), stride=1, dilation=1,
                                      drop_out_prob=0.3)
                conv2s.append(('conv1d_{}'.format(idx + 1), convTemp))
                inputSize = 768

        conv1 = Cov1dBlock(input_size=inputSize, output_size=896, kernal_size=(29,), stride=1, dilation=2, drop_out_prob=0.4)
        conv2s.append(('conv1d_{}'.format(16), conv1))
        conv1 = Cov1dBlock(input_size=896, output_size=1024, kernal_size=(1,), stride=1, dilation=1, drop_out_prob=0.4)
        conv2s.append(('conv1d_{}'.format(17), conv1))
        conv1 = Cov1dBlock(input_size=1024, output_size=len(self._labels), kernal_size=(1,),stride=1,bn=False,activationUse=False)
        conv2s.append(('conv1d_{}'.format(18), conv1))

        self.conv1ds = nn.Sequential(OrderedDict(conv2s))
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x):
        x = self.frontEnd(x)
        x = x.squeeze(1)
        x = self.conv1ds(x)
        x = x.transpose(1,2)
        x = self.inference_softmax(x)

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
