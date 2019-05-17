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

#TODO try and understand and remove the hard coded values in the forward pass
class CDCK2(nn.Module):
    def __init__(self, timestep):

        super(CDCK2, self).__init__()
        self.timestep = timestep
        self.encoder = nn.Sequential(  # downsampling factor = 160
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 256).cuda()
        else:
            return torch.zeros(1, batch_size, 256)

    def makeTheInputPercUsingSeqLens(self,batchSize,seq_lens):
        input_percentages = torch.FloatTensor(batchSize)
        maxSeqLen = torch.max(seq_lens)
        for i in range(batchSize):
            seqLen = seq_lens[i].item()
            input_percentages[i] = seqLen/float(maxSeqLen)
        return input_percentages,maxSeqLen
    def forward(self, x, hidden,seq_lens):
        batch = x.size()[0]
        seq_perc,maxLen = self.makeTheInputPercUsingSeqLens(batch,seq_lens)

        t_samples_audios = torch.LongTensor([torch.randint(seq_len / 160 - self.timestep, size=(1,)).long() for seq_len in seq_lens])

        max_sample_audios = torch.max(t_samples_audios)
        # t_samples = torch.randint(self.seq_len / 160 - self.timestep, size=(1,)).long()  # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        seqAfterEncoder = z.size(1)
        sizesAfterEncoder = seq_perc.mul_(int(seqAfterEncoder)).int()
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512
        forward_seq_var = torch.zeros((batch, max_sample_audios, 512)).float()
        for batchNum in range(batch):
            t_samples = t_samples_audios[batchNum]
            zForCurrentSample = z[batchNum, :sizesAfterEncoder[batchNum], :]
            for i in np.arange(1, self.timestep + 1):

                encode_samples[i - 1] = zForCurrentSample[ t_samples + i, :].view(1, 512)  # z_tk e.g. size 8*512

            forward_seq_var[batchNum,:,:] = zForCurrentSample[:t_samples + 1, :]
        #original :
        # forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512

        output, hidden = self.gru(forward_seq_var, hidden)  # output size e.g. 8*100*256
        sizesAfterGru = seq_perc.mul_(int(output.size(1))).int()
        c_t_var = torch.zeros((batch, output.size(1), 256)).float()
        for batchNum in range(batch):
            t_samples = t_samples_audios[batchNum]
            outputForItem = output[batchNum,:sizesAfterGru,:]
            c_t = outputForItem[t_samples, :].view(1, 256)  # c_t e.g. size 8*256
            c_t_var[batchNum,:t_samples] = c_t

        pred = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t_var)  # Wk*c_t e.g. size 8*512

        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)  # output size e.g. 8*128*256

        return output, hidden  # return every frame
        # return output[:,-1,:], hidden # only return the last frame per utt

if __name__ == '__main__':
    pass
