import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from .layers import *
from .utils import *

import numpy as np

class FastWaveNet(nn.Module):
    """Default values found from various sources

    Original WaveNet Tensorflow (https://github.com/ibab/tensorflow-wavenet)
        Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)

    FastWaveNet Tensorflow (https://github.com/tomlepaine/fast-wavenet)
        num_blocks=2,
        num_layers=14,

    Args:
        layers: number of layers per blocks
        blocks: number of blocks
        residual_channels: How many filters to learn for the residual.
        dilation_channels: How many filters to learn for the dilated
            convolution.
        skip_channels: How many filters to learn that contribute to the
            quantized softmax output.
        quantization_channels: mu-law quantization channels.  Essentially the
            number of output classes in the final softmax layer
    Potential Args: not used yet, but perhaps I should
        initial_filter_width: The width of the initial filter of the
            convolution applied to the scalar input. This is only relevant
            if scalar_input=True.

    Author Notes:
        the original paper dilates by a factor of two always.  However, an
        invalid dilation will occur if the dilation width is not a factor of
        the input length.  thus a rotating deque of dilation widths which are
        prime factors of the input length are used instead of 2.

        i.e. 16000 = 2 * 2 * 2 * 2 * 2 * 2 * 5 * 5 * 5

    """
    def __init__(self,
                 blocks=2,
                 layers=10, # depends on number of prime factors of input_len
                 residual_channels=128,
                 dilation_channels=256,
                 skip_channels=128,
                 quantization_channels=256,
                 input_len=8000,
                 audio_channels=1,
                 kernel_size=2):
        super(FastWaveNet, self).__init__()
        # variables
        self.scope_mul = prime_factors(input_len) # used to insure valid dilations
        if layers is None or layers > len(self.scope_mul):
            print("setting # of layers to {}".format(len(self.scope_mul)))
            self.layers = len(self.scope_mul)
        else:
            self.layers = layers
        self.blocks = blocks
        self.audio_channels = audio_channels
        self.residual_channels = residual_channels * audio_channels
        self.dilation_channels = dilation_channels * audio_channels
        self.skip_channels = skip_channels * audio_channels
        self.kernel_size = kernel_size
        self.quantization_channels = quantization_channels

        # debugging
        self.sizes = []

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = []
        self.skip_convs = []

        # filter non-linearity
        self.filter_act = F.tanh

        # non-linearity in "pre-softmax" layers
        self.nl_out = nn.ReLU()
        self.nl_end = nn.ReLU()
        #self.nl1 = nn.SELU
        #self.nl1 = nn.Hardtanh
        #self.nl2 = nn.PReLU(self.quantization_channels)

        # initial convolution
        self.conv0 = Conv1dExt(in_channels=audio_channels,
                               out_channels=self.residual_channels,
                               kernel_size=1,
                               bias=False)
        # convolution out of blocks
        self.conv_out = Conv1dExt(in_channels=self.skip_channels,
                                  out_channels=self.quantization_channels * self.audio_channels,
                                  kernel_size=1,
                                  bias=True)
        # final convolution before leaving the network
        self.conv_end = Conv1dExt(in_channels=self.quantization_channels * self.audio_channels,
                                  out_channels=self.quantization_channels * self.audio_channels,
                                  kernel_size=1,
                                  bias=False)
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for l in range(self.layers):
                self.dilations.append((new_dilation, init_dilation))
                self.dilated_queues.append(DilatedQueue(max_length=int((kernel_size - 1) * new_dilation + 1),
                                                        num_channels=self.residual_channels,
                                                        dilation=new_dilation))
                self.filter_convs.append(Conv1dExt(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=False))
                self.gate_convs.append(Conv1dExt(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=False))
                self.residual_convs.append(Conv1dExt(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=1,
                                                     bias=False))
                self.skip_convs.append(Conv1dExt(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=1,
                                                 bias=False))
                receptive_field += additional_scope
                #print("receptive field (layer: {}, block: {}): {}".format(l+1, b+1, receptive_field))
                additional_scope *= self.scope_mul[0]
                init_dilation = new_dilation
                new_dilation *= self.scope_mul[0]
                self.scope_mul.rotate(-1)

        # define deps
        self.conv0.input_tied_modules.append(self.filter_convs[0])
        for i in range(int(blocks*self.layers)):

            self.filter_convs[i].input_tied_modules.append(self.residual_convs[i])
            self.filter_convs[i].input_tied_modules.append(self.skip_convs[i])
            self.filter_convs[i].output_tied_modules.append(self.gate_convs[i])

            self.gate_convs[i].input_tied_modules.append(self.residual_convs[i])
            self.gate_convs[i].input_tied_modules.append(self.skip_convs[i])
            self.gate_convs[i].output_tied_modules.append(self.filter_convs[i])

            self.skip_convs[i].input_tied_modules.append(self.conv_out)
            self.skip_convs[i].output_tied_modules = [skip for ind, skip in enumerate(self.skip_convs) if ind != i]
            if i < blocks*self.layers-1:
                # final layer
                self.residual_convs[i].input_tied_modules.append(self.filter_convs[i + 1])
                self.residual_convs[i].input_tied_modules.append(self.gate_convs[i + 1])
            if i > 0:
                # all but first layer
                self.residual_convs[i].output_tied_modules.append(self.residual_convs[i-1])
                self.residual_convs[i].output_tied_modules.append(self.filter_convs[i-1])
                self.residual_convs[i].output_tied_modules.append(self.gate_convs[i-1])
                self.residual_convs[i].input_tied_modules.append(self.skip_convs[i-1])
                self.residual_convs[i].input_tied_modules.append(self.filter_convs[i])
                self.residual_convs[i].input_tied_modules.append(self.gate_convs[i])
    def forward(self, input):
        ob, oc, ol = input.size()
        self.sizes.append(input.size())

        x = self.conv0(input)
        self.sizes.append(x.size())
        skip = 0

        for i in range(int(self.blocks*self.layers)):
            (dil, init_dil) = self.dilations[i]
            res, pad_res = dilate(x, dil)

            # dilation Convolutions
            if pad_res == 0:
                res = pad1d(res,(1, 0, 0, 0),pad_value=0)
            else:
                print("pad_res: {}".format(pad_res))
                pass
            fil = self.filter_convs[i](res)
            fil = self.filter_act(fil)

            gate = self.gate_convs[i](res)
            gate = F.sigmoid(gate)
            self.sizes.append(x.size())

            x = fil * gate
            self.sizes.append(x.size())

            s = x
            if x.size(2) != 1:
                s, pad_skip = dilate(x, self.audio_channels)
            s = self.skip_convs[i](s)

            try:
                # this is designed to remove the front padding
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0

            skip = s + skip # Note the skip is ultimately part of the output
            self.sizes.append(("skip size:",)+skip.size())

            x = self.residual_convs[i](x)
            x = x + res[:, :, (self.kernel_size - 1):]
            self.sizes.append(x.size())

        # the multiple non-linearities modeled after tensorflow version
        x = self.nl_out(skip)
        x = self.conv_out(x)
        self.sizes.append("last conv before resize")
        self.sizes.append(x.size())
        x, _ = dilate(x, self.audio_channels) # only works with 1 channel for now
        x = self.nl_end(x)
        x = self.conv_end(x)
        self.sizes.append(x.size())

        return x

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
