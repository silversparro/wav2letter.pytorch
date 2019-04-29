import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

'''based on code from: https://github.com/vincentherrmann/pytorch-wavenet
    also see:
        https://github.com/musyoku/wavenet
        https://github.com/ibab/tensorflow-wavenet
        https://github.com/tomlepaine/fast-wavenet


'''

class Conv1dExt(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1dExt, self).__init__(*args, **kwargs)
        self.init_ncc()
        self.input_tied_modules = [] # modules whose inputs sensitive to output size
        self.output_tied_modules = [] # modules whose outputs sensitive to input size

    def init_ncc(self):
        w = self.weight.view(self.weight.size(0), -1) # (G, F*J) what are these?
        #mean = torch.mean(w, dim=1).unsqueeze(1).expand_as(w)
        mean = torch.mean(w, dim=1).unsqueeze(1) # 0.2 broadcasting
        self.t0_factor = w - mean
        self.t0_norm = torch.norm(w, p=2, dim=1) # p=2 is the L2 norm
        self.start_ncc = Variable(torch.zeros(self.out_channels))
        self.start_ncc = self.normalized_cross_correlation()

    def normalized_cross_correlation(self):
        w = self.weight.view(self.weight.size(0), -1)
        t_norm = torch.norm(w, p=2, dim=1)
        if self.in_channels == 1 & sum(self.kernel_size) == 1:
            ncc = w.squeeze() / torch.norm(self.t0_norm, p=2)
            ncc = ncc - self.start_ncc
            return ncc
        #mean = torch.mean(w, dim=1).unsqueeze(1).expand_as(w)
        mean = torch.mean(w, dim=1).unsqueeze(1) # 0.2 broadcasting
        t_factor = w - mean
        h_product = self.t0_factor * t_factor
        cov = torch.sum(h_product, dim=1) # (w.size(1) - 1)
        # had normalization code commented out
        denom = self.t0_norm * t_norm

        ncc = cov / denom
        ncc = ncc - self.start_ncc
        return ncc

    def split_output_channel(self, channel_i):
        '''Split one output channel (a feature) into two, but retain summed value

            Args:
                channel_i: (int) number of channel to be split.  the ith channel
        '''

        # weight tensor: (out_channels, in_channels, kernel_size)
        self.out_channels += 1

        orig_weight = self.weight.data
        split_pos = 2 * torch.rand(self.in_channels, self.kernel_size[0])

        new_weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0])
        if channel_i > 0:
            new_weight[:channel_i, :, :] = orig_weight[:channel_i,:, :]
        new_weight[channel_i, :, :] = orig_weight[channel_i, :, :] * split_pos
        new_weight[channel_i + 1, :, :] = orig_weight[channel_i, :, :] * (2 - split_pos)
        if channel_i + 2 < self.out_channels:
            new_weight[channel_i + 2, :, :] = orig_weight[channel_i+1, :, :]
        if self.bias is not None:
            orig_bias = self.bias.data
            new_bias = torch.zeros(self.out_channels)
            new_bias[:(channel_i + 1)] = orig_bias[:(channel_i + 1)]
            new_bias[(channel_i + 1):] = orig_bias[channel_i:] # why no +1?
            self.bias = Parameter(new_bias)

        self.weight = Parameter(new_weight)
        self.init_ncc()

    def split_input_channel(self, channel_i):

        if channel_i > self.in_channels:
            print("cannot split channel {} of {}".format(channel_i, self.in_channels))
            return

        self.in_channels += 1
        orig_weight = self.weight.data
        dup_slice = orig_weight[:, channel_i, :] * .5

        new_weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0])
        if channel_i > 0:
            new_weight[:, :channel_i, :] = orig_weight[:, :channel_i, :]
        new_weight[:, channel_i, :] = dup_slice
        new_weight[:, channel_i + 1, :] = dup_slice
        if channel_i + 1 < self.in_channels:
            new_weight[:, channel_i + 2, :] = orig_weight[:, channel_i + 1, :]
        self.weight = Parameter(new_weight)
        self.init_ncc()

    def split_feature(self, feature_i):
        '''Splits feature in output and input channels

            Args:
                feature_i: (int)
        '''
        self.split_output_channel(channel_i=feature_i)
        for dep in self.input_tied_modules:
            dep.split_input_channel(channel_i=feature_i)
        for dep in self.output_tied_modules:
            dep.split_output_channel(channel_i=feature_i)

    def split_features(self, threshold):
        '''Decides which features to split if they are below a specific threshold

            Args:
                threshold: (float?) less than 1.
        '''
        ncc = self.normalized_cross_correlation()
        for i, ncc_val in enumerate(ncc):
            if ncc_val < threshold:
                print("ncc (feature {}): {}".format(i, ncc_val))
                self.split_feature(i)

class DilatedQueue:
    '''This is the queue to do the fast-wavenet implementation
    arXiv 1611.09482
    '''
    def __init__(self,
                 max_length,
                 data=None,
                 dilation=1,
                 num_deq=1,
                 num_channels=1,
                 dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data is None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]
        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())

def dilate(sigs, dilation):
    """

    Note this will fail if the dilation doesn't allow a whole number amount of padding

    :param x: Tensor or Variable of size (N, L, C), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated Tensor or Variable of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    n, c, l = sigs.size()
    dilation_factor = dilation / n
    if dilation_factor == 1:
        return sigs, 0.

    # zero padding for reshaping
    new_n = int(dilation)
    new_l = int(np.ceil(l*n/dilation))
    pad_len = (new_n*new_l-n*l)/n
    if pad_len > 0:
        print("Padding: {}, {}, {}".format(new_n, new_l, pad_len))
        # TODO pad output tensor unevenly for indivisible dilations
        assert pad_len == int(pad_len)
        # "squeeze" then "unsqueeze" due to limitation of pad function
        # which only works with 4d/5d tensors
        padding = (int(pad_len), 0, 0, 0) # (d3_St, d3_End, d2_St, d2_End), d0 and d1 unpadded
        sigs = pad1d(sigs, padding)

    # reshape according to dilation
    sigs = sigs.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    sigs = sigs.view(c, new_l, new_n)
    sigs = sigs.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return sigs, pad_len

class ConstantPad1d(nn.Module):
    r"""Pads the input tensor boundaries with a constant value.

    Accepts 3d, 4d, 5d tensors, which is different than the normal PadXd functions

    Args:
        padding (int, tuple): the size of the padding.
            If is int, uses the same padding in all boundaries.
            if a 2-tuple, uses: (d2_padding, d1_padding), equal on both sides
            If a 4-tuple, uses
            (d2_paddingFront, d2_paddingBack,
             d1_paddingFront, d1_paddingBack)


    Shape:
        - Input: :math:`(d0, d1_{in}, d2_{in})`
        - Output: :math:`(d0, d1_{out}, d2_{out})` where
          :math:`d2_{out} = d2_{in} + d2_paddingFront + d2_paddingBack`
          :math:`d1_{out} = d1_{in} + d1_paddingFront + d1_paddingBack`

    Examples::

        >>> m = nn.ConstantPad1d(3, 3.5)
        >>> input = autograd.Variable(torch.randn(3, 320, 480))
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad1d((3, 3, 6, 6), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value=0):
        super(ConstantPad1d, self).__init__()
        self.padding = self._quadruple(padding)
        self.value = value

    def forward(self, input):
        x = input
        if len(x.size()) == 3:
            x = x.view((1,)+x.size())
            x = F.pad(x, self.padding, 'constant', self.value)
            x = x.view(x.size()[1:])
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' ' + str(self.padding)

    def _quadruple(self, padding):
        if isinstance(padding, int):
            padding = tuple([padding]*4)
        elif len(padding) == 2:
            padding = tuple([padding[0]]*2+[padding[1]]*2)
        assert len(padding) == 4
        return padding

def pad1d(input,padding,pad_value=0):
    return ConstantPad1d(padding, pad_value)(input)
