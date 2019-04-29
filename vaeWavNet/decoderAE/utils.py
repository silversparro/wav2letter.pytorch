from collections import deque
import torch
import numpy as np

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return deque(factors)

def mu_law_encoding(x, quantization_channels):
    mu = quantization_channels - 1.
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((x_mu + 1) / 2 * mu + 0.5).astype(int)

def mu_law_expansion(x_mu, quantization_channels):
    mu = quantization_channels - 1.
    x = ((x_mu) / mu) * 2 - 1.
    return np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu



def time_to_batch(sigs, sr):
    '''Adds zero padding to inputs and reshapes by sample rate.  This essentially
    rebatches the input into one second batches.

    Used to perform 1D dilated convolution

    Args:
      sig: (tensor) in (Bx)SxC; B = # batches, S = # samples, C = # channels
      sr: (int) sample rate of audio signal
    Outputs:
      sig: (tensor) also in SecBatchesx(B x sr)xC, SecBatches = # of seconds in
            padded sample
    '''

    unsqueezed = False

    # check if sig is a batch, if not make a batch of 1
    if len(sigs.size()) == 1:
        sigs.unsqueeze_(0)
        unsqueezed = True

    assert len(sigs.size()) == 3

    # pad to the second (i.e. sample rate)
    b_num, s_num, c_num = sigs.size()
    width_pad = int(sr * np.ceil(s_num / sr + 1))
    lpad_len = width_pad - s_num
    lpad = torch.zeros(b_num, pad_left_len, c_num)
    sigs = torch.cat((lpad, sigs), 1) # concat on sample dimension

    # reshape to batches of one second each
    secs_num = width_pad // sr
    sigs = sigs.view(secs_num, -1, c_num) # seconds x (batches*rate) x channels

    return sigs

def batch_to_time(sigs, sr, lcrop=0):
    ''' Reshape to 1d signal from batches of 1 second.

    I'm using the same variable names as above as opposed to the original
    author's variables

    Used to perform dilated conv1d

    Args:
      sig: (tensor) second_batches_num x (batch_size x sr) x channels
      sr: (int)
      lcrop: (int)
    Outputs:
      sig: (tensor) batch_size x # of samples x channels
    '''

    assert len(sigs.size()) == 3

    secs_num, bxsr, c_num = sigs.size()
    b_num = bxsr // sr
    width_pad = int(secs_num * sr)

    sigs = sigs.view(-1, width_pad, c_num) # missing dim should be b_num

    assert sigs.size(0) == b_num

    if lcrop > 0:
        sigs = sigs[:,lcrop:, :]

    return sigs
