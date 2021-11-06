import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio 
import numpy as np
# from tqdm import tqdm
from einops import rearrange
# from collections import OrderedDict

class Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Spec = torchaudio.transforms.Spectrogram(n_fft = 500,
                                                      power = None,
                                                      hop_length = 250)
        self.IPD = IPDExtractor()
        self.cla_net = FConvNet(133, 3)
        
    def forward(self, x):
        x = self.Spec(x)
        ipd = self.IPD(x)
        x = rearrange(x, 'b m f t c -> b (m c) f t')
        x = torch.cat([x, ipd], -3)
        x_cla = self.cla_net(x)
        return x_cla

    def reset_state(self):
        self.cla_net.h = None

class FConvNet(nn.Module):
    def __init__(self, in_features, out_features, batchnorm = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.tconv = nn.Sequential(nn.Conv1d(in_features, 128, 5, 2, 2),
                                   nn.ReLU(),
                                   nn.Conv1d(128, 128, 5, 2, 2),
                                   nn.ReLU(),
                                   nn.Conv1d(128, 128, 5, 2, 2),
                                   nn.ReLU(),
                                   nn.Conv1d(128, 128, 5, 2, 2),
                                   nn.ReLU())

        self.fconv = nn.Sequential(*[ConvBlock(*args) for args in [(128, 128, 5, 2, 'f', False),
                                                                   (128, 128, 3, 1, 'f', False),
                                                                   (128, 128, 5, 2, 'f', False),
                                                                   (128, 128, 3, 1, 'f', False),
                                                                   (128, 128, 5, 2, 'f', False),
                                                                   (128, 128, 3, 1, 'f', False),
                                                                   (128, 128, 5, 2, 'f', False),
                                                                   (128, 128, 3, 1, 'f', False),
                                                                   (128, 128, 5, 2, 'f', False),
                                                                   (128, 128, 3, 2, 'f', False)]])

        self.tgru = nn.GRU(128 * 4, 512, num_layers = 2, batch_first = True)
        self.linear = nn.Linear(512, out_features)

        self.h = None

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c f t -> (b f) c t')
        x = self.tconv(x)
        x = rearrange(x, '(b f) c t -> b c f t', b = b)
        x = self.fconv(x)
        x = rearrange(x, 'b c f t -> b t (f c)', b = b)
        x, self.h = self.tgru(x, self.h.detach() if type(self.h) == torch.Tensor else self.h)
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, c_i, c_o, k, s, axis, depthwise = True):
        super().__init__()

        if depthwise:
            self.net = [nn.Conv2d(c_i, c_o, 1, 1, 0),
                        nn.BatchNorm2d(c_o),
                        nn.ReLU6()]
            self.net += [nn.Conv2d(c_o, c_o, (k, 1), (s, 1), (k // 2, 0), groups = c_o),
                         nn.BatchNorm2d(c_o),
                         nn.ReLU6()]
        else:
            self.net = [nn.Conv2d(c_i, c_o, (k, 1), (s, 1), (k // 2, 0)),
                        nn.BatchNorm2d(c_o),
                        nn.ReLU6()]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class IPDExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x      = torch.view_as_complex(x)
        x_mag  = torch.log(torch.abs(x) + 1e-5)
        x_pha  = torch.angle(x)
        cospha = torch.cos(x_pha)
        sinpha = torch.sin(x_pha)
        x      = rearrange(x, 'b m f t -> b m 1 f t')
        x_conj = rearrange(torch.conj(x), 'b m 1 f t -> b 1 m f t')
        ipdmat = rearrange((x * x_conj).angle(), 'b m1 m2 f t -> b (m1 m2) f t')
        cosipd = torch.cos(ipdmat)
        sinipd = torch.sin(ipdmat)
        ipd    = torch.cat([x_mag, cospha, sinpha, cosipd, sinipd], -3)
        return ipd

    def sample_points(self, num_sampling_points):
        points = np.random.randn(num_sampling_points, 3)
        norm = np.sqrt(np.sum(points ** 2, -1, keepdim = True))
        points[:, 2] = np.abs(points[:, 2])
        points = points / norm
        return points

    

# if __name__ == '__main__':
    # net = FConvEstimator(batchnorm = False)
    # for i in tqdm(range(1000)):
        # x_cla, x_dir = net(torch.rand(1, 7, 16000 * 10))

