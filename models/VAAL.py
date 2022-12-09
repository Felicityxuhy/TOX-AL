import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=1, f_filt=4):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(
            nn.Conv1d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32  torch.Size([64, 128, 524])
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16  torch.Size([64, 256, 262])
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8  torch.Size([64, 512, 131])
            nn.BatchNorm1d(512),
            nn.ReLU(True),                                                      
            nn.Conv1d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024, 4, 4  torch.Size([64, 1024, 65])
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            View((-1, 1024*65)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*65, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*65, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*131),                           # B, 1024*8*8
            View((-1, 1024, 131)),                               # B, 1024,  8,  8
            nn.ConvTranspose1d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        z = self._encode(x)
        # print(z.shape) # torch.Size([1040, 4096])
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)
        # print(x_recon.shape)

        return x_recon.squeeze(1), z.squeeze(1), mu.squeeze(1), logvar.squeeze(1)

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
