import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import hexagdly

from IPython import embed


# Runs basic linear regression on our number of edges
#   For testing 
class TestSAMNet(nn.Module):
    def __init__(self, n_dim=2):
        super(TestSAMNet, self).__init__()
        self.fc1 = nn.Linear(n_dim, 1)

    def forward(self, x):
        x = self.fc1(x)

        return x

    @property
    def coef_(self):
        return self.fc1.weight.detach().numpy()

    @property
    def intercept_(self):
        return self.fc1.bias.item()

# One hidden layer net
class SAMNet(nn.Module):
    def __init__(self, n_patch_dim=36, n_hidden=36, n_layers=1, n_out=1, drop_out=0.0):
        super(SAMNet, self).__init__()

        layers = []

        for i in range(n_layers):
            if i == 0:
                L = nn.Linear(n_patch_dim, n_hidden)
            else:
                L = nn.Linear(n_hidden, n_hidden)
            
            layers.append(L)
            layers.append(nn.ReLU())
            
            if drop_out > 0:
                layers.append(nn.Dropout(drop_out))

        self.layers = nn.Sequential(*layers)
        self.o = nn.Linear(n_hidden, n_out)

    def forward(self, x):

        out = self.layers(x)
        out = self.o(out)

        return out

    @property
    def n_layers(self):
        return len(self.layers)


class SAMConvNet(nn.Module):

    def __init__(self, n_out_channels=4, n_hidden=36, n_layers=1, n_out=1, drop_out=0.0, ny=13, nz=11):
        super(SAMConvNet, self).__init__()

        self.n_out_channels = n_out_channels
        # INPUT: (1 x 6 x 6)
        # Conv Output: (1 x 6 x 6)
        # Pool Output: (1 x 3 x 3)
        self.layer1 = nn.Sequential(
            hexagdly.Conv2d(1, n_out_channels, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            hexagdly.MaxPool2d(kernel_size=1, stride=2))

        # Determine pooling output size (ugh, hackish)
        dummy = torch.rand((1,1,nz,ny))
        p = hexagdly.MaxPool2d(kernel_size=1, stride=2)
        self.n_pool_out = np.prod(p(dummy).shape) * n_out_channels

        # Fully-connected layer(s)
        self.fc = SAMNet(self.n_pool_out, n_hidden, n_layers, n_out, drop_out)

    def forward(self, x):

        out = self.layer1(x)
        out = out.reshape(-1, self.n_pool_out)
        out = self.fc(out)

        return out

# Conv filters, but one output node (no hidden layers)
# Only learned weights are the filters
class SAMFixedConvNet(nn.Module):

    def __init__(self, n_out_channels=4, n_hidden=36, n_layers=1, n_out=1, drop_out=0.0, ny=13, nz=11):
        super(SAMFixedConvNet, self).__init__()

        self.n_out_channels = n_out_channels
        # INPUT: (1 x 6 x 6)
        # Conv Output: (1 x 6 x 6)
        # Pool Output: (1 x 3 x 3)
        self.layer1 = nn.Sequential(
            hexagdly.Conv2d(1, n_out_channels, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            hexagdly.MaxPool2d(kernel_size=1, stride=2))

        # Determine pooling output size (ugh, hackish)
        dummy = torch.rand((1,1,nz,ny))
        p = hexagdly.MaxPool2d(kernel_size=1, stride=2)
        self.n_pool_out = np.prod(p(dummy).shape) * n_out_channels

        # Fully-connected layer(s)
        self.o = nn.Linear(self.n_pool_out, n_out, bias=True)
        #for p in self.o.parameters():
        #    p.requires_grad = False
        #self.o.weight[:] = 1

    def forward(self, x):

        out = self.layer1(x)
        out = out.reshape(-1, self.n_pool_out)
        out = self.o(out)

        return out

