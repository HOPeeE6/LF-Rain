import torch
from torch import nn
from Option import args

#################################################################################
# "Implement feature extraction and its derived behaviors."
#################################################################################

def sai2mpi(sai, A=args.A):
    B, C, H, W = sai.shape
    h, w = H//A, W//A

    j_1 = torch.arange(0, W) // A
    j_1 = j_1.repeat(H, 1).to(sai.device)
    j_2 = torch.arange(0, W) % A
    j_2 = j_2.repeat(H, 1).to(sai.device)
    j = (j_2 * w) + j_1

    i_1 = torch.arange(0, H) // A
    i_1 = i_1.unsqueeze(0).T
    i_1 = i_1.repeat(1, W).to(sai.device)
    i_2 = torch.arange(0, H) % A
    i_2 = i_2.unsqueeze(0).T
    i_2 = i_2.repeat(1, W).to(sai.device)
    i = (i_2 * h) + i_1
    mpi = sai[:, :, i, j]
    return mpi

def mpi2sai(mpi, A=args.A):
    B, C, H, W = mpi.shape
    h, w = H//A, W//A

    j_1 = torch.arange(0, W) // w
    j_1 = j_1.repeat(H, 1).to(mpi.device)
    j_2 = torch.arange(0, W) % w
    j_2 = j_2.repeat(H, 1).to(mpi.device)
    j = (j_2 * A) + j_1

    i_1 = torch.arange(0, H) // h
    i_1 = i_1.unsqueeze(0).T
    i_1 = i_1.repeat(1, W).to(mpi.device)
    i_2 = torch.arange(0, H) % h
    i_2 = i_2.unsqueeze(0).T
    i_2 = i_2.repeat(1, W).to(mpi.device)
    i = (i_2 * A) + i_1

    sai = mpi[:, :, i, j]

    return sai


class PixelShuffle1D(nn.Module):
    def __init__(self, factor, direction):
        super(PixelShuffle1D, self).__init__()
        assert direction in ['H', 'V'], "direction must be 'H' or 'V'"
        self.factor = factor
        self.direction = direction

    def forward(self, x):
        B, FC, H, W = x.size()
        assert FC % self.factor == 0, "Invalid input: incompatible number of channels."
        C = FC // self.factor
        x = x.view(B, self.factor, C, H, W)

        if self.direction == 'H':
            x = x.permute(0, 2, 3, 4, 1).reshape(B, C, H, W * self.factor)
        elif self.direction == 'V':
            x = x.permute(0, 2, 3, 4, 1).reshape(B, C, H * self.factor, W)
        return x
