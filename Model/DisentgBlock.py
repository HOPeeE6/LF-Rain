import torch
import torch.nn as nn
from Option import args
from Model.LF_Operator import *


class DisentgBlock(nn.Module):
    def __init__(self, dim, A=args.A):
        super(DisentgBlock, self).__init__()
        S_dim, A_dim, E_dim = dim, dim//2, dim
        self.SFB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=S_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=S_dim, out_channels=S_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.AFB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=A_dim, kernel_size=A, stride=A, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=A_dim, out_channels=A_dim*A*A, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(A)
        )
        self.EPE_HB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=E_dim, kernel_size=(1, A*3), stride=(1, A), padding=(0, A)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=E_dim, out_channels=E_dim*A, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(factor=A, direction='H')
        )
        self.EPE_VB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=E_dim, kernel_size=(A*3, 1), stride=(A, 1), padding=(A, 0)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=E_dim, out_channels=E_dim*A, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(factor=A, direction='V')
        )
        self.FF = nn.Sequential(
            nn.Conv2d(in_channels=S_dim+E_dim+A_dim+E_dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        S = self.SFB(x)
        mpi = sai2mpi(x)
        A = mpi2sai(self.AFB(mpi))
        H = mpi2sai(self.EPE_HB(mpi))
        V = mpi2sai(self.EPE_VB(mpi))
        out = self.FF(torch.cat([S, H, V, A], dim=1))

        return out + x
