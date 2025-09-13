import torch
import torch.nn as nn
from Model.Attention import AttentionBlock
from Model.LF_Operator import *
from Option import args
import torch.nn.functional as F

class MSDABlock(nn.Module):
    def __init__(self, dim, num_heads, A=args.A):
        super(MSDABlock, self).__init__()
        S_dim, A_dim, E_dim = dim, dim//2, dim
        self.SFB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=S_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            AttentionBlock(dim=S_dim, num_heads=num_heads)
        )
        self.AFB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=A_dim, kernel_size=A, stride=A, padding=0),
            nn.Conv2d(in_channels=A_dim, out_channels=A_dim*A*A, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(A),
            nn.Conv2d(in_channels=A_dim, out_channels=A_dim*2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.EPE_HB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=E_dim, kernel_size=(1, A*3), stride=(1, A), padding=(0, A)),
            nn.Conv2d(in_channels=E_dim, out_channels=E_dim*A, kernel_size=1, stride=1, padding=0),
            PixelShuffle1D(factor=A, direction='H'),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.EPE_VB = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=E_dim, kernel_size=(A*3, 1), stride=(A,1), padding=(A, 0)),
            nn.Conv2d(in_channels=E_dim, out_channels=E_dim*A, kernel_size=1, stride=1, padding=0),
            PixelShuffle1D(factor=A, direction='V'),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.Conv_HV = nn.Sequential(
            nn.Conv2d(in_channels=E_dim*2, out_channels=E_dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.Conv_E = nn.Sequential(
            nn.Conv2d(in_channels=E_dim*2, out_channels=E_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=S_dim + A_dim + E_dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.Att = AttentionBlock(dim=dim, num_heads=num_heads)

    def forward(self, x):
        S = self.SFB(x)
        mpi = sai2mpi(x)
        A_1, A_2 = self.AFB(mpi).chunk(2, dim=1)
        A = A_1 * F.tanh(A_2)
        A = mpi2sai(A)
        H = self.EPE_HB(mpi)
        V = self.EPE_VB(mpi)
        HV = self.Conv_HV(torch.cat([H, V], dim=1))
        H = H + H * HV
        V = V + V * HV
        E = torch.cat([H, V], dim=1)
        E = self.Conv_E(E)
        E = mpi2sai(E)
        LF = torch.cat([S, E, A], dim=1)
        LF = self.Conv(LF)
        out = self.Att(LF)

        return out + x