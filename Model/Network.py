import torch
import torch.nn as nn
from Model.ResASPP import ResASPP
from Model.MSDABlock import MSDABlock
from Model.DisentgBlock import DisentgBlock

class myNetwork(nn.Module):
    def __init__(self, dim, num_heads):
        super(myNetwork, self).__init__()
        self.Conv_in = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.ResASPP = ResASPP(dim=dim)
        self.MSDA_1 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads)
        )
        self.Disen_1 = DisentgBlock(dim=dim)
        self.MSDA_2 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads)
        )

        self.MSDA_3 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads),
            MSDABlock(dim=dim, num_heads=num_heads)
        )
        self.Disen_2 = DisentgBlock(dim=dim)
        self.MSDA_4 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads),
            MSDABlock(dim=dim, num_heads=num_heads)
        )

        self.MSDA_5 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads),
            MSDABlock(dim=dim, num_heads=num_heads)
        )
        self.Disen_3 = DisentgBlock(dim=dim)
        self.MSDA_6 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads),
            MSDABlock(dim=dim, num_heads=num_heads)
        )

        self.FF_1 = nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.MSDA_7 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads),
            MSDABlock(dim=dim, num_heads=num_heads)
        )
        self.Disen_4 = DisentgBlock(dim=dim)
        self.MSDA_8 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads),
            MSDABlock(dim=dim, num_heads=num_heads)
        )

        self.FF_2 = nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.MSDA_9 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads)
        )
        self.Disen_5 = DisentgBlock(dim=dim)
        self.MSDA_10 = nn.Sequential(
            MSDABlock(dim=dim, num_heads=num_heads)
        )
        self.Conv_out = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        x = self.Conv_in(img)

        x = self.MSDA_1(x)
        x = self.Disen_1(x)
        x_1 = self.MSDA_2(x)

        x = self.MSDA_3(x_1)
        x = self.Disen_2(x)
        x_2 = self.MSDA_4(x)

        x = self.MSDA_5(x)
        x = self.Disen_3(x)
        x = self.MSDA_6(x)

        x = self.FF_1(torch.cat((x, x_2), dim=1))
        x = self.MSDA_7(x)
        x = self.Disen_4(x)
        x = self.MSDA_8(x)

        x = self.FF_2(torch.cat((x, x_1), dim=1))
        x = self.MSDA_9(x)
        x = self.Disen_5(x)
        x = self.MSDA_10(x)

        out = self.Conv_out(x)

        return out + img



