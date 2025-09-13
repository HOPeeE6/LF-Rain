import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, in_ch):
        super(LayerNorm, self).__init__()
        if isinstance(in_ch, numbers.Integral):
            in_ch = (in_ch,)
        in_ch = torch.Size(in_ch)
        assert len(in_ch) == 1

        self.weight = nn.Parameter(torch.ones(in_ch))
        self.bias = nn.Parameter(torch.zeros(in_ch))

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + 1e-5)*self.weight + self.bias
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


##########################################################################
# Gated-Dw-Conv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, in_ch):
        super(FeedForward, self).__init__()
        self.project_in = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*4, kernel_size=1, stride=1, padding=0)
        self.DwConv = nn.Conv2d(in_channels=in_ch*4, out_channels=in_ch*4, kernel_size=3, stride=1, padding=1, groups=in_ch*4)
        self.project_out = nn.Conv2d(in_channels=in_ch*2, out_channels=in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.DwConv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


#################################################################################
# Multi-Class Feature Depth-wise Convolution Self-Attention
class Attention(nn.Module):
    def __init__(self, in_ch, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*3, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.Conv_out = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.Conv_out(out)

        return out


#################################################################################
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.LN_1 = LayerNorm(in_ch=dim)
        self.LN_2 = LayerNorm(in_ch=dim)
        self.attention = Attention(in_ch=dim, num_heads=num_heads)
        self.FeedForward = FeedForward(in_ch=dim)

    def forward(self, x):
        x = self.attention(self.LN_1(x)) + x
        x = self.FeedForward(self.LN_2(x)) + x
        return x