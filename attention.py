import torch
import torch.nn
from torch import nn

__all__ = ["Patten", "Catten"]


class Patten(nn.Module):
    def __int__(self, in_dim, n):
        super(Patten, self).__int__()
        self.in_channel = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // n, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // n, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, h, w = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, h, w)

        out -= self.gamma * out + x
        return out


class Catten(nn.Module):
    def __int__(self, in_dim):
        super(Catten, self).__int__()
        self.channel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
