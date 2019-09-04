import torch
import torch.nn as nn
from DHDN_12 import DHDN
from SpyNet import SpyNet, Backward


class model_tsr(nn.Module):
    def __init__(self):
        super(model_tsr, self).__init__()
        self.SpyNet = SpyNet()
        self.DHDN = DHDN(6, 3)

    def forward(self, data0, data1):

        flow01 = self.SpyNet(data0, data1)
        flow10 = self.SpyNet(data1, data0)

        flow12_0 = - (1 - 0.5) * 0.5 * flow01 + 0.5 * 0.5 * flow10
        flow12_1 = (1 - 0.5) * (1 - 0.5) * flow01 - 0.5 * (1 - 0.5) * flow10

        data0_12 = Backward(data0, flow12_0)
        data1_12 = Backward(data1, flow12_1)

        output = self.DHDN(torch.cat([data0_12, data1_12], 1))

        return output
