import torch
import torch.nn as nn
from DHDN_1434 import DHDN
from SpyNet import SpyNet, Backward


class model_tsr(nn.Module):
    def __init__(self):
        super(model_tsr, self).__init__()
        self.SpyNet = SpyNet()
        self.DHDN = DHDN(6, 3)

    def forward(self, data0, data1):

        flow01 = self.SpyNet(data0, data1)
        flow10 = self.SpyNet(data1, data0)

        flow14_0 = - (1 - 0.25) * 0.25 * flow01 + 0.25 * 0.25 * flow10
        flow14_1 = (1 - 0.25) * (1 - 0.25) * flow01 - 0.25 * (1 - 0.25) * flow10

        data0_14 = Backward(data0, flow14_0)
        data1_14 = Backward(data1, flow14_1)

        output = self.DHDN(torch.cat([data0_14, data1_14], 1))

        return output
