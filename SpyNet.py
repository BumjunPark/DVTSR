import torch
import math
import torch.nn as nn

arguments_strModel = 'sintel-final'
Backward_tensorGrid = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                                                                                  tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                                                                                tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
    # en
    tensorFlow = torch.cat(
    [tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return nn.functional.grid_sample(input=tensorInput,
                                     grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3,
                                                                                                             1),
                                     mode='bilinear', padding_mode='border')


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, tensorInput):
        tensorRed = (tensorInput[:, 0:1, :, :] - 0.4176) / 0.2599
        tensorGreen = (tensorInput[:, 1:2, :, :] - 0.4149) / 0.2543
        tensorBlue = (tensorInput[:, 2:3, :, :] - 0.4003) / 0.2718

        return torch.cat((tensorRed, tensorGreen, tensorBlue), 1)


class Basic(nn.Module):
    def __init__(self, intLevel):
        super(Basic, self).__init__()

        self.moduleBasic = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

    def forward(self, tensorInput):
        return self.moduleBasic(tensorInput)


class SpyNet(nn.Module):
    def __init__(self):
        super(SpyNet, self).__init__()
        self.normalize = Normalize()
        self.moduleBasic = nn.ModuleList([Basic(intLevel) for intLevel in range(6)])
        self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, tensorFirst, tensorSecond):
        tensorFlow = []

        tensorFirst_ = [self.normalize(tensorFirst)]
        tensorSecond_ = [self.normalize(tensorSecond)]

        for intLevel in range(5):
            if tensorFirst_[0].size(2) > 2 or tensorFirst_[0].size(3) > 2:
                tensorFirst_.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst_[0], kernel_size=2, stride=2, count_include_pad=False))  # [1 3 208 512], [1 3 416 1024] 이런식으로 앞에 작은게 append됨.
                tensorSecond_.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond_[0], kernel_size=2, stride=2, count_include_pad=False))
        # end

        tensorFlow = tensorFirst_[0].new_zeros([tensorFirst_[0].size(0), 2, int(math.floor(tensorFirst_[0].size(2) / 2.0)), int(math.floor(tensorFirst_[0].size(3) / 2.0))])

        for intLevel in range(len(tensorFirst_)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tensorUpsampled.size(2) != tensorFirst_[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled,
                                                                                                                   pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst_[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled,
                                                                                                                   pad=[0, 1, 0, 0], mode='replicate')

            tensorFlow = self.moduleBasic[intLevel](
                torch.cat([tensorFirst_[intLevel], Backward(tensorInput=tensorSecond_[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled],
                          1)) + tensorUpsampled
        # end

        return tensorFlow
# end

