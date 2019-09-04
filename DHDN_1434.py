import torch
import torch.nn as nn


class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.PReLU()

    def forward(self, x):

        out = self.relu1(self.conv1(x))

        out = self.relu2(self.conv2(out))

        out = torch.add(out, x)

        return out


class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2 * channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out


class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        out = self.relu(self.conv(x))

        out = self.subpixel(out)

        return out


class DHDN(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DHDN, self).__init__()
        cn = 30
        self.conv_i = nn.Conv2d(in_channels=inchannel, out_channels=cn, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, cn)
        self.down1 = self.make_layer(_down, cn)
        self.DCR_block21 = self.make_layer(_DCR_block, cn * 2)
        self.down2 = self.make_layer(_down, cn * 2)
        self.DCR_block31 = self.make_layer(_DCR_block, cn * 4)
        self.down3 = self.make_layer(_down, cn * 4)
        self.DCR_block41 = self.make_layer(_DCR_block, cn * 8)
        self.down4 = self.make_layer(_down, cn * 8)
        self.DCR_block51 = self.make_layer(_DCR_block, cn * 16)
        self.up4 = self.make_layer(_up, cn * 32)
        self.DCR_block42 = self.make_layer(_DCR_block, cn * 16)
        self.up3 = self.make_layer(_up, cn * 16)
        self.DCR_block32 = self.make_layer(_DCR_block, cn * 8)
        self.up2 = self.make_layer(_up, cn * 8)
        self.DCR_block22 = self.make_layer(_DCR_block, cn * 4)
        self.up1 = self.make_layer(_up, cn * 4)
        self.DCR_block12 = self.make_layer(_DCR_block, cn * 2)
        self.conv_r = nn.Conv2d(in_channels=cn * 2, out_channels=cn, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()
        self.conv_f = nn.Conv2d(in_channels=cn, out_channels=outchannel, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.relu1(self.conv_i(x))

        conc1 = self.DCR_block11(residual)

        out = self.down1(conc1)

        conc2 = self.DCR_block21(out)

        out = self.down2(conc2)

        conc3 = self.DCR_block31(out)

        out = self.down3(conc3)

        conc4 = self.DCR_block41(out)

        conc5 = self.down4(conc4)

        out = self.DCR_block51(conc5)

        out = self.up4(torch.cat([conc5, out], 1))

        out = self.DCR_block42(torch.cat([conc4, out], 1))

        out = self.up3(out)

        out = self.DCR_block32(torch.cat([conc3, out], 1))

        out = self.up2(out)

        out = self.DCR_block22(torch.cat([conc2, out], 1))

        out = self.up1(out)

        out = self.DCR_block12(torch.cat([conc1, out], 1))

        out = self.relu2(self.conv_r(out))

        out = torch.add(residual, out)

        out = self.relu3(self.conv_f(out))

        return out
