import torch
import torch.nn as nn
import torch.nn.functional as F

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class ConvOut(nn.Module):
    def __init__(self, in_channel):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)


class MP(nn.Module): # Multi-scale perception (MP) module
    def __init__(self, channel):
        super(MP, self).__init__()
        self.conv1 = DSConv3x3(channel, channel, stride=1, dilation=1)
        self.conv2 = DSConv3x3(channel, channel, stride=1, dilation=2)
        self.conv3 = DSConv3x3(channel, channel, stride=1, dilation=4)
        self.conv4 = DSConv3x3(channel, channel, stride=1, dilation=8)

        self.fuse = DSConv3x3(channel, channel, relu=False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x+out1)
        out3 = self.conv3(x+out2)
        out4 = self.conv4(x+out3)

        out = self.fuse(out1 + out2 + out3 + out4)

        return out + x


class KAA(nn.Module): #Kernel Area Acquisition
    def __init__(self):
        super(KAA, self).__init__()
    
    def forward(self, feature, sal):
        if torch.cuda.is_available():
            sal0_1 = torch.where(sal>=torch.tensor(0.5).cuda(), torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
        else:
            sal0_1 = torch.where(sal>=torch.tensor(0.5), torch.tensor(1.0), torch.tensor(0.0))
        feature = feature * sal0_1

        return feature


class ESA(nn.Module): # Efficient Spatial Attention
    def __init__(self, channel):
        super(ESA, self).__init__()
        self.gate = nn.Sequential(
            DSConv3x3(channel, channel, dilation=6),
            DSConv3x3(channel, channel, dilation=4),
            DSConv3x3(channel, channel, dilation=2),
            nn.Conv2d(channel, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, DF, SF):
        SA = self.gate(SF)
        F_SA = SA * (DF + SF)

        return F_SA