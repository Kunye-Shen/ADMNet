import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

class ADMNet(nn.Module):
    def __init__(self):
        super(ADMNet, self).__init__()
        #Encoder
        ##stage 1
        self.encoder1 = nn.Sequential(
                convbnrelu(3, 16, k=3, s=2, p=1),
                MRDC(16)
                )

        ##stage 2
        self.encoder2 = nn.Sequential(
                DSConv3x3(16, 32, stride=2),
                MRDC(32)
                )

        ##stage 3
        self.encoder3_1 = nn.Sequential(
                DSConv3x3(32, 64, stride=2),
                MRDC(64)
                )
        self.encoder3_2 = MRDC(64)
        self.encoder3_3 = MRDC(64)
        self.encoder3_4 = MRDC(64)
                
        ##stage 4
        self.encoder4_1 = nn.Sequential(
                DSConv3x3(64, 96, stride=2),
                MRDC(96)
                )
        self.encoder4_2 = MRDC(96)
        self.encoder4_3 = MRDC(96)
        self.encoder4_4 = MRDC(96)
        self.encoder4_5 = MRDC(96)
        self.encoder4_6 = MRDC(96)

        ##stage 5
        self.encoder5_1 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                MRDC(128)
                )
        self.encoder5_2 = MRDC(128)
        self.encoder5_3 = MRDC(128)

        #Decoder
        self.kaa = KAA()

        ##stage 5
        self.d5_f1 = DSConv3x3(128, 128, dilation=2)
        self.d5_f2 = DSConv5x5(128, 96, dilation=1)
        ##stage 4
        self.d4_f1 = DSConv3x3(96, 96, dilation=2)
        self.esa4 = ESA(96)
        self.d4_f2 = DSConv5x5(96, 64, dilation=1)
        ##stage 3
        self.d3_f1 = DSConv3x3(64, 64, dilation=2)
        self.esa3 = ESA(64)
        self.d3_f2 = DSConv5x5(64, 32, dilation=1)
        ##stage 2
        self.d2_f1 = DSConv3x3(32, 32, dilation=2)
        self.esa2 = ESA(32)
        self.d2_f2 = DSConv5x5(32, 16, dilation=1)
        ##stage 1
        self.d1_f1 = DSConv3x3(16, 16, dilation=2)
        self.esa1 = ESA(16)
        self.d1_f2 = DSConv5x5(16, 16, dilation=1)

        #Output
        self.conv_out5 = ConvOut(in_channel=96)
        self.conv_out4 = ConvOut(in_channel=64)
        self.conv_out3 = ConvOut(in_channel=32)
        self.conv_out2 = ConvOut(in_channel=16)
        self.conv_out1 = ConvOut(in_channel=16)

    def forward(self, x):
        #Encoder
        ##stage 1
        score1 = self.encoder1(x)
        ##stage 2
        score2 = self.encoder2(score1)
        ##stage 3
        score3_1 = self.encoder3_1(score2)
        score3_2 = self.encoder3_2(score3_1)
        score3_3 = self.encoder3_3(score3_1 + score3_2)
        score3 = self.encoder3_4(score3_1 + score3_2 + score3_3)
        ##stage 4
        score4_1 = self.encoder4_1(score3)
        score4_2 = self.encoder4_2(score4_1)
        score4_3 = self.encoder4_3(score4_1 + score4_2)
        score4_4 = self.encoder4_4(score4_1 + score4_2 + score4_3)
        score4_5 = self.encoder4_5(score4_1 + score4_2 + score4_3 + score4_4)
        score4 = self.encoder4_6(score4_1 + score4_2 + score4_3 + score4_4 + score4_5)
        ##stage 5
        score5_1 = self.encoder5_1(score4)
        score5_2 = self.encoder5_2(score5_1)
        score5 = self.encoder5_3(score5_1 + score5_2)

        #Decoder
        ##stage 5
        scored5 = self.d5_f2(self.d5_f1(score5))
        out5 = self.conv_out5(scored5)
        scored5_e = self.kaa(scored5, out5)
        scored5_e = interpolate(scored5_e, score4.size()[2:])
        t = interpolate(scored5, score4.size()[2:])
        ##stage 4
        scored4 = self.d4_f1(score4 + t)
        scored4 = self.esa4(scored4, scored5_e)
        scored4 = self.d4_f2(scored4)
        out4 = self.conv_out4(scored4)
        scored4_e = self.kaa(scored4, out4)
        scored4_e = interpolate(scored4_e, score3.size()[2:])
        t = interpolate(scored4, score3.size()[2:])
        ##stage 3
        scored3 = self.d3_f1(score3 + t)
        scored3 = self.esa3(scored3, scored4_e)
        scored3 = self.d3_f2(scored3)
        out3 = self.conv_out3(scored3)
        scored3_e = self.kaa(scored3, out3)
        scored3_e = interpolate(scored3_e, score2.size()[2:])
        t = interpolate(scored3, score2.size()[2:])
        ##stage 2
        scored2 = self.d2_f1(score2 + t)
        scored2 = self.esa2(scored2, scored3_e)
        scored2 = self.d2_f2(scored2)
        out2 = self.conv_out2(scored2)
        scored2_e = self.kaa(scored2, out2)
        scored2_e = interpolate(scored2_e, score1.size()[2:])
        t = interpolate(scored2, score1.size()[2:])
        ##stage 1
        scored1 = self.d1_f1(score1 + t)
        scored1 = self.esa1(scored1, scored2_e)
        scored1 = self.d1_f2(scored1)
        out1 = self.conv_out1(scored1)

        #Output
        out1 = interpolate(out1, x.size()[2:])
        out2 = interpolate(out2, x.size()[2:])
        out3 = interpolate(out3, x.size()[2:])
        out4 = interpolate(out4, x.size()[2:])
        out5 = interpolate(out5, x.size()[2:])

        return out1, out2, out3, out4, out5

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)