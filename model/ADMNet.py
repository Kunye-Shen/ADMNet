import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import *

class ADMNet(nn.Module):
    def __init__(self, mode='normal'):
        super(ADMNet, self).__init__()
        #Encoder
        ##stage 1
        if mode == 'normal':
                self.encoder1 = nn.Sequential(
                convbnrelu(3, 16, k=3, s=2, p=1),
                MP(16)
                )
        else:
             self.encoder1 = nn.Sequential(
                convbnrelu(3, 16, k=3, s=1, p=1),
                MP(16)
                )

        ##stage 2
        self.encoder2 = nn.Sequential(
                DSConv3x3(16, 32, stride=2),
                MP(32)
                )

        ##stage 3
        self.encoder3_1 = nn.Sequential(
                DSConv3x3(32, 64, stride=2),
                MP(64)
                )
        self.encoder3_2 = MP(64)
        self.encoder3_3 = MP(64)
        self.encoder3_4 = MP(64)
                
        ##stage 4
        self.encoder4_1 = nn.Sequential(
                DSConv3x3(64, 96, stride=2),
                MP(96)
                )
        self.encoder4_2 = MP(96)
        self.encoder4_3 = MP(96)
        self.encoder4_4 = MP(96)
        self.encoder4_5 = MP(96)
        self.encoder4_6 = MP(96)

        ##stage 5
        self.encoder5_1 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                MP(128)
                )
        self.encoder5_2 = MP(128)
        self.encoder5_3 = MP(128)

        #Decoder
        self.kaa = KAA()

        ##stage 5
        self.d5_f1 = DSConv3x3(128, 128, dilation=2)
        self.d5_f2 = DSConv5x5(128, 96, dilation=1)

        ##stage 4
        # DA
        self.d4_f1 = DSConv3x3(96, 96, dilation=2)
        self.esa4 = ESA(96)
        # DSConv 5X5
        self.d4_f2 = DSConv5x5(96, 64, dilation=1)

        ##stage 3
        # DA
        self.d3_f1 = DSConv3x3(64, 64, dilation=2)
        self.esa3 = ESA(64)
        # DSConv 5X5
        self.d3_f2 = DSConv5x5(64, 32, dilation=1)

        ##stage 2
        # DA
        self.d2_f1 = DSConv3x3(32, 32, dilation=2)
        self.esa2 = ESA(32)
        # DSConv 5X5
        self.d2_f2 = DSConv5x5(32, 16, dilation=1)

        ##stage 1
        # DA
        self.d1_f1 = DSConv3x3(16, 16, dilation=2)
        self.esa1 = ESA(16)
        # DSConv 5X5
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
        F_D5 = self.d5_f2(self.d5_f1(score5))
        out5 = self.conv_out5(F_D5)
        t = interpolate(F_D5, score4.size()[2:])

        ##stage 4
        ### DA
        DF4 = self.d4_f1(score4 + t)
        SF5 = self.kaa(F_D5, out5)
        SF5 = interpolate(SF5, score4.size()[2:])

        F_SA4 = self.esa4(DF4, SF5)
        F_D4 = self.d4_f2(F_SA4)
        out4 = self.conv_out4(F_D4)
        t = interpolate(F_D4, score3.size()[2:])

        ##stage 3
        ### DA
        DF3 = self.d3_f1(score3 + t)
        SF4 = self.kaa(F_D4, out4)
        SF4 = interpolate(SF4, score3.size()[2:])

        F_SA3 = self.esa3(DF3, SF4)
        F_D3 = self.d3_f2(F_SA3)
        out3 = self.conv_out3(F_D3)
        t = interpolate(F_D3, score2.size()[2:])

        ##stage 2
        DF2 = self.d2_f1(score2 + t)
        SF3 = self.kaa(F_D3, out3)
        SF3 = interpolate(SF3, score2.size()[2:])

        F_SA2 = self.esa2(DF2, SF3)
        F_D2 = self.d2_f2(F_SA2)
        out2 = self.conv_out2(F_D2)
        t = interpolate(F_D2, score1.size()[2:])
        ##stage 1
        DF1 = self.d1_f1(score1 + t)
        SF2 = self.kaa(F_D2, out2)
        SF2 = interpolate(SF2, score1.size()[2:])

        F_SA1 = self.esa1(DF1, SF2)
        F_D1 = self.d1_f2(F_SA1)
        out1 = self.conv_out1(F_D1)

        #Output
        out1 = interpolate(out1, x.size()[2:])
        out2 = interpolate(out2, x.size()[2:])
        out3 = interpolate(out3, x.size()[2:])
        out4 = interpolate(out4, x.size()[2:])
        out5 = interpolate(out5, x.size()[2:])

        return out1, out2, out3, out4, out5

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
