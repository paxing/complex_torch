# -*- coding: utf-8 -*-
"""
@author: Paul Xing
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from models.ComplexTorch import *
from models.Complex_blocks import *



def _make_nComplexConv2d(nchan, depth, kernel_size=3, stride=1, padding=1):
    layers = []
    for _ in range(depth):
        layers.append(CReLUConv2d(nchan, nchan, kernel_size, stride=stride, padding=padding))
    return ComplexSequential(*layers)



class ComplexInceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block = None):
        super(ComplexInceptionA, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv2d

        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.avg_pool = ComplexAvgPool2d( kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)


    def _forward(self, x_real, x_im):
        branch1x1 = self.branch1x1(x_real, x_im)

        branch5x5 = self.branch5x5_1(x_real, x_im)
        branch5x5 = self.branch5x5_2(*branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x_real, x_im)
        branch3x3dbl = self.branch3x3dbl_2(*branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(*branch3x3dbl)

        branch_pool = self.avg_pool(x_real, x_im)
        branch_pool = self.branch_pool(*branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x_real, x_im):
        outputs = self._forward(x_real, x_im)
        return torch.cat([i for i,j in outputs], 1), torch.cat([j for i,j in outputs],1)



class ComplexInceptionB(nn.Module):

    def __init__(self,in_channels, conv_block = None):
        super(ComplexInceptionB, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)
        self.max_pool = ComplexMaxPool2d( kernel_size=3, stride=2)

    def _forward(self, x_real, x_im):
        branch3x3 = self.branch3x3(x_real, x_im)

        branch3x3dbl = self.branch3x3dbl_1(x_real, x_im)
        branch3x3dbl = self.branch3x3dbl_2(*branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(*branch3x3dbl)

        branch_pool = self.max_pool(x_real, x_im)


        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x_real, x_im):
        outputs = self._forward(x_real, x_im)
        return torch.cat([i for i,j in outputs], 1), torch.cat([j for i,j in outputs], 1)




class ComplexInceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block = None):
        super(ComplexInceptionC, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.avg_pool = ComplexAvgPool2d( kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x_real, x_im):
        branch1x1 = self.branch1x1(x_real, x_im)

        branch7x7 = self.branch7x7_1(x_real, x_im)
        branch7x7 = self.branch7x7_2(*branch7x7)
        branch7x7 = self.branch7x7_3(*branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x_real, x_im)
        branch7x7dbl = self.branch7x7dbl_2(*branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(*branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(*branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(*branch7x7dbl)

        branch_pool = self.avg_pool(x_real, x_im)
        branch_pool = self.branch_pool(*branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x_real, x_im):
        outputs = self._forward(x_real, x_im)
        return torch.cat([i for i,j in outputs], 1), torch.cat([j for i,j in outputs],1)



class ComplexInceptionD(nn.Module):

    def __init__(self, in_channels, conv_block = None):
        super(ComplexInceptionD, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

        self.max_pool = ComplexMaxPool2d( kernel_size=3, stride=2)

    def _forward(self, x_real, x_im):
        branch3x3 = self.branch3x3_1(x_real, x_im)
        branch3x3 = self.branch3x3_2(*branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x_real, x_im)
        branch7x7x3 = self.branch7x7x3_2(*branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(*branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(*branch7x7x3)

        branch_pool = self.max_pool(x_real, x_im)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x_real, x_im):
        outputs = self._forward(x_real, x_im)
        return torch.cat([i for i,j in outputs], 1), torch.cat([j for i,j in outputs],1)



class ComplexInceptionE(nn.Module):

    def __init__(self, in_channels, conv_block = None):
        super(ComplexInceptionE, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.avg_pool = ComplexAvgPool2d( kernel_size=3, stride=1, padding=1)
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x_real, x_im):
        branch1x1 = self.branch1x1(x_real, x_im)

        branch3x3 = self.branch3x3_1(x_real, x_im)
        branch3x3 = [
            self.branch3x3_2a(*branch3x3),
            self.branch3x3_2b(*branch3x3),
        ]
        branch3x3 = torch.cat([i for i,j in branch3x3], 1),  torch.cat([j for i,j in branch3x3], 1)

        branch3x3dbl = self.branch3x3dbl_1(x_real, x_im)
        branch3x3dbl = self.branch3x3dbl_2(*branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(*branch3x3dbl),
            self.branch3x3dbl_3b(*branch3x3dbl),
        ]
        branch3x3dbl = torch.cat([i for i,j in branch3x3dbl], 1), torch.cat([j for i,j in branch3x3dbl], 1)

        branch_pool = self.avg_pool(x_real, x_im)
        branch_pool = self.branch_pool(*branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs


    def forward(self, x_real, x_im):
        outputs = self._forward(x_real, x_im)
        return torch.cat([i for i,j in outputs], 1), torch.cat([j for i,j in outputs],1)



class ComplexInceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block = None):
        super(ComplexInceptionAux, self).__init__()
        if conv_block is None:
            conv_block = CReLUConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.ComplexConv.convWr.stddev = 0.01  # type: ignore[assignment]
        self.conv1.ComplexConv.convWi.stddev = 0.01  # type: ignore[assignment]
        self.fc = ComplexLinear(768, num_classes)
        self.fc.l_r.stddev = 0.001  # type: ignore[assignment]
        self.fc.l_i.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x_real, x_im):
        # N x 768 x 17 x 17
        x_real = F.avg_pool2d(x_real, kernel_size=5, stride=3)
        x_im = F.avg_pool2d(x_im, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x_real, x_im = self.conv0(x_real, x_im)
        # N x 128 x 5 x 5
        x_real, x_im = self.conv1(x_real, x_im)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x_real = F.adaptive_avg_pool2d(x_real, (1, 1))
        x_im = F.adaptive_avg_pool2d(x_im, (1, 1))
        # N x 768 x 1 x 1
        x_real = torch.flatten(x_real, 1)
        x_im =torch.flatten(x_im, 1)
        # N x 768
        x_real, x_im = self.fc(x_real, x_im)
        # N x num_classes
        return x_real, x_im




class Inception3(nn.Module):

    def __init__( self, num_classes= 128, ntx=3, aux_logits = False, transform_input = False, inception_blocks = None):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                CReLUConv2d, ComplexInceptionA, ComplexInceptionB, ComplexInceptionC,
                ComplexInceptionD, ComplexInceptionE, ComplexInceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(ntx, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = ComplexMaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = ComplexMaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = ComplexAdaptativeAvgPool2d((1, 1))
        self.dropout = ComplexDropout()
        self.fc = ComplexLinear(2048, num_classes)
    #TODO
    def _transform_input(self, x_real, x_im):
        # if self.transform_input:
        #     x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        #     x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        #     x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        #     x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x_real, x_im

    def _forward(self, x_real, x_im):
        # N x m x H x W
        # N x m x 330 x 128
        x = self.Conv2d_1a_3x3(x_real, x_im)
        # N x 32 x (H-1)/2 x (W-1)/2
        # N x 32 x 164 x 63
        x = self.Conv2d_2a_3x3(*x)
        # N x 32 x 162 x 61
        x = self.Conv2d_2b_3x3(*x)
        # N x 64 x 162 x 61
        x = self.maxpool1(*x)
        # N x 64 x 80 x 30
        x = self.Conv2d_3b_1x1(*x)
        # N x 80 x 80 x 30
        x = self.Conv2d_4a_3x3(*x)
        # N x 192 x 78 x 28
        x = self.maxpool2(*x)
        # N x 192 x 38 x 13
        x = self.Mixed_5b(*x)
        # N x 256 x 38 x 13
        x = self.Mixed_5c(*x)
        # N x 288 x 38 x 13
        x = self.Mixed_5d(*x)
        # N x 288 x 38 x 13
        x = self.Mixed_6a(*x)
        # N x 768 x 18 x 6
        x = self.Mixed_6b(*x)
        # N x 768 x 18 x 6
        x = self.Mixed_6c(*x)
        # N x 768 x 18 x 6
        x = self.Mixed_6d(*x)
        # N x 768 x 18 x 6
        x = self.Mixed_6e(*x)
        # N x 768 x 18 x 6
        aux = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(*x)
        # N x 768 x 18 x 6
        x = self.Mixed_7a(*x)
        # N x 1280 x 8 x 2
        x = self.Mixed_7b(*x)
        # N x 2048 x 8 x 2
        x = self.Mixed_7c(*x)
        # N x 2048 x 8 x 2
        # Adaptive average pooling
        x = self.avgpool(*x)
        # N x 2048 x 1 x 1
        x_real, x_im = self.dropout(*x)
        # N x 2048 x 1 x 1
        x_real = torch.flatten(x_real, 1)
        x_im = torch.flatten(x_im, 1)
        # N x 2048
        x_real, x_im = self.fc(x_real, x_im)
        # N x  (num_classes)
        return x_real, x_im

    def forward(self, x_real, x_im):
        return self._forward(x_real, x_im)
