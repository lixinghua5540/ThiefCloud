import torch.nn as nn
import torch.nn.functional as F
from models.networks_other import init_weights
import torch

class cloudUNet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=False, in_channels=3, is_batchnorm=True):
        super(cloudUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        self.b1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(inplace=True)
                                )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.b2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, dilation=5, padding=5),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.b3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, kernel_size=3, dilation=5, padding=5),
                                nn.ReLU(inplace=True))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.b4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(inplace=True))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.b5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, dilation=5, padding=5),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True)
                                )

        self.b6_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.b6 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, kernel_size=3, dilation=5, padding=5),
                                nn.ReLU(inplace=True))

        self.b7_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.b7 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(inplace=True))

        self.b8_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.b8 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, dilation=5, padding=5),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True))

        self.b9_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.b9 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, dilation=2, padding=2),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, kernel_size=3, dilation=5, padding=5),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 2, kernel_size=3, dilation=1, padding=1),
                                nn.ReLU(inplace=True))


        self.halfchannel_b6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.halfchannel_b7 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.halfchannel_b8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.halfchannel_b9 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.classfier = nn.Sequential(nn.Conv2d(2, n_classes, kernel_size=1, dilation=2),
                                       nn.ReLU(inplace=True),
                                       nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):

        b1_data = self.b1(inputs)
        maxpool1_data =  self.maxpool1(b1_data)

        b2_data = self.b2(maxpool1_data)
        maxpool2_data = self.maxpool3(b2_data)

        b3_data = self.b3(maxpool2_data)
        maxpool3_data = self.maxpool3(b3_data)

        b4_data = self.b4(maxpool3_data)
        maxpool4_data = self.maxpool4(b4_data)

        b5_data = self.b5(maxpool4_data)

        b6_up_data = self.halfchannel_b6(self.b6_up(b5_data))
        b6_data = self.b6(torch.cat([b4_data, b6_up_data],1))

        b7_up_data = self.halfchannel_b7(self.b7_up(b6_data))
        b7_data = self.b7(torch.cat([b3_data, b7_up_data], 1))

        b8_up_data = self.halfchannel_b8(self.b8_up(b7_data))
        b8_data = self.b8(torch.cat([b2_data, b8_up_data], 1))

        b9_up_data = self.halfchannel_b9(self.b9_up(b8_data))
        b9_data = self.b9(torch.cat([b1_data, b9_up_data], 1))

        finall = self.classfier(b9_data)
        return finall

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p













