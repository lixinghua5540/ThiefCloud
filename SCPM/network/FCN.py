import math
import torch.nn as nn
from .unet_utils import unetConv2, unetUp
import torch.nn.functional as F
from models.networks_other import init_weights

class FCN(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=False, in_channels=3, is_batchnorm=True):
        super(FCN, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # conv1
        self.conv1_1 = nn.Conv2d(4, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # classifier 32s
        self.score_fr = nn.Conv2d(4096, n_classes, 1)

        self.upscore = nn.ConvTranspose2d(
            n_classes, n_classes, 64, stride=32, bias=False)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        # inputs:[2,4,320,320]
        x = inputs

        h = self.relu1_1(self.conv1_1(inputs))  #[2,64,518,518]
        h = self.relu1_2(self.conv1_2(h))       #[2,64,518,518]
        h1 =h
        h = self.pool1(h)                       #[2,64,259,259]
        pool1 = h
        # down to 1/2, channel=64

        h = self.relu2_1(self.conv2_1(h))       #[2,128,259,259]
        h = self.relu2_2(self.conv2_2(h))       #[2,128,259,259]
        h = self.pool2(h)                       #[2,128,130,130]
        pool2 = h
        # down to 1/4, channel=128

        h = self.relu3_1(self.conv3_1(h))       #[2,256,130,130]
        h = self.relu3_2(self.conv3_2(h))       #[2,256,130,130]
        h = self.relu3_3(self.conv3_3(h))       #[2,256,130,130]
        h = self.pool3(h)                       #[2,256, 65, 65]
        pool3 = h
        # 1/8, channel=256

        h = self.relu4_1(self.conv4_1(h))       #[2,512, 65, 65]
        h = self.relu4_2(self.conv4_2(h))       #[2,512, 65, 65]
        h = self.relu4_3(self.conv4_3(h))       #[2,512, 65, 65]
        h = self.pool4(h)                       #[2,512, 33, 33]
        pool4 = h
        # 1/16, channel=512

        h = self.relu5_1(self.conv5_1(h))       #[2,512, 33, 33]
        h = self.relu5_2(self.conv5_2(h))       #[2,512, 33, 33]
        h = self.relu5_3(self.conv5_3(h))       #[2,512, 33, 33]
        h = self.pool5(h)                       #[2,512, 17, 17]
        # 1/32, channel=512

        h = self.relu6(self.fc6(h))             # [2,4096,11,11]
        h = self.drop6(h)                       # [2,4096,11,11]
        h6=h

        # [2,4096,11,11]
        h = self.relu7(self.fc7(h))             # [2,4096,11,11]
        h = self.drop7(h)                       # [2,4096,11,11]
        h7 = h

        # classifier 32s
        h = self.score_fr(h)                    # [2,2,11,11]
        h = self.upscore(h)                    # [2,2,24,24]

        h = h[:,:,19:19 +x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p













