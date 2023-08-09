import torch
import torch.nn as nn
from utils.transforms import generate_target
import numpy as np


def pts2gau_map(pts, heatmap_size=256):
    '''
    :param pts: Tensor cuda coordinates, [batch, n_classes, 2], ranging from 0 to 1
    :return: Tensor cuda heatmaps, [b_size, n_classes, 64, 64] Gaussian map, ranging from 0 to 1
    '''
    try:
        pts_numpy = pts.clone()
        pts_numpy = pts_numpy.cpu().detach().numpy()
    except:
        pts_numpy = pts.copy()
    pts_numpy = pts_numpy * float(heatmap_size-1)
    b_size = pts_numpy.shape[0]
    # target = np.zeros((b_size, pts_numpy.shape[1], heatmap_size, heatmap_size))
    target = np.zeros((b_size, 1, heatmap_size, heatmap_size))

    for i_b in range(b_size):
        for i_c in range(pts_numpy.shape[1]):
            target[i_b][0] = generate_target(target[i_b][0], pts_numpy[i_b][i_c])
    return torch.FloatTensor(target).cuda()


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        nn.LeakyReLU(0.1),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        nn.LeakyReLU(0.1)
    )

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0),
        nn.LeakyReLU(0.1)
    )

class UpScale(nn.Module):

    def __init__(self, n_in, n_out):
        super(UpScale, self).__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=n_out * 4, out_channels=n_out * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
        )

    def forward(self, input):
        return self.upscale(input)

# class Douconv_UpScale(nn.Module):
#
#     def __init__(self, n_in, n_out):
#         super(Douconv_UpScale, self).__init__()
#         self.upscale = nn.Sequential(
#             nn.Conv2d(in_channels=n_in, out_channels=n_out * 4, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1),
#             nn.PixelShuffle(2),
#         )
#
#     def forward(self, input):
#         return self.upscale(input)


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=3):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        # self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = UpScale(512, 256)
        self.upsample2 = UpScale(256+256, 256)
        self.upsample3 = UpScale(256+128, 128)
        self.upsample4 = UpScale(128+64, 64)
        #
        # self.conv1 = nn.Sequential(
        #
        # )

        self.conv_last = nn.Sequential(
            nn.Conv2d(65, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid())


    def forward(self, x):
        x_o = x
        conv1 = self.dconv_down1(x)

        conv2 = self.dconv_down2(conv1)

        conv3 = self.dconv_down3(conv2)

        x = self.dconv_down4(conv3)

        x = self.upsample1(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.upsample3(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.upsample4(x)
        x = torch.cat([x, x_o], dim=1)

        x = self.conv_last(x)

        return x