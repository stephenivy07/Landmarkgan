import os
import os.path as osp

from torch import nn
import torch
import math
import numpy as np
import sys, copy
from patch_gan.networks import GPPatchMcResDis, Lm_conv_dis, Hm_conv_dis
from patch_gan.U_net import UNet

# from apex import amp
import matplotlib.pyplot as plt

import dsntnn
from utils.transforms import generate_target

mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D_1 = np.stack([mean_face_x, mean_face_y], axis=1).reshape(1, -1)
mean_face_cuda = torch.FloatTensor(landmarks_2D_1).cuda()

mean_face_77_828 = np.array([[156, 219], [194, 192], [241, 181], [286, 186], [329, 196], [328, 218], [288, 220], [241, 211], [194, 220], [498, 196], [541, 186], [587, 181], [635, 192], [671, 220], [631, 218], [587, 211], [541, 216], [497,222], [414, 282], [414, 333], [414, 384], [414, 437], [360, 477], [383,  483], [414, 487], [444, 483], [469, 477], [220, 284], [241, 271], [270, 263], [295, 269], [336, 288], [301, 300], [276, 303], [242, 302], [495,  290], [533, 269], [558, 263], [586, 271], [607, 284], [586, 300], [553, 303], [528, 298], [299, 589], [339, 560], [388, 542], [414, 547], [438, 542], [488, 561], [528, 589], [485, 607], [452, 618], [414, 621], [377, 618], [344, 606], [311, 584], [379, 572], [414, 572], [449, 572], [519, 586], [447, 586], [414, 585], [380, 585], [270, 283], [558, 283], [93, 360], [101, 450], [113, 524], [134, 598], [168, 672], [218, 720], [611, 720], [659, 672], [695, 598], [715, 524], [727, 450], [737, 360]], dtype=float)

mean_face_98_828 = np.array([[156, 219], [194, 192], [241, 181], [286, 186], [329, 196], [328, 218], [288, 220], [241, 211], [194, 220], [498, 196], [541, 186], [587, 181], [635, 192], [671, 220], [631, 218], [587, 211], [541, 216], [497,222], [414, 282], [414, 333], [414, 384], [414, 437], [360, 477], [383,  483], [414, 487], [444, 483], [469, 477], [220, 284], [241, 271], [270, 263], [295, 269], [336, 288], [301, 300], [276, 303], [242, 302], [495,  290], [533, 269], [558, 263], [586, 271], [607, 284], [586, 300], [553, 303], [528, 298], [299, 589], [339, 560], [388, 542], [414, 547], [438, 542], [488, 561], [528, 589], [485, 607], [452, 618], [414, 621], [377, 618], [344, 606], [311, 584], [379, 572], [414, 572], [449, 572], [519, 586], [447, 586], [414, 585], [380, 585], [270, 283], [558, 283],     [84, 318], [93, 360], [92, 405], [101, 450], [107, 487], [113, 524], [123, 560], [134, 598], [148, 638], [168, 672], [194, 701], [218, 720], [247, 739], [279, 754], [313, 772], [351, 785], [413, 790], [476, 785], [513, 773], [550, 755], [581, 740], [611, 720], [635, 701], [659, 672], [679, 640], [695, 598], [704, 562], [715, 524], [722, 488], [727, 450], [736, 407], [737, 360], [744, 318]], dtype=float)

mean_face_x = mean_face_77_828[:, 0]
mean_face_y = mean_face_77_828[:, 1]
mean_face_x -= 126.0
mean_face_x /= 574.0
mean_face_y -= 126.0
mean_face_y /= 574.0
meanface_77_1 = np.zeros(mean_face_77_828.shape)
meanface_77_1[:, 0] = mean_face_x
meanface_77_1[:, 1] = mean_face_y
# mean_face_65_cuda = meanface_65_1.reshape(1, -1)

mean_face_77_cuda = torch.FloatTensor(meanface_77_1).cuda()



mean_face_x = mean_face_98_828[:, 0]
mean_face_y = mean_face_98_828[:, 1]
mean_face_x /= 824.0
mean_face_y /= 824.0
meanface_98_1 = np.zeros(mean_face_98_828.shape)
meanface_98_1[:, 0] = mean_face_x
meanface_98_1[:, 1] = mean_face_y
# mean_face_65_cuda = meanface_65_1.reshape(1, -1)

mean_face_98_cuda = torch.FloatTensor(meanface_98_1).cuda()

class compute_2d_loss(nn.Module):
    '''
    Attention is available now!
    '''
    def __init__(self, attention_value=0, p_norm=2):
        super(compute_2d_loss, self).__init__()
        self.attention_value = attention_value
        self.p_norm = p_norm

    def forward(self, x, target, dynamic_attention=0.0, stop_index=65):
        if x.size()[-1] != 2:
            x = x.view(x.size()[0], -1, 2)
        if target.size()[-1] != 2:
            target = target.view(target.size()[0], -1, 2)

        diff = x-target
        diff_norm = torch.norm(diff, self.p_norm, 2)
        if self.attention_value != 0:
            attention_matrix = torch.FloatTensor(np.zeros(diff_norm.size())).cuda()
            attention_matrix[:, 0:stop_index] = self.attention_value + dynamic_attention - 1.0
            diff_attention = diff_norm * attention_matrix
            diff_norm_final = diff_norm + diff_attention
        else:
            diff_norm_final = diff_norm
        diff_mean = torch.sum(diff_norm_final, 1)
        diff_mean_final = torch.mean(diff_mean, 0)
        assert diff_mean_final.requires_grad == True

        return diff_mean_final


def norm_batch_tensor(b_tensor, type='norm'):
    ori_tensor_size = b_tensor.size()
    if len(ori_tensor_size) != 2:
        b_tensor = b_tensor.view(ori_tensor_size[0], -1)
    tensor_size = b_tensor.size()
    assert len(tensor_size) == 2

    mmean = torch.mean(b_tensor, 1).view(tensor_size[0], -1)
    mmin = torch.min(b_tensor, 1)[0].view(tensor_size[0], -1)
    mmax = torch.max(b_tensor, 1)[0].view(tensor_size[0], -1)
    sstd = torch.std(b_tensor, 1).view(tensor_size[0], -1)
    # mmean = torch.mean(b_tensor, 1)
    if type == 'rescaling':
        mmin = mmin.repeat(1, tensor_size[1])
        mmax = mmax.repeat(1, tensor_size[1])
        norm_batch_tensor = (b_tensor-mmin) / (mmax-mmin)
    elif type == 'norm':
        mmean = mmean.repeat(1, tensor_size[1])
        sstd = sstd.repeat(1, tensor_size[1])
        norm_batch_tensor = (b_tensor - mmean) / sstd
    elif type == 'unit':
        nnorm = torch.norm(b_tensor, 2, 1).view(tensor_size[0], -1)
        nnorm = nnorm.repeat(1, tensor_size[1])
        norm_batch_tensor = b_tensor / nnorm
    elif type == 'mean_norm':
        mmin = mmin.repeat(1, tensor_size[1])
        mmax = mmax.repeat(1, tensor_size[1])
        mmean = mmean.repeat(1, tensor_size[1])
        norm_batch_tensor = (b_tensor-mmean) / (mmax-mmin)


    if len(ori_tensor_size) != 2:
        norm_batch_tensor = norm_batch_tensor.view(ori_tensor_size)

    return norm_batch_tensor.requires_grad_()



def norm2_batch_tensor(b_tensor, type='norm'):
    ori_tensor_size = b_tensor.size()
    if len(ori_tensor_size) != 3:
        b_tensor = b_tensor.view(ori_tensor_size[0], -1, 2)
    tensor_size = b_tensor.size()
    assert len(tensor_size) == 3



    mmean = torch.mean(b_tensor, 1).view(tensor_size[0], -1, 2)
    mmin = torch.min(b_tensor, 1)[0].view(tensor_size[0], -1, 2)
    mmax = torch.max(b_tensor, 1)[0].view(tensor_size[0], -1, 2)
    sstd = torch.std(b_tensor, 1).view(tensor_size[0], -1, 2)
    # mmean = torch.mean(b_tensor, 1)
    if type == 'rescaling':
        mmin = mmin.repeat(1, tensor_size[1], 1)
        mmax = mmax.repeat(1, tensor_size[1], 1)
        norm_batch_tensor = (b_tensor-mmin) / (mmax-mmin)
    elif type == 'norm':
        mmean = mmean.repeat(1, tensor_size[1], 1)
        sstd = sstd.repeat(1, tensor_size[1], 1)
        norm_batch_tensor = (b_tensor - mmean) / sstd
    elif type == 'unit':
        nnorm = torch.norm(b_tensor, 2, 1).view(tensor_size[0], -1, 2)
        nnorm = nnorm.repeat(1, tensor_size[1], 1)
        norm_batch_tensor = b_tensor / nnorm
    elif type == 'mean_norm':
        mmin = mmin.repeat(1, tensor_size[1], 1)
        mmax = mmax.repeat(1, tensor_size[1], 1)
        mmean = mmean.repeat(1, tensor_size[1], 1)
        norm_batch_tensor = (b_tensor-mmean) / (mmax-mmin)


    # if len(ori_tensor_size) != 2:
    #     norm_batch_tensor = norm_batch_tensor.view(ori_tensor_size)

    return norm_batch_tensor.requires_grad_()


def plot_heatmap(hm, sample=0, index=0, sshow=True):
    if not sshow:
        return None
    try:
        hm_show = hm.copy()
    except:
        hm_show = hm.clone()

    # if len(hm.shape) == 4:
    hm_show = hm_show.cpu().detach().numpy()

    hm_show = hm_show[index]

    plt.figure("heatmap")
    plt.imshow(hm_show, cmap='bwr')
    plt.axis('on')
    plt.title('heatmap')
    plt.show()

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

def x1362x102(x):
    if len(x.size()) == 2:
        if x.size()[1] == 102:
            return x
        else:
            x = x.view(x.size()[0], 68, 2)
    if x.size()[1] == 68:
        x = x[:, 17:68, :]
    return x.view(x.size()[0], -1)


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

def gau_map2lm(gau_map, scale=20.0):
    '''
    :param gau_map: Tensor cuda heatmaps, [b_size, n_classes, 64, 64] Gaussian map, ranging from 0 to 1
    :param scale:
    :return: Tensor cuda coordinates, [batch, n_classes, 2], ranging from 0 to 1
    '''
    heatmap_size = float(gau_map.shape[2])
    # gau_map*= scale
    learn_heatmaps = dsntnn.flat_softmax(gau_map*scale)
    learn_cors = dsntnn.dsnt(learn_heatmaps)
    cors = ((learn_cors+1.0) * heatmap_size - 1.0) / 2.0
    cors = cors / (heatmap_size - 1.0)
    return cors, learn_cors, learn_heatmaps


def compute_dsntnn_loss(lm_dsntnn, xl, xh, attention_value=2.5):
    '''
    :param lm_dsntnn: label landmarks
    :param xl: input landmarks
    :param xh: input heatmaps
    :return:
    '''
    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(xl, lm_dsntnn) #[b_size, 98]
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(xh, lm_dsntnn, sigma_t=1.0) #[b_size, 98]
    if attention_value != 1.0:
        attention_matrix = torch.FloatTensor(np.ones(euc_losses.size())).cuda()
        attention_matrix[:, 0:65] = attention_value
        euc_losses_final = euc_losses * attention_matrix
        reg_losses_final = reg_losses * attention_matrix
    else:
        euc_losses_final = euc_losses
        reg_losses_final = reg_losses
    # Combine losses into an overall loss
    return dsntnn.average_loss(euc_losses_final + reg_losses_final)


def compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg):
    xh = dsntnn.flat_softmax(xg)
    xl = dsntnn.dsnt(xh)
    return compute_dsntnn_loss(lm_dsntnn, xl, xh)

def dsn_cors_2_norm_cors(dsn_cors, heatmap_size=64.0):
    return ((((dsn_cors + 1.0) * heatmap_size - 1.0) / 2.0) / (heatmap_size - 1))

def norm_cors_2_dsn_cors(norm_cors, heatmap_size=64.0):
    return ((norm_cors *(heatmap_size - 1.0)* 2.0 + 1.0) / heatmap_size -1.0)


class FSNet64(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet64, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
        )
        code_dims = 4 * 4 * 1024
        self.fc1 = nn.Linear(in_features=code_dims, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, images, decoder_id):
        x = self.encoder(images)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size()[0], 1024, 4, 4)
        x = self.upscale(x)
        x = self.decoder_list[decoder_id](x)
        return x


class FSNet128(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet128, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1)
        )
        code_dims = 4 * 4 * 1024
        self.fc1 = nn.Linear(in_features=code_dims, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, images, decoder_id):
        x = self.encoder(images)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size()[0], 1024, 4, 4)
        x = self.upscale(x)
        x = self.decoder_list[decoder_id](x)
        return x


class FSNet256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1)
        )
        code_dims = 4 * 4 * 1024
        self.fc1 = nn.Linear(in_features=code_dims, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, images, decoder_id):
        # Visualize dimensions
        # images: torch.Size([1, 3, 256, 256])
        x = self.encoder(images)
        # torch.Size([1, 1024, 4, 4])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 16384])
        x = self.fc1(x)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x


class FSNet_lmarks_256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv136_1024 = nn.Sequential(
            nn.Conv2d(in_channels=136, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        # torch.Size([1, 68, 2, 2])
        x = x.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv136_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x


class FSNet_lmarks_512(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_512, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv136_1024 = nn.Sequential(
            nn.Conv2d(in_channels=136, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)
        # self.upscale_256_512 = UpScale(n_in=3, n_out=3)

    def forward(self, x, decoder_id):
        # torch.Size([1, 68, 2, 2])
        x = x.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv136_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        # x = self.upscale_256_512(x)
        # torch.Size([1, 3, 512, 512])
        return x

class FSNet_lmarks_cat_meanface_256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_cat_meanface_256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024


        self.fc_1 = nn.Linear(in_features=136, out_features=136)
        self.conv238_1024 = nn.Sequential(
            nn.Conv2d(in_channels=238, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)
        self.mean_face = torch.FloatTensor(landmarks_2D_1).cuda()


    def forward(self, x, decoder_id):
        batch_size = x.shape[0]
        mean_face_batch = self.mean_face.repeat(batch_size, 1)

        x = x.view(x.size()[0], -1)

        x = self.fc_1(x)
        x = torch.cat((x, mean_face_batch), 1)
        # torch.Size([1, 68, 2, 2])
        x = x.view(x.size()[0], -1, 1, 1)

        x = self.conv238_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)

        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x

# TODO
class FSNet_gaumap2face_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_gaumap2face_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv_gaumap_1024 = nn.Sequential(
            nn.Conv2d(in_channels=(2*self.n_landmarks), out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        # import pdb; pdb.set_trace()
        if self.n_landmarks == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68,:]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, 2)
                x = x[:, 17:68, :]

        # torch.Size([1, 68, 2])
        x = x.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x


class FSNet_lmarks_meanfaceloss_256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_meanfaceloss_256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()

        self.lm_MLP = nn.Sequential(
                nn.Linear(in_features=136, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=136),
                nn.LeakyReLU(0.1),
            )

        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv136_1024 = nn.Sequential(
            nn.Conv2d(in_channels=136, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        x = x.view(x.size()[0], -1)
        x_lm_mean = self.lm_MLP(x)
        x_lm_mean = x_lm_mean.view(x.size()[0], 68, 2)
        x_lm_mean_1767 = x_lm_mean[:, 17:68, :].view(x.size()[0], -1)
        # torch.Size([1, 68, 2, 2])
        x = x_lm_mean.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv136_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x_lm_mean_1767, x


class FSNet_lmarks_meanfaceloss_1767_256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_meanfaceloss_1767_256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()

        self.lm_MLP = nn.Sequential(
                nn.Linear(in_features=102, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=102),
                nn.LeakyReLU(0.1),
            )
        self.fc_face = nn.Linear(in_features=102, out_features=102)

        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=102, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        x = x[:, 17:68, :]
        x = x.view(x.size()[0], -1)

        x_lm_mean = self.lm_MLP(x)
        x_meanface = self.fc_face(x_lm_mean)
        # x_lm_mean = x_lm_mean.view(x.size()[0], 68, 2)
        # x_lm_mean_1767 = x_lm_mean[:, 17:68, :].view(x.size()[0], -1)
        # torch.Size([1, 68, 2, 2])
        x = x_lm_mean.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x_meanface, x


class FSNet_lmarks_meanfacevae_shrareen_1767_256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_meanfacevae_shrareen_1767_256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        self.lm_decoder_list = nn.ModuleList()

        self.lm_en_MLP = nn.Sequential(
                nn.Linear(in_features=102, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=102)
            )

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                    nn.Linear(in_features=102, out_features=512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_features=512, out_features=512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_features=512, out_features=102)
                ))
        # self.fc_face = nn.Linear(in_features=102, out_features=102)

        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=102, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        # x = x[:, 17:68, :]
        x = x.view(x.size()[0], -1)

        x_lm_mean = self.lm_en_MLP(x)
        x_lm_rec = self.lm_decoder_list[decoder_id](x_lm_mean)
        # x_meanface = self.fc_face(x_lm_mean)
        # x_lm_mean = x_lm_mean.view(x.size()[0], 68, 2)
        # x_lm_mean_1767 = x_lm_mean[:, 17:68, :].view(x.size()[0], -1)
        # torch.Size([1, 68, 2, 2])
        x = x_lm_mean.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x_lm_mean, x_lm_rec, x


class FSNet_lmarks_meanfacevae_selflmen_1767_256(nn.Module):

    def __init__(self, decoder_num=2):
        super(FSNet_lmarks_meanfacevae_selflmen_1767_256, self).__init__()
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        self.lm_decoder_list = nn.ModuleList()
        self.lm_encoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.lm_encoder_list.append(nn.Sequential(
                    nn.Linear(in_features=102, out_features=512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_features=512, out_features=512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_features=512, out_features=102)
                ))

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                    nn.Linear(in_features=102, out_features=512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_features=512, out_features=512),
                    nn.LeakyReLU(0.1),
                    nn.Linear(in_features=512, out_features=102)
                ))
        # self.fc_face = nn.Linear(in_features=102, out_features=102)

        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=102, out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        # x = x[:, 17:68, :]
        x = x.view(x.size()[0], -1)

        x_lm_mean = self.lm_encoder_list[decoder_id](x)
        x_lm_rec = self.lm_decoder_list[decoder_id](x_lm_mean)
        # x_meanface = self.fc_face(x_lm_mean)
        # x_lm_mean = x_lm_mean.view(x.size()[0], 68, 2)
        # x_lm_mean_1767 = x_lm_mean[:, 17:68, :].view(x.size()[0], -1)
        # torch.Size([1, 68, 2, 2])
        x = x_lm_mean.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x_lm_mean, x_lm_rec, x


class FSNet_lmarks_1767_patchgan_256(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_1767_patchgan_256, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg
        self.gen = FSNet_lmarks_1767_256(decoder_num)
        self.dis = GPPatchMcResDis(cfg.dis)
        # self.gen_test = copy.deepcopy(self.gen)

    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='gen_test', optimizer=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        rand_add = np.random.randint(1, cfg.dis.num_classes)
        decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes

        cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
        cls_trans = cls_trans.repeat(batch_size)
        if loss_mode == 'gen_update':
            x_face_rec = self.gen(x_lm, decoder_id)
            x_face_trans= self.gen(x_lm, decoder_id_trans)
            # c_xa = self.gen.enc_content(xa)
            # s_xa = self.gen.enc_class_model(xa)
            # s_xb = self.gen.enc_class_model(xb)
            # xt = self.gen.decode(c_xa, s_xb)  # translation
            # xr = self.gen.decode(c_xa, s_xa)  # reconstruction
            l_adv_r, acc, xrec_gan_feat = self.dis.calc_gen_loss(x_face_rec, cls)

            if cfg.trans.is_transloss:
                l_adv_trans, acc_trans, _ = self.dis.calc_gen_loss(x_face_trans, cls_trans)

            # l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(x_img, cls)
            _, xreal_gan_feat = self.dis(x_img, cls)
            # _, xa_gan_feat = self.dis(xa, la)
            l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                      xreal_gan_feat.mean(3).mean(2))
            # l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                      # xb_gan_feat.mean(3).mean(2))
            l_x_rec = recon_criterion(x_face_rec, x_img)
            # l_adv =
            if cfg.trans.is_transloss:
                acc = 0.5 * (acc + acc_trans)
                l_adv = 0.5 * (l_adv_r + l_adv_trans)
            else:
                l_adv = l_adv_r
            l_total = (cfg['gan_w'] * l_adv + cfg['r_w'] * l_x_rec + cfg[
                'fm_w'] * l_c_rec)
            if cfg.is_apex:
                with amp.scale_loss(l_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, acc
        elif loss_mode == 'dis_update':
            x_img.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(x_img, cls)
            l_real = cfg['gan_w'] * l_real_pre
            if cfg.is_apex:
                with amp.scale_loss(l_real, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                l_real.backward(retain_graph=True)
            # TODO
            # l_reg_pre = self.dis.calc_grad2(resp_r, x_img)
            # l_reg = 10 * l_reg_pre
            # if cfg.is_apex:
                # with amp.scale_loss(l_reg, optimizer) as scaled_loss:
                    # scaled_loss.backward()
            # else:
                # l_reg.backward()


            with torch.no_grad():
                # c_xa = self.gen.enc_content(xa) #[bsize, 512, 16, 16]
                # s_xb = self.gen.enc_class_model(xb) # [b, 64, 1, 1]
                # xt = self.gen.decode(c_xa, s_xb) # [b, 3, 128, 128]
                x_face_rec = self.gen(x_lm, decoder_id)
                if cfg.trans.is_transloss:
                    x_face_trans = self.gen(x_lm, decoder_id_trans)

            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(x_face_rec, cls)
            if cfg.trans.is_transloss:
                l_fake_tr, acc_tr, resp_tr = self.dis.calc_dis_fake_loss(x_face_trans, cls_trans)

            if cfg.trans.is_transloss:
                l_fake = cfg['gan_w'] * (l_fake_p + l_fake_tr) / 2
            else:
                l_fake = cfg['gan_w'] * (l_fake_p)

            if cfg.is_apex:
                with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_fake.backward()
            l_total = l_fake + l_real
            if cfg.trans.is_transloss:
                acc = 0.25 * acc_f + 0.5 * acc_r + 0.25 * acc_tr
            else:
                acc = 0.5 * acc_f + 0.5 * acc_r

            return l_total, l_fake_p, l_real_pre, acc
        elif loss_mode == 'gen_test':
            self.gen.eval()
            return self.gen(x_lm, decoder_id)
            self.gen.train()
        else:
            assert 0, 'Not support operation'

    def test(self, x_lm, decoder_id):
        self.eval()
        self.gen.eval()
        # self.gen_test.eval()
        x_lm = self.gen(x_lm, decoder_id)
        self.train()
        return x_lm


class FSNet_lmarks_consistency_1767_256(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_consistency_1767_256, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.gen = FSNet_lmarks_1767_256(decoder_num)

        self.lm_encoder = lmark_encoder()
        self.lm_decoder = lmark_decoder(decoder_num)

        # self.face_lm = face_lm_model

        self.l1_loss_fn = nn.L1Loss()
        self.mean_lm = mean_face_cuda

        # TODO GAN
        # self.dis = GPPatchMcResDis(cfg.dis)
        # self.gen_test = copy.deepcopy(self.gen)

    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        # cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        # cls = cls.repeat(batch_size)

        rand_add = np.random.randint(1, cfg.dis.num_classes)
        decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes

        # cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
        # cls_trans = cls_trans.repeat(batch_size)

        mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
        lm_136 = x_lm.view(x_lm.size()[0], -1)


        if loss_mode == 'train':
            # rec loss

            xl_lm_mean = self.lm_encoder(x_lm)
            xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

            ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)

            ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_136)

            if self.cfg.trans.is_transloss:
                xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)

                xl_lm_tansback = self.lm_decoder(self.lm_encoder(xl_lm_trans), decoder_id)

                ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_136)
            # TODO replace x_lm_mean by lm_decoder result and x_lm_mean

            xf_face_rec = self.gen(xl_lm_rec, decoder_id)

            lf_rec_face = self.l1_loss_fn(xf_face_rec, x_img)

            xl_face_lm_rec = face_lm_model(xf_face_rec)
            xl_face_lm_rec_136 = xl_face_lm_rec.view(batch_size, 136)

            ll_rec_facelm = self.l1_loss_fn(xl_face_lm_rec_136, lm_136)

            if self.cfg.trans.is_transloss:

                xf_face_trans = self.gen(xl_lm_trans, decoder_id_trans)
                xl_face_trans_lm = face_lm_model(xf_face_trans)

                xl_face_trans_lm_transback = self.lm_decoder(self.lm_encoder(xl_face_trans_lm), decoder_id)

                ll_face_trans_lm_transback = self.l1_loss_fn(xl_face_trans_lm_transback, lm_136)
            if self.cfg.trans.is_transloss:
                # TODO
                pass
                # l_total = (cfg.weights.face_rec) * (lf_rec_face) + (cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback + ll_rec_facelm + ll_face_trans_lm_transback) /4.0  + (cfg.weights.mean_face) * (ll_mean_lm_in)
            else:
                l_total = (cfg.weights.face_rec) * (lf_rec_face) + (cfg.weights.lm_rec) * (ll_rec_lm_in) + (cfg.weights.lm2face2lm_rec)*(ll_rec_facelm)  + (cfg.weights.mean_face) * (ll_mean_lm_in)

            if cfg.is_apex:
                with amp.scale_loss(l_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_total.backward()

            if self.cfg.trans.is_transloss:
                return l_total, lf_rec_face, ll_rec_lm_in, ll_rec_lm_transback, ll_rec_facelm, ll_face_trans_lm_transback, ll_mean_lm_in
            else:
                return l_total, lf_rec_face, ll_rec_lm_in, ll_rec_facelm, ll_mean_lm_in


        # elif loss_mode == 'gen_test':
        #     self.gen.eval()
        #     return self.gen(x_lm, decoder_id)
        #     self.gen.train()
        else:
            assert 0, 'Not support operation'

    def test(self, x_lm, decoder_id):
        self.eval()
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        self.gen.eval()
        # self.gen_test.eval()
        x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
        face = self.gen(x_lm, decoder_id)
        self.train()
        return face




class FSNet_lmarks_1767_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_lmarks_1767_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=128),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid(),
            ))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=(2*self.n_landmarks), out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id):
        # import pdb; pdb.set_trace()
        if self.n_landmarks == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68,:]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, 2)
                x = x[:, 17:68, :]

        # torch.Size([1, 68, 2])
        x = x.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x




class FSNet_lmarks_multiscale_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_lmarks_multiscale_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num
        self.decoder_list = nn.ModuleList()
        self.decoder_list_64_3 = nn.ModuleList()

        self.decoder_list_128 = nn.ModuleList()
        self.decoder_list_128_3 = nn.ModuleList()
        self.decoder_list_256 = nn.ModuleList()
        self.decoder_list_256_3 = nn.ModuleList()



        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                UpScale(n_in=256, n_out=128),
                UpScale(n_in=128, n_out=128)
            ))

        for i in range(decoder_num):
            self.decoder_list_64_3.append(nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()
            ))

        for i in range(decoder_num):
            self.decoder_list_128.append(nn.Sequential(
                UpScale(n_in=128, n_out=128)
            ))

        for i in range(decoder_num):
            self.decoder_list_128_3.append(nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()
            ))

        for i in range(decoder_num):
            self.decoder_list_256.append(nn.Sequential(
                UpScale(n_in=128, n_out=64)
            ))

        for i in range(decoder_num):
            self.decoder_list_256_3.append(nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()
            ))




        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=(2*self.n_landmarks), out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id, is_multiscale=False):
        # import pdb; pdb.set_trace()
        # if self.n_landmarks == 51:
        #     if len(x.size()) == 3:
        #         if x.size()[1] == 68:
        #             x = x[:, 17:68,:]
        #     elif x.size()[1] == 136:
        #         x = x.view(x.size()[0], 68, 2)
        #         x = x[:, 17:68, :]
        #
        if is_multiscale:
            x = x.view(x.size()[0], -1, 1, 1)
            # torch.Size([1, 136, 1, 1])
            x = self.conv102_1024(x)
            # torch.Size([1, 1024, 1, 1])
            x = x.view(x.size()[0], -1)
            # torch.Size([1, 1024])
            x = self.fc2(x)
            # torch.Size([1, 16384])
            x = x.view(x.size()[0], 1024, 4, 4)
            # torch.Size([1, 1024, 4, 4])
            x = self.upscale(x)
            # torch.Size([1, 512, 8, 8])
            x = self.decoder_list[decoder_id](x)

            x_64 = self.decoder_list_64_3[decoder_id](x)
            x = self.decoder_list_128[decoder_id](x)
            x_128 = self.decoder_list_128_3[decoder_id](x)
            x = self.decoder_list_256[decoder_id](x)
            x = self.decoder_list_256_3[decoder_id](x)

            # torch.Size([1, 3, 256, 256])
            return x_64, x_128, x

        else:
            # torch.Size([1, 68, 2])
            x = x.view(x.size()[0], -1, 1, 1)
            # torch.Size([1, 136, 1, 1])
            x = self.conv102_1024(x)
            # torch.Size([1, 1024, 1, 1])
            x = x.view(x.size()[0], -1)
            # torch.Size([1, 1024])
            x = self.fc2(x)
            # torch.Size([1, 16384])
            x = x.view(x.size()[0], 1024, 4, 4)
            # torch.Size([1, 1024, 4, 4])
            x = self.upscale(x)
            # torch.Size([1, 512, 8, 8])
            x = self.decoder_list[decoder_id](x)
            x = self.decoder_list_128[decoder_id](x)
            x = self.decoder_list_256[decoder_id](x)
            x = self.decoder_list_256_3[decoder_id](x)

            # torch.Size([1, 3, 256, 256])
            return x


class FSNet_lmarks_deepdecoder_1767_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_lmarks_deepdecoder_1767_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num

        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                UpScale(n_in=256, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=(2*self.n_landmarks), out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

        # self.conv_add1 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1))

    def forward(self, x, decoder_id):
        if self.n_landmarks == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68,:]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, 2)
                x = x[:, 17:68, :]
        # torch.Size([1, 68, 2, 2])
        x = x.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.fc2(x)
        # torch.Size([1, 16384])
        x = x.view(x.size()[0], 1024, 4, 4)
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        # x = self.conv_add1(x)

        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x



class FSNet_lmarks_deepdecoder_fullconv_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_lmarks_deepdecoder_fullconv_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num

        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                UpScale(n_in=256, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=(2*self.n_landmarks), out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        # self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale_1 = UpScale(n_in=1024, n_out=1024)
        self.upscale_2 = UpScale(n_in=1024, n_out=1024)
        self.upscale = UpScale(n_in=1024, n_out=512)

        # self.conv_add1 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1))

    def forward(self, x, decoder_id):
        if self.n_landmarks == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68,:]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, 2)
                x = x[:, 17:68, :]
        # torch.Size([1, 68, 2, 2])
        x = x.view(x.size()[0], -1, 1, 1)
        # torch.Size([1, 136, 1, 1])
        x = self.conv102_1024(x)
        # torch.Size([1, 1024, 1, 1])
        # x = x.view(x.size()[0], -1)
        # torch.Size([1, 1024])
        x = self.upscale_1(x)
        x = self.upscale_2(x)
        # torch.Size([1, 16384])
        # torch.Size([1, 1024, 4, 4])
        x = self.upscale(x)
        # torch.Size([1, 512, 8, 8])
        # x = self.conv_add1(x)

        x = self.decoder_list[decoder_id](x)
        # torch.Size([1, 3, 256, 256])
        return x



class FSNet_c1hm_deepdecoder_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_c1hm_deepdecoder_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num

        self.decoder_list = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_list.append(UNet(1, 3))

        # self.conv_add1 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.1))

    def forward(self, x, decoder_id):
        if x.size()[-1] < 3:
            if x.size()[-1] == 1:
                x = x.view(-1, self.n_landmarks, 2)
            assert x.size()[-1] == 2
            x = pts2gau_map(x)
            x.requires_grad_()
            # torch.Size([1, 512, 8, 8])
            # x = self.conv_add1(x)

            x = self.decoder_list[decoder_id](x)
            # torch.Size([1, 3, 256, 256])
            return x
        elif x.size()[-1] == 256:
            x = self.decoder_list[decoder_id](x)
            # torch.Size([1, 3, 256, 256])
            return x


class FSNet_lmarks_deepdecoder_multiscale_256(nn.Module):

    def __init__(self, decoder_num=2, n_landmarks=51):
        super(FSNet_lmarks_deepdecoder_multiscale_256, self).__init__()
        self.n_landmarks = n_landmarks
        self.decoder_num = decoder_num

        self.decoder_list = nn.ModuleList()
        self.decoder_list_64_3 = nn.ModuleList()

        self.decoder_list_128 = nn.ModuleList()
        self.decoder_list_128_3 = nn.ModuleList()
        self.decoder_list_256 = nn.ModuleList()
        self.decoder_list_256_3 = nn.ModuleList()

        for i in range(decoder_num):
            self.decoder_list.append(nn.Sequential(
                UpScale(n_in=512, n_out=256),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                UpScale(n_in=256, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                UpScale(n_in=128, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            ))

        for i in range(decoder_num):
            self.decoder_list_64_3.append(nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()
            ))

        for i in range(decoder_num):
            self.decoder_list_128.append(nn.Sequential(
                UpScale(n_in=128, n_out=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            ))

        for i in range(decoder_num):
            self.decoder_list_128_3.append(nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()
            ))

        for i in range(decoder_num):
            self.decoder_list_256.append(nn.Sequential(
                UpScale(n_in=128, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
            ))

        for i in range(decoder_num):
            self.decoder_list_256_3.append(nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2),
                nn.Sigmoid()
            ))

        code_dims = 4 * 4 * 1024

        self.conv102_1024 = nn.Sequential(
            nn.Conv2d(in_channels=(2 * self.n_landmarks), out_channels=1024, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Linear(in_features=1024, out_features=code_dims)
        self.upscale = UpScale(n_in=1024, n_out=512)

    def forward(self, x, decoder_id, is_multiscale=False):
        # import pdb; pdb.set_trace()
        # if self.n_landmarks == 51:
        #     if len(x.size()) == 3:
        #         if x.size()[1] == 68:
        #             x = x[:, 17:68,:]
        #     elif x.size()[1] == 136:
        #         x = x.view(x.size()[0], 68, 2)
        #         x = x[:, 17:68, :]
        #
        if is_multiscale:
            x = x.view(x.size()[0], -1, 1, 1)
            # torch.Size([1, 136, 1, 1])
            x = self.conv102_1024(x)
            # torch.Size([1, 1024, 1, 1])
            x = x.view(x.size()[0], -1)
            # torch.Size([1, 1024])
            x = self.fc2(x)
            # torch.Size([1, 16384])
            x = x.view(x.size()[0], 1024, 4, 4)
            # torch.Size([1, 1024, 4, 4])
            x = self.upscale(x)
            # torch.Size([1, 512, 8, 8])
            x = self.decoder_list[decoder_id](x)

            x_64 = self.decoder_list_64_3[decoder_id](x)
            x = self.decoder_list_128[decoder_id](x)
            x_128 = self.decoder_list_128_3[decoder_id](x)
            x = self.decoder_list_256[decoder_id](x)
            x = self.decoder_list_256_3[decoder_id](x)

            # torch.Size([1, 3, 256, 256])
            return x_64, x_128, x

        else:
            # torch.Size([1, 68, 2])
            x = x.view(x.size()[0], -1, 1, 1)
            # torch.Size([1, 136, 1, 1])
            x = self.conv102_1024(x)
            # torch.Size([1, 1024, 1, 1])
            x = x.view(x.size()[0], -1)
            # torch.Size([1, 1024])
            x = self.fc2(x)
            # torch.Size([1, 16384])
            x = x.view(x.size()[0], 1024, 4, 4)
            # torch.Size([1, 1024, 4, 4])
            x = self.upscale(x)
            # torch.Size([1, 512, 8, 8])
            x = self.decoder_list[decoder_id](x)
            x = self.decoder_list_128[decoder_id](x)
            x = self.decoder_list_256[decoder_id](x)
            x = self.decoder_list_256_3[decoder_id](x)

            # torch.Size([1, 3, 256, 256])
            return x


class FSNet_lmarks_1767_patchgan_lmconsis_256(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_1767_patchgan_lmconsis_256, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg
        if cfg.gen.is_deepdecoder:
            self.gen = FSNet_lmarks_deepdecoder_1767_256(decoder_num, cfg.face.num_classes)
        else:
            self.gen = FSNet_lmarks_1767_256(decoder_num, cfg.face.num_classes)

        self.dis = GPPatchMcResDis(cfg.dis)


        self.lm_encoder = lmark_encoder_conv(cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            assert cfg.lm_AE.is_hm_AE == 0
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_77_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]


        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0

    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'gen_update':
            x_img.requires_grad_()
            lm_2d = x_lm.view(x_lm.size()[0], -1)
            if cfg.face.num_classes == 51:
                lm_2d = x1362x102(lm_2d)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:
                 mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
                 xl_lm_mean = self.lm_encoder(x_lm)
                 xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

                 # lm mean face loss
                 ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)
                 # lm autoencoder loss
                 ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

            xf_face_rec = self.gen(xl_lm_rec, decoder_id)
            # face rec loss
            lf_rec_face = self.l1_loss_fn(xf_face_rec, x_img)
            _, xl_face_lm_rec, _, xh_face_lm_rec = face_lm_model(xf_face_rec)

            # lm2face2lm loss, using dsntnn loss
            loss_dsntnn_total = 0
            for i_dsntnn in range(len(xl_face_lm_rec)):
                loss_dsntnn_total += compute_dsntnn_loss(lm_dsntnn, xl_face_lm_rec[i_dsntnn], xh_face_lm_rec[i_dsntnn])

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans), decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback)/3.0
                else:
                    xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)
                    xl_lm_tansback = self.lm_decoder(self.lm_encoder(xl_lm_trans), decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)
                    ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

            l_total_lmconsis = (cfg.weights.face_rec) * lf_rec_face + (cfg.weights.mean_face) * (ll_mean_lm_in) + (cfg.weights.lm_rec) * (ll_rec_lm_in) + (cfg.weights.lm2face2lm_rec) * (loss_dsntnn_total)

            # GAN loss
            l_adv_r, acc, xrec_gan_feat = self.dis.calc_gen_loss(xf_face_rec, cls)
            _, xreal_gan_feat = self.dis(x_img, cls)
            # unclear about this loss. copied from FUNIT:https://github.com/NVlabs/FUNIT/blob/master/funit_model.py
            l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                      xreal_gan_feat.mean(3).mean(2))

            # translation loss in GAN
            if self.cfg.trans.is_transloss:
                if not self.cfg.trans.is_lm_transloss:
                    if cfg.lm_AE.is_hm_AE:
                        xl_lm_trans = self.lm_decoder(xg_lm_mean, decoder_id_trans)[0]
                    else:
                        xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)

                xf_face_trans = self.gen(xl_lm_trans, decoder_id_trans)

                if self.cfg.trans.is_lm2face2lm_loss:
                    _, _, xg_trans, _ = face_lm_model(xf_face_trans)
                    _, _, xg_transback, _ = self.lm_decoder(self.lm_encoder(xg_lm_trans), decoder_id)
                    ll_dsntnn_lm2face2lm_trans = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_transback)

                    l_total_lmconsis += (cfg.weights.lm2face2lm_rec) * ll_dsntnn_lm2face2lm_trans / 2.0

                    # xl_face_trans_lm, _, _1 = face_lm_model(xf_face_trans)
                    #
                    # xl_face_trans_lm_transback = self.lm_decoder(self.lm_encoder(xl_face_trans_lm), decoder_id)
                    #
                    # ll_face_trans_lm_transback = self.l1_loss_fn(xl_face_trans_lm_transback, lm_2d)
                l_adv_trans, acc_trans, _ = self.dis.calc_gen_loss(xf_face_trans, cls_trans)
                acc = 0.5 * (acc + acc_trans)
                l_adv = (2.0 * l_adv_r + l_adv_trans) / 3.0
            else:
                l_adv = l_adv_r

            l_total = ((cfg.weights.gan_w) * l_adv + (cfg.weights.fm_w) * l_c_rec) + l_total_lmconsis

            if cfg.is_apex:
                with amp.scale_loss(l_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'l_adv': l_adv, 'l_c_rec': l_c_rec, 'acc': acc, 'lf_rec_face': lf_rec_face, 'll_rec_lm_in': ll_rec_lm_in, 'loss_dsntnn_total': loss_dsntnn_total, 'll_mean_lm_in': ll_mean_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen
        elif loss_mode == 'dis_update':
            x_img.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(x_img, cls)
            l_real = (cfg.weights.gan_w) * l_real_pre
            if cfg.is_apex:
                with amp.scale_loss(l_real, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                l_real.backward(retain_graph=True)

            with torch.no_grad():
                if cfg.lm_AE.is_hm_AE:
                    gm_lm = pts2gau_map(x_lm)
                    xg_lm_input = self.gaumap_encoder(gm_lm)
                    xl_lm_rec = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id)[0]
                else:
                    xl_lm_rec = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
                xf_face_rec = self.gen(xl_lm_rec, decoder_id)
                if cfg.trans.is_transloss:
                    if cfg.lm_AE.is_hm_AE:
                        xl_lm_tans = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id_trans)[0]
                    else:
                        xl_lm_tans = self.lm_decoder(self.lm_encoder(x_lm), decoder_id_trans)
                    xf_face_trans = self.gen(xl_lm_tans, decoder_id_trans)
                    xf_face_trans.requires_grad_()
            xf_face_rec.requires_grad_()
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xf_face_rec, cls)

            if cfg.trans.is_transloss:
                l_fake_tr, acc_tr, resp_tr = self.dis.calc_dis_fake_loss(xf_face_trans, cls_trans)
                l_fake = (cfg.weights.gan_w) * (2.0 * l_fake_p + l_fake_tr) / 3.0
            else:
                l_fake = (cfg.weights.gan_w) * (l_fake_p)

            if cfg.is_apex:
                with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_fake.backward()

            l_total = l_fake + l_real
            if cfg.trans.is_transloss:
                acc = 0.25 * acc_f + 0.5 * acc_r + 0.25 * acc_tr
            else:
                acc = 0.5 * acc_f + 0.5 * acc_r
            meta_dis = {'l_total': l_total, 'l_fake_p': l_fake_p, 'l_real_pre': l_real_pre, 'acc': acc, 'acc_f': acc_f, 'acc_r': acc_r}
            return meta_dis
        elif loss_mode == 'gen_test':
            self.gen.eval()
            return self.gen(x_lm, decoder_id)
        else:
            assert 0, 'Not support operation'

    def test(self, x_lm, decoder_id):
        self.eval()
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        self.gen.eval()
        try:
            if self.cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder.eval()
                x_lm = x_lm[0]
                assert len(x_lm.shape) == 3
                gm_lm = pts2gau_map(x_lm)
                xg_lm_input = self.gaumap_encoder(gm_lm)
                xl_lm_rec = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id)[0]
                face = self.gen(xl_lm_rec, decoder_id)
                return face
            else:
                x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
                face = self.gen(x_lm, decoder_id)
                return face
        except:
            # self.gen_test.eval()
            x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
            face = self.gen(x_lm, decoder_id)
            return face


class FSNet_lmarks_multiscale_patchgan_lmconsis_256(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_multiscale_patchgan_lmconsis_256, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg
        if cfg.gen.is_deepdecoder:
            self.gen = FSNet_lmarks_deepdecoder_multiscale_256(decoder_num, cfg.face.num_classes)
        else:
            self.gen = FSNet_lmarks_multiscale_256(decoder_num, cfg.face.num_classes)

        self.dis = GPPatchMcResDis(cfg.dis)
        self.lm_encoder = lmark_encoder_conv(cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            assert cfg.lm_AE.is_hm_AE == 0
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_77_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]


        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0

    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        x_img_64 = x_img[0]
        x_img_128 = x_img[1]
        x_img_256 = x_img[2]

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'gen_update':
            x_img_64.requires_grad_()
            x_img_128.requires_grad_()
            x_img_256.requires_grad_()
            lm_2d = x_lm.view(x_lm.size()[0], -1)
            if cfg.face.num_classes == 51:
                lm_2d = x1362x102(lm_2d)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:
                 mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
                 xl_lm_mean = self.lm_encoder(x_lm)
                 xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

                 # lm mean face loss
                 ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)
                 # lm autoencoder loss
                 ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

            xf_face_rec_64, xf_face_rec_128, xf_face_rec_256 = self.gen(xl_lm_rec, decoder_id, True)
            # face rec loss
            lf_rec_face = 0
            lf_rec_face += self.l1_loss_fn(xf_face_rec_256, x_img_256)
            lf_rec_face += self.l1_loss_fn(xf_face_rec_128, x_img_128)
            lf_rec_face += self.l1_loss_fn(xf_face_rec_64, x_img_64)

            _, xl_face_lm_rec, _, xh_face_lm_rec = face_lm_model(xf_face_rec_256)

            # lm2face2lm loss, using dsntnn loss
            loss_dsntnn_total = 0
            for i_dsntnn in range(len(xl_face_lm_rec)):
                loss_dsntnn_total += compute_dsntnn_loss(lm_dsntnn, xl_face_lm_rec[i_dsntnn], xh_face_lm_rec[i_dsntnn])

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans), decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback)/3.0
                else:
                    xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)
                    xl_lm_tansback = self.lm_decoder(self.lm_encoder(xl_lm_trans), decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)
                    ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

            l_total_lmconsis = (cfg.weights.face_rec) * lf_rec_face + (cfg.weights.mean_face) * (ll_mean_lm_in) + (cfg.weights.lm_rec) * (ll_rec_lm_in) + (cfg.weights.lm2face2lm_rec) * (loss_dsntnn_total)

            # GAN loss
            l_adv_r, acc, xrec_gan_feat = self.dis.calc_gen_loss(xf_face_rec_256, cls)
            _, xreal_gan_feat = self.dis(x_img_256, cls)
            # unclear about this loss. copied from FUNIT:https://github.com/NVlabs/FUNIT/blob/master/funit_model.py
            l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                      xreal_gan_feat.mean(3).mean(2))

            # translation loss in GAN
            if self.cfg.trans.is_transloss:
                if not self.cfg.trans.is_lm_transloss:
                    if cfg.lm_AE.is_hm_AE:
                        xl_lm_trans = self.lm_decoder(xg_lm_mean, decoder_id_trans)[0]
                    else:
                        xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)

                _, _, xf_face_trans = self.gen(xl_lm_trans, decoder_id_trans, True)

                if self.cfg.trans.is_lm2face2lm_loss:
                    _, _, xg_trans, _ = face_lm_model(xf_face_trans)
                    _, _, xg_transback, _ = self.lm_decoder(self.lm_encoder(xg_lm_trans), decoder_id)
                    ll_dsntnn_lm2face2lm_trans = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_transback)

                    l_total_lmconsis += (cfg.weights.lm2face2lm_rec) * ll_dsntnn_lm2face2lm_trans / 2.0

                    # xl_face_trans_lm, _, _1 = face_lm_model(xf_face_trans)
                    #
                    # xl_face_trans_lm_transback = self.lm_decoder(self.lm_encoder(xl_face_trans_lm), decoder_id)
                    #
                    # ll_face_trans_lm_transback = self.l1_loss_fn(xl_face_trans_lm_transback, lm_2d)
                l_adv_trans, acc_trans, _ = self.dis.calc_gen_loss(xf_face_trans, cls_trans)
                acc = 0.5 * (acc + acc_trans)
                l_adv = (2.0 * l_adv_r + l_adv_trans) / 3.0
            else:
                l_adv = l_adv_r

            l_total = ((cfg.weights.gan_w) * l_adv + (cfg.weights.fm_w) * l_c_rec) + l_total_lmconsis

            if cfg.is_apex:
                with amp.scale_loss(l_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'l_adv': l_adv, 'l_c_rec': l_c_rec, 'acc': acc, 'lf_rec_face': lf_rec_face, 'll_rec_lm_in': ll_rec_lm_in, 'loss_dsntnn_total': loss_dsntnn_total, 'll_mean_lm_in': ll_mean_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen
        elif loss_mode == 'dis_update':
            x_img = x_img_256
            x_img.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(x_img, cls)
            l_real = (cfg.weights.gan_w) * l_real_pre
            if cfg.is_apex:
                with amp.scale_loss(l_real, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                l_real.backward(retain_graph=True)

            with torch.no_grad():
                if cfg.lm_AE.is_hm_AE:
                    gm_lm = pts2gau_map(x_lm)
                    xg_lm_input = self.gaumap_encoder(gm_lm)
                    xl_lm_rec = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id)[0]
                else:
                    xl_lm_rec = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
                xf_face_rec = self.gen(xl_lm_rec, decoder_id)
                if cfg.trans.is_transloss:
                    if cfg.lm_AE.is_hm_AE:
                        xl_lm_tans = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id_trans)[0]
                    else:
                        xl_lm_tans = self.lm_decoder(self.lm_encoder(x_lm), decoder_id_trans)
                    xf_face_trans = self.gen(xl_lm_tans, decoder_id_trans)
                    xf_face_trans.requires_grad_()
            xf_face_rec.requires_grad_()
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xf_face_rec, cls)

            if cfg.trans.is_transloss:
                l_fake_tr, acc_tr, resp_tr = self.dis.calc_dis_fake_loss(xf_face_trans, cls_trans)
                l_fake = (cfg.weights.gan_w) * (2.0 * l_fake_p + l_fake_tr) / 3.0
            else:
                l_fake = (cfg.weights.gan_w) * (l_fake_p)

            if cfg.is_apex:
                with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_fake.backward()

            l_total = l_fake + l_real
            if cfg.trans.is_transloss:
                acc = 0.25 * acc_f + 0.5 * acc_r + 0.25 * acc_tr
            else:
                acc = 0.5 * acc_f + 0.5 * acc_r
            meta_dis = {'l_total': l_total, 'l_fake_p': l_fake_p, 'l_real_pre': l_real_pre, 'acc': acc, 'acc_f': acc_f, 'acc_r': acc_r}
            return meta_dis
        elif loss_mode == 'gen_test':
            self.gen.eval()
            return self.gen(x_lm, decoder_id)
        else:
            assert 0, 'Not support operation'

    def test(self, x_lm, decoder_id):
        self.eval()
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        self.gen.eval()
        try:
            if self.cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder.eval()
                x_lm = x_lm[0]
                assert len(x_lm.shape) == 3
                gm_lm = pts2gau_map(x_lm)
                xg_lm_input = self.gaumap_encoder(gm_lm)
                xl_lm_rec = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id)[0]
                face = self.gen(xl_lm_rec, decoder_id)
                return face
            else:
                x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
                face = self.gen(x_lm, decoder_id)
                return face
        except:
            # self.gen_test.eval()
            x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
            face = self.gen(x_lm, decoder_id)
            return face


class FSNet_lmarks_mean_lm_id_patchgan_lmconsis_256(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_mean_lm_id_patchgan_lmconsis_256, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg
        if cfg.gen.is_deepdecoder:
            self.gen = FSNet_lmarks_deepdecoder_1767_256(decoder_num, cfg.face.num_classes)
        else:
            self.gen = FSNet_lmarks_1767_256(decoder_num, cfg.face.num_classes)

        self.dis = GPPatchMcResDis(cfg.dis)

        self.lm_encoder = lmark_encoder_conv(cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            assert cfg.lm_AE.is_hm_AE == 0
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_77_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]


        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0

        self.mean_lm_id_all = []
        for i_id in range(self.decoder_num):
            mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding), 'mean_lm_id_{}.txt'.format(i_id))

            mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)

            mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()

            self.mean_lm_id_all.append(mean_lm_id)

    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'gen_update':
            x_img.requires_grad_()
            lm_2d = x_lm.view(x_lm.size()[0], -1)
            if cfg.face.num_classes == 51:
                lm_2d = x1362x102(lm_2d)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:
                 mean_lm_id_cuda = self.mean_lm_id_all[decoder_id]
                 mean_face_id_cuda_bsize = mean_lm_id_cuda.repeat(batch_size, 1)

                 xl_lm_mean = self.lm_encoder(x_lm)

                 xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

                 # lm mean face loss
                 ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, 10.0 * (lm_2d - mean_face_id_cuda_bsize))
                 # lm autoencoder loss
                 ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

            xf_face_rec = self.gen(xl_lm_rec, decoder_id)
            # face rec loss
            lf_rec_face = self.l1_loss_fn(xf_face_rec, x_img)
            _, xl_face_lm_rec, _, xh_face_lm_rec = face_lm_model(xf_face_rec)

            # lm2face2lm loss, using dsntnn loss
            loss_dsntnn_total = 0
            for i_dsntnn in range(len(xl_face_lm_rec)):
                loss_dsntnn_total += compute_dsntnn_loss(lm_dsntnn, xl_face_lm_rec[i_dsntnn], xh_face_lm_rec[i_dsntnn])

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans), decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback)/3.0
                else:
                    xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)
                    xl_lm_tansback = self.lm_decoder(self.lm_encoder(xl_lm_trans), decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)
                    ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

            l_total_lmconsis = (cfg.weights.face_rec) * lf_rec_face + (cfg.weights.mean_face) * (ll_mean_lm_in) + (cfg.weights.lm_rec) * (ll_rec_lm_in) + (cfg.weights.lm2face2lm_rec) * (loss_dsntnn_total)

            # GAN loss
            l_adv_r, acc, xrec_gan_feat = self.dis.calc_gen_loss(xf_face_rec, cls)
            _, xreal_gan_feat = self.dis(x_img, cls)
            # unclear about this loss. copied from FUNIT:https://github.com/NVlabs/FUNIT/blob/master/funit_model.py
            l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                      xreal_gan_feat.mean(3).mean(2))

            # translation loss in GAN
            if self.cfg.trans.is_transloss:
                if not self.cfg.trans.is_lm_transloss:
                    if cfg.lm_AE.is_hm_AE:
                        xl_lm_trans = self.lm_decoder(xg_lm_mean, decoder_id_trans)[0]
                    else:
                        xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)

                xf_face_trans = self.gen(xl_lm_trans, decoder_id_trans)

                if self.cfg.trans.is_lm2face2lm_loss:
                    _, _, xg_trans, _ = face_lm_model(xf_face_trans)
                    _, _, xg_transback, _ = self.lm_decoder(self.lm_encoder(xg_lm_trans), decoder_id)
                    ll_dsntnn_lm2face2lm_trans = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_transback)

                    l_total_lmconsis += (cfg.weights.lm2face2lm_rec) * ll_dsntnn_lm2face2lm_trans / 2.0

                    # xl_face_trans_lm, _, _1 = face_lm_model(xf_face_trans)
                    #
                    # xl_face_trans_lm_transback = self.lm_decoder(self.lm_encoder(xl_face_trans_lm), decoder_id)
                    #
                    # ll_face_trans_lm_transback = self.l1_loss_fn(xl_face_trans_lm_transback, lm_2d)
                l_adv_trans, acc_trans, _ = self.dis.calc_gen_loss(xf_face_trans, cls_trans)
                acc = 0.5 * (acc + acc_trans)
                l_adv = (2.0 * l_adv_r + l_adv_trans) / 3.0
            else:
                l_adv = l_adv_r

            l_total = ((cfg.weights.gan_w) * l_adv + (cfg.weights.fm_w) * l_c_rec) + l_total_lmconsis

            if cfg.is_apex:
                with amp.scale_loss(l_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'l_adv': l_adv, 'l_c_rec': l_c_rec, 'acc': acc, 'lf_rec_face': lf_rec_face, 'll_rec_lm_in': ll_rec_lm_in, 'loss_dsntnn_total': loss_dsntnn_total, 'll_mean_lm_in': ll_mean_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen
        elif loss_mode == 'dis_update':
            x_img.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(x_img, cls)
            l_real = (cfg.weights.gan_w) * l_real_pre
            if cfg.is_apex:
                with amp.scale_loss(l_real, optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                l_real.backward(retain_graph=True)

            with torch.no_grad():
                if cfg.lm_AE.is_hm_AE:
                    gm_lm = pts2gau_map(x_lm)
                    xg_lm_input = self.gaumap_encoder(gm_lm)
                    xl_lm_rec = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id)[0]
                else:
                    xl_lm_rec = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
                xf_face_rec = self.gen(xl_lm_rec, decoder_id)
                if cfg.trans.is_transloss:
                    if cfg.lm_AE.is_hm_AE:
                        xl_lm_tans = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id_trans)[0]
                    else:
                        xl_lm_tans = self.lm_decoder(self.lm_encoder(x_lm), decoder_id_trans)
                    xf_face_trans = self.gen(xl_lm_tans, decoder_id_trans)
                    xf_face_trans.requires_grad_()
            xf_face_rec.requires_grad_()
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xf_face_rec, cls)

            if cfg.trans.is_transloss:
                l_fake_tr, acc_tr, resp_tr = self.dis.calc_dis_fake_loss(xf_face_trans, cls_trans)
                l_fake = (cfg.weights.gan_w) * (2.0 * l_fake_p + l_fake_tr) / 3.0
            else:
                l_fake = (cfg.weights.gan_w) * (l_fake_p)

            if cfg.is_apex:
                with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_fake.backward()

            l_total = l_fake + l_real
            if cfg.trans.is_transloss:
                acc = 0.25 * acc_f + 0.5 * acc_r + 0.25 * acc_tr
            else:
                acc = 0.5 * acc_f + 0.5 * acc_r
            meta_dis = {'l_total': l_total, 'l_fake_p': l_fake_p, 'l_real_pre': l_real_pre, 'acc': acc, 'acc_f': acc_f, 'acc_r': acc_r}
            return meta_dis
        elif loss_mode == 'gen_test':
            self.gen.eval()
            return self.gen(x_lm, decoder_id)
        else:
            assert 0, 'Not support operation'

    def test(self, x_lm, decoder_id):
        self.eval()
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        self.gen.eval()
        try:
            if self.cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder.eval()
                x_lm = x_lm[0]
                assert len(x_lm.shape) == 3
                gm_lm = pts2gau_map(x_lm)
                xg_lm_input = self.gaumap_encoder(gm_lm)
                xl_lm_rec = self.lm_decoder(self.lm_encoder(xg_lm_input), decoder_id)[0]
                face = self.gen(xl_lm_rec, decoder_id)
                return face
            else:
                x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
                face = self.gen(x_lm, decoder_id)
                return face
        except:
            # self.gen_test.eval()
            x_lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
            face = self.gen(x_lm, decoder_id)
            return face


class FSNet_lmarks_patchgan_lmconsis_onedecoder_256_othergan(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_patchgan_lmconsis_onedecoder_256_othergan, self).__init__()
        self.decoder_num = 1
        self.cfg = cfg
        if cfg.gen.is_deepdecoder:
            if 'is_fullconv' in list(cfg.gen.keys()):
                if cfg.gen.is_fullconv:
                    self.gen = FSNet_lmarks_deepdecoder_fullconv_256(decoder_num, cfg.face.num_classes)
                else:
                    self.gen = FSNet_lmarks_deepdecoder_1767_256(decoder_num, cfg.face.num_classes)
            else:
                self.gen = FSNet_lmarks_deepdecoder_1767_256(decoder_num, cfg.face.num_classes)
        else:
            self.gen = FSNet_lmarks_1767_256(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE_v2' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE_v2:
                self.gen = FSNet_c1hm_deepdecoder_256(decoder_num, cfg.face.num_classes)

        self.dis = GPPatchMcResDis(cfg.dis)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0

    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([0]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        # rand_add = np.random.randint(1, cfg.dis.num_classes)
        # decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
        # cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
        # cls_trans = cls_trans.repeat(batch_size)

        if loss_mode == 'gen_update':
            if decoder_id == self.cfg.gen.one_decoder_id:

                x_img.requires_grad_()
                lm_2d = x_lm.view(x_lm.size()[0], -1)
                if cfg.face.num_classes == 51:
                    lm_2d = x1362x102(lm_2d)

                lm_dsntnn = norm_cors_2_dsn_cors(x_lm)


                xf_face_rec = self.gen(x_lm, 0)
                # face rec loss
                lf_rec_face = self.l1_loss_fn(xf_face_rec, x_img)
                _, xl_face_lm_rec, _, xh_face_lm_rec = face_lm_model(xf_face_rec)

                # lm2face2lm loss, using dsntnn loss
                loss_dsntnn_total = 0
                for i_dsntnn in range(len(xl_face_lm_rec)):
                    loss_dsntnn_total += compute_dsntnn_loss(lm_dsntnn, xl_face_lm_rec[i_dsntnn], xh_face_lm_rec[i_dsntnn])

                # translation loss in lm autoencoder

                l_total_lmconsis = (cfg.weights.face_rec) * lf_rec_face + (cfg.weights.lm2face2lm_rec) * (loss_dsntnn_total)

                # GAN loss
                l_adv_r, acc, xrec_gan_feat = self.dis.calc_gen_loss(xf_face_rec, cls)
                _, xreal_gan_feat = self.dis(x_img, cls)
                # unclear about this loss. copied from FUNIT:https://github.com/NVlabs/FUNIT/blob/master/funit_model.py
                l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                          xreal_gan_feat.mean(3).mean(2))



                l_adv = l_adv_r

                l_total = ((cfg.weights.gan_w) * l_adv + (cfg.weights.fm_w) * l_c_rec)

                if self.cfg.gen.is_distribued_loss:
                    l_total.backward(retain_graph=True)
                    l_total_lmconsis.backward()
                else:
                    # print('hahah')
                    l_total += l_total_lmconsis
                    l_total.backward()
                # l_adv.backward()


                # if cfg.is_apex:
                #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                    # l_total.backward()

                meta_gen = {'l_total': l_total, 'l_adv': l_adv, 'l_c_rec': l_c_rec, 'acc': acc, 'lf_rec_face': lf_rec_face, 'loss_dsntnn_total': loss_dsntnn_total}
                return meta_gen
            else:
                x_img.requires_grad_()
                lm_2d = x_lm.view(x_lm.size()[0], -1)
                if cfg.face.num_classes == 51:
                    lm_2d = x1362x102(lm_2d)

                xf_face_rec = self.gen(x_lm, 0)

                # GAN loss
                l_adv_r_other, acc_other, xrec_gan_feat = self.dis.calc_gen_loss(xf_face_rec, cls)
                _, xreal_gan_feat = self.dis(x_img, cls)
                # unclear about this loss. copied from FUNIT:https://github.com/NVlabs/FUNIT/blob/master/funit_model.py
                l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                          xreal_gan_feat.mean(3).mean(2))



                l_adv = l_adv_r_other

                l_total = ((cfg.weights.gan_w) * l_adv + (cfg.weights.fm_w) * l_c_rec)


                l_total.backward()
                # l_adv.backward()


                # if cfg.is_apex:
                #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                    # l_total.backward()

                meta_gen = {'l_adv_r_other': l_adv_r_other, 'acc_other': acc_other}
                return meta_gen

        elif loss_mode == 'dis_update':
            if decoder_id == self.cfg.gen.one_decoder_id:
                x_img.requires_grad_()
                l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(x_img, cls)
                l_real = (cfg.weights.gan_w) * l_real_pre

                if cfg.is_apex:
                    with amp.scale_loss(l_real, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    l_real.backward(retain_graph=True)

                with torch.no_grad():
                    xf_face_rec = self.gen(x_lm, 0)

                xf_face_rec.requires_grad_()
                l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xf_face_rec, cls)

                l_fake = (cfg.weights.gan_w) * (l_fake_p)

                if cfg.is_apex:
                    with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    l_fake.backward()

                l_total = l_fake + l_real
                if cfg.trans.is_transloss:
                    acc = 0.25 * acc_f + 0.5 * acc_r + 0.25 * acc_tr
                else:
                    acc = 0.5 * acc_f + 0.5 * acc_r
                meta_dis = {'l_total': l_total, 'l_fake_p': l_fake_p, 'l_real_pre': l_real_pre, 'acc': acc, 'acc_f': acc_f, 'acc_r': acc_r}
                return meta_dis
            else:
                with torch.no_grad():
                    xf_face_rec_other = self.gen(x_lm, 0)
                xf_face_rec_other.requires_grad_()

                l_fake_p_other, acc_f_other, resp_f = self.dis.calc_dis_fake_loss(xf_face_rec_other, cls)

                l_fake = (cfg.weights.gan_w) * (l_fake_p_other)

                if cfg.is_apex:
                    with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    l_fake.backward()

                meta_dis = {'l_fake_p_other': l_fake_p_other, 'acc_f_other': acc_f_other}
                return meta_dis


        elif loss_mode == 'gen_test':
            self.gen.eval()
            return self.gen(x_lm, decoder_id)
        else:
            assert 0, 'Not support operation'
    def test(self, x_lm, decoder_id=0):
        assert decoder_id == 0
        self.gen.eval()
        xf_face_rec = self.gen(x_lm, decoder_id)
        return xf_face_rec



class FSNet_lmarks_patchgan_lmconsis_onedecoder_256(nn.Module):

    def __init__(self, decoder_num, cfg):
        super(FSNet_lmarks_patchgan_lmconsis_onedecoder_256, self).__init__()
        self.decoder_num = 1
        self.cfg = cfg
        if cfg.gen.is_deepdecoder:
            self.gen = FSNet_lmarks_deepdecoder_1767_256(decoder_num, cfg.face.num_classes)
        else:
            self.gen = FSNet_lmarks_1767_256(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE_v2' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE_v2:
                self.gen = FSNet_c1hm_deepdecoder_256(decoder_num, cfg.face.num_classes)

        self.dis = GPPatchMcResDis(cfg.dis)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0

    def forward(self, x_lm, decoder_id=0, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        assert decoder_id==0
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if loss_mode == 'gen_update':
            x_img.requires_grad_()
            lm_2d = x_lm.view(x_lm.size()[0], -1)
            if cfg.face.num_classes == 51:
                lm_2d = x1362x102(lm_2d)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)


            xf_face_rec = self.gen(x_lm, decoder_id)
            # face rec loss
            lf_rec_face = self.l1_loss_fn(xf_face_rec, x_img)
            _, xl_face_lm_rec, _, xh_face_lm_rec = face_lm_model(xf_face_rec)

            # lm2face2lm loss, using dsntnn loss
            loss_dsntnn_total = 0
            for i_dsntnn in range(len(xl_face_lm_rec)):
                loss_dsntnn_total += compute_dsntnn_loss(lm_dsntnn, xl_face_lm_rec[i_dsntnn], xh_face_lm_rec[i_dsntnn])

            # translation loss in lm autoencoder

            l_total_lmconsis = (cfg.weights.face_rec) * lf_rec_face + (cfg.weights.lm2face2lm_rec) * (loss_dsntnn_total)

            # GAN loss
            l_adv_r, acc, xrec_gan_feat = self.dis.calc_gen_loss(xf_face_rec, cls)
            _, xreal_gan_feat = self.dis(x_img, cls)
            # unclear about this loss. copied from FUNIT:https://github.com/NVlabs/FUNIT/blob/master/funit_model.py
            l_c_rec = recon_criterion(xrec_gan_feat.mean(3).mean(2),
                                      xreal_gan_feat.mean(3).mean(2))



            l_adv = l_adv_r

            l_total = ((cfg.weights.gan_w) * l_adv + (cfg.weights.fm_w) * l_c_rec) + l_total_lmconsis

            l_total.backward()
            # l_adv.backward(retain_graph=True)
            # l_c_rec.backward(retain_graph=True)
            # l_total_lmconsis.backward()

            # if cfg.is_apex:
            #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
                # l_total.backward()

            meta_gen = {'l_total': l_total, 'l_adv': l_adv, 'l_c_rec': l_c_rec, 'acc': acc, 'lf_rec_face': lf_rec_face, 'loss_dsntnn_total': loss_dsntnn_total}
            return meta_gen

        elif loss_mode == 'dis_update':
            x_img.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(x_img, cls)
            l_real = l_real_pre
            # l_real = (cfg.weights.gan_w) * l_real_pre
            if cfg.is_apex:
                # with amp.scale_loss(l_real, optimizer) as scaled_loss:
                #     scaled_loss.backward(retain_graph=True)
                pass
            else:
                l_real.backward(retain_graph=True)

            with torch.no_grad():
                xf_face_rec = self.gen(x_lm, decoder_id)

            xf_face_rec.requires_grad_()
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xf_face_rec, cls)


            # l_fake = (cfg.weights.gan_w) * (l_fake_p)
            l_fake = l_fake_p

            if cfg.is_apex:
                # with amp.scale_loss(l_fake, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                pass
            else:
                l_fake.backward()

            l_total = l_fake + l_real
            if cfg.trans.is_transloss:
                acc = 0.25 * acc_f + 0.5 * acc_r + 0.25 * acc_tr
            else:
                acc = 0.5 * acc_f + 0.5 * acc_r
            meta_dis = {'l_total': l_total, 'l_fake_p': l_fake_p, 'l_real_pre': l_real_pre, 'acc': acc, 'acc_f': acc_f, 'acc_r': acc_r}
            return meta_dis
        elif loss_mode == 'gen_test':
            self.gen.eval()
            return self.gen(x_lm, decoder_id)
        else:
            assert 0, 'Not support operation'
    def test(self, x_lm, decoder_id=0):
        assert decoder_id == 0
        self.gen.eval()
        xf_face_rec = self.gen(x_lm, decoder_id)
        self.gen.train()
        return xf_face_rec



class Lm_autoencoder(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_encoder = lmark_encoder_conv(cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:
                mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
                xl_lm_en, xl_lm_mean = self.lm_encoder(x_lm)
                xl_lm_rec = self.lm_decoder(xl_lm_en, decoder_id)

                # lm mean face loss
                ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)


            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(xl_lm_en, decoder_id_trans)
                    xl_lm_tansback = self.lm_decoder(self.lm_encoder(xl_lm_trans)[0], decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)
                    ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

            l_total = (cfg.weights.mean_face) * (ll_mean_lm_in) + (
                cfg.weights.lm_rec) * (ll_rec_lm_in)



            if cfg.is_apex:
                with amp.scale_loss(l_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in, 'll_mean_lm_in': ll_mean_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen

    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        lm = self.lm_decoder(self.lm_encoder(x_lm)[0], decoder_id)
        return lm


class Lm_autoencoder_mainroad(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg
        if not 'lm2d_AE' in list(cfg.lm_AE.keys()):
            self.lm_encoder = lmark_encoder_conv_mainroad(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder_conv_mainroad(decoder_num, cfg.face.num_classes)
        else:
            print('lm AE mode', cfg.lm_AE.lm2d_AE)
            self.lm_encoder = lmark_encoder_conv_mainroad(cfg.face.num_classes, cfg.lm_AE.lm2d_AE)
            self.lm_decoder = lmark_decoder_conv_mainroad(decoder_num, cfg.face.num_classes, cfg.lm_AE.lm2d_AE)
        print('this is main road!')

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)



        self.l1_loss_fn = nn.L1Loss()
        try:
            if cfg.lm_AE.lm2d_loss != 0:
                self.l1_loss_fn = compute_2d_loss(cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
                print('lm2d_loss!', cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
        except:
            pass
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:
                mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
                xl_lm_mean = self.lm_encoder(x_lm)
                xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

                # lm mean face loss
                ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)
                    xl_lm_trans_mean = self.lm_encoder(xl_lm_trans)
                    # TODO not use ll_mean_transback in original 1210_4 version
                    ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)

                    xl_lm_tansback = self.lm_decoder(xl_lm_trans_mean, decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)
                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            ll_mean_transback.backward(retain_graph=True)
                            ll_rec_lm_transback.backward()

                l_total = (cfg.weights.mean_face) * (ll_mean_lm_in+ll_mean_transback) + (
                cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback)

            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in, 'll_mean_lm_in': ll_mean_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
                meta_gen['ll_mean_transback'] = ll_mean_transback
            return meta_gen

    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
        return lm


class Lm_autoencoder_mean_id_mainroad(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mean_id_mainroad, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_encoder = lmark_encoder_conv_mainroad(cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv_mainroad(decoder_num, cfg.face.num_classes)

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0


        self.mean_lm_id_all = []
        for i_id in range(self.decoder_num):
            mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root, str(i_id), 'mean_lm_id_{}.txt'.format(i_id))

            mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)

            mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()

            self.mean_lm_id_all.append(mean_lm_id)


        if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
            self.margin_mean_general = cfg.lm_AE.margin_mean_general
            self.margin_mean_id = cfg.lm_AE.margin_mean_id
        print('this is main road mean id!')

        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        hm_lm = pts2gau_map(x_lm, 256)
        hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:

                mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
                xl_lm_mean = self.lm_encoder(x_lm)
                xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

                # lm mean face loss
                ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)



            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    mean_lm_id_cuda = self.mean_lm_id_all[decoder_id_trans]
                    mean_face_id_cuda_bsize = mean_lm_id_cuda.repeat(batch_size, 1)

                    xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)
                    ll_trans_mean_id = self.l1_loss_fn(xl_lm_trans, mean_face_id_cuda_bsize)

                    xl_transback_mean = self.lm_encoder(xl_lm_trans)
                    ll_mean_transback = self.l1_loss_fn(xl_transback_mean, mean_face_cuda_bsize)

                    xl_lm_tansback = self.lm_decoder(xl_transback_mean, decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)

                    l_rec_total = ll_rec_lm_in + ll_rec_lm_transback
                    l_rec_total.backward(retain_graph=True)

                    # TODO seems like small batchsize is a must, 1 is best
                    if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
                        if self.margin_mean_general > ll_mean_lm_in.mean():
                            ll_mean_lm_in.backward(retain_graph=True)
                        if self.margin_mean_general > ll_mean_transback.mean():
                            ll_mean_transback.backward(retain_graph=True)
                        if self.margin_mean_id > ll_trans_mean_id.mean():
                            ll_trans_mean_id.backward()
                    else:
                        ll_trans_mean_id.backward(retain_graph=True)
                        ll_mean_total = ll_mean_lm_in + ll_mean_transback
                        ll_mean_total.backward()


                    # ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

                    # ll_mean_lm_in = ll_mean_lm_in + ll_trans_mean_id + ll_mean_transback

            l_total = (cfg.weights.mean_face) * (ll_mean_lm_in + ll_trans_mean_id + ll_mean_transback) + (
                cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback)



            # if cfg.is_apex:
            #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in, 'll_mean_lm_in': ll_mean_lm_in}
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
                meta_gen['ll_trans_mean_id'] = ll_trans_mean_id
                meta_gen['ll_mean_transback'] = ll_mean_transback

            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen

    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
        return lm


class Lm_autoencoder_mainroad_gan(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad_gan, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_encoder = lmark_encoder_conv_mainroad(cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv_mainroad(decoder_num, cfg.face.num_classes)
        self.dis = Lm_conv_dis(decoder_num, cfg.face.num_classes)

        print('this is main road with GAN!')

        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE:
                self.gaumap_encoder = gaumap_encoder(cfg.face.num_classes)
                self.lm_encoder = heatmap_encoder(cfg.face.num_classes)
                self.lm_decoder = heatmap_decoder(decoder_num, cfg.face.num_classes)

        elif not cfg.lm_AE.is_lm_ae_conv:
            self.lm_encoder = lmark_encoder(cfg.face.num_classes)
            self.lm_decoder = lmark_decoder(decoder_num, cfg.face.num_classes)

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='gen_train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        real_label = torch.LongTensor([1]).cuda()
        real_label = real_label.repeat(batch_size)

        fake_label = torch.LongTensor([0]).cuda()
        fake_label = fake_label.repeat(batch_size)

        lm_2d = x_lm.view(x_lm.size()[0], -1)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'gen_train':

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:
                mean_face_cuda_bsize = self.mean_lm.repeat(batch_size, 1)
                xl_lm_mean = self.lm_encoder(x_lm)
                xl_lm_rec = self.lm_decoder(xl_lm_mean, decoder_id)

                # lm mean face loss
                ll_mean_lm_in = self.l1_loss_fn(xl_lm_mean, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

                # TODO
                acc_mean = 2.0
                ll_mean_gan, acc_mean = self.dis.calc_encoder_loss(decoder_id, xl_lm_mean, fake_label)
                ll_rec_gan, acc_rec = self.dis.calc_decoder_loss(decoder_id, xl_lm_rec, fake_label)

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_gan.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)
                        ll_mean_gan.backward(retain_graph=True)

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(xl_lm_mean, decoder_id_trans)
                    xl_lm_trans_mean = self.lm_encoder(xl_lm_trans)
                    # TODO not use ll_mean_transback in original 1210_4 version
                    ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)

                    xl_lm_tansback = self.lm_decoder(xl_lm_trans_mean, decoder_id)
                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_tansback, lm_2d)

                    acc_mean_trans = 2.0
                    ll_mean_gan_trans, acc_mean_trans = self.dis.calc_encoder_loss(decoder_id_trans, xl_lm_trans_mean, fake_label)
                    ll_rec_gan_trans, acc_rec_trans = self.dis.calc_decoder_loss(decoder_id_trans, xl_lm_trans, fake_label)

                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            ll_mean_transback.backward(retain_graph=True)
                            ll_mean_gan_trans.backward(retain_graph=True)
                            ll_rec_gan_trans.backward(retain_graph=True)
                            ll_rec_lm_transback.backward(retain_graph=True)

                l_total = (cfg.weights.mean_face) * (ll_mean_lm_in+ll_mean_transback) + (cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback) +(cfg.weights.gan_w_lm) * (ll_mean_gan+ll_rec_gan+ll_mean_gan_trans+ll_rec_gan_trans)
                # l_total = (cfg.weights.mean_face) * (ll_mean_lm_in+ll_mean_transback) + (cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback) +(cfg.weights.gan_w_lm) * (ll_rec_gan+ll_rec_gan_trans)




            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'ll_rec_lm_in': ll_rec_lm_in, 'll_mean_lm_in': ll_mean_lm_in, 'll_mean_gan': ll_mean_gan, 'll_rec_gan': ll_rec_gan, 'acc_mean': acc_mean, 'acc_rec': acc_rec}
            # meta_gen = {'ll_rec_lm_in': ll_rec_lm_in, 'll_mean_lm_in': ll_mean_lm_in, 'll_rec_gan': ll_rec_gan, 'acc_rec': acc_rec, 'acc_mean':acc_mean}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
                meta_gen['ll_mean_transback'] = ll_mean_transback
                meta_gen['ll_mean_gan_trans'] = ll_mean_gan_trans
                meta_gen['ll_rec_gan_trans'] = ll_rec_gan_trans

                meta_gen['acc_mean_trans'] = acc_mean_trans
                meta_gen['acc_rec_trans'] = acc_rec_trans
            return meta_gen

        elif loss_mode == 'dis_train':
            l_real_dis, acc_real_dis = self.dis.calc_dis_loss(decoder_id, lm_2d, real_label)
            l_real_dis.backward(retain_graph=True)

            l_fake_dis, acc_fake_dis = 2.0, 2.0
            with torch.no_grad():
                xl_rec = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
            xl_rec.requires_grad_()
            l_fake_dis, acc_fake_dis = self.dis.calc_dis_loss(decoder_id, xl_rec, fake_label)
            l_fake_dis.backward(retain_graph=True)

            l_fake_dis_other, acc_fake_dis_other = self.dis.calc_dis_loss(decoder_id_trans, lm_2d, fake_label)
            l_fake_dis_other.backward(retain_graph=True)

            meta_dis = {'l_real_dis': l_real_dis, 'l_fake_dis': l_fake_dis, 'acc_real_dis':acc_real_dis, 'acc_fake_dis':acc_fake_dis, 'l_fake_dis_other': l_fake_dis_other, 'acc_fake_dis_other': acc_fake_dis_other}
            return meta_dis



    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        lm = self.lm_decoder(self.lm_encoder(x_lm), decoder_id)
        return lm



class Lm_autoencoder_mainroad_norm(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad_norm, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_decoder = lmark_decoder_conv_mainroad_norm(decoder_num, cfg.face.num_classes)
        print('this is main road norm')

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        mean_face_cuda_bsize_2 = self.mean_lm.repeat(batch_size, 1, 1) # [b_size, 98, 2]

        distance_nose = mean_face_cuda_bsize_2[:, 21, :] - x_lm[:, 21, :] # [b_size, 2]
        distance_nose = distance_nose.view(batch_size, 1, 2)
        distance_nose = distance_nose.repeat(1, x_lm.size()[1], 1)

        x_lm_same_nose = x_lm + distance_nose
        x_lm_same_nose.requires_grad_()

        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':
            lm_2d = x_lm_same_nose.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:

                mean_face_cuda_bsize = mean_face_cuda_bsize_2.view(x_lm.size()[0], -1)

                diff_mean = lm_2d - mean_face_cuda_bsize #[b_size, 196]

                diff_mean_normed = norm_batch_tensor(diff_mean, type=self.cfg.lm_AE.norm_type)

                anti_norm, xl_lm_rec = self.lm_decoder(diff_mean_normed, decoder_id, mean_face_cuda_bsize)

                # lm mean face loss
                # ll_mean_lm_in = self.l1_loss_fn(anti_norm, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        # ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    anti_norm_trans, xl_lm_trans = self.lm_decoder(diff_mean_normed, decoder_id_trans, mean_face_cuda_bsize)
                    #
                    # xl_lm_trans = xl_lm_trans.view(batch_size, -1, 2)
                    # distance_nose_trans =  mean_face_cuda_bsize_2[:, 21, :] - xl_lm_trans[:, 21, :]
                    # distance_nose_trans = distance_nose_trans.view(batch_size, 1, 2)
                    # distance_nose_trans = distance_nose_trans.repeat(1, x_lm.size()[1], 1)
                    #
                    # xl_lm_trans = xl_lm_trans + distance_nose_trans

                    diff_mean_trans = xl_lm_trans - mean_face_cuda_bsize

                    diff_mean_normed_trans = norm_batch_tensor(diff_mean_trans, type=self.cfg.lm_AE.norm_type)
                    anti_norm_trans_back, xl_lm_trans_back = self.lm_decoder(diff_mean_normed_trans, decoder_id, mean_face_cuda_bsize)

                    # TODO not use ll_mean_transback in original 1210_4 version
                    # ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)

                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_trans_back, lm_2d)
                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            # ll_mean_transback.backward(retain_graph=True)
                            ll_rec_lm_transback.backward()

                l_total =  (cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback)

            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
                # meta_gen['ll_mean_transback'] = ll_mean_transback
            return meta_gen




    def test(self, x_lm, decoder_id):
        x_lm = x_lm[0]
        assert x_lm.size()[-1] == 2

        batch_size = x_lm.size()[0]

        mean_face_cuda_bsize_2 = self.mean_lm.repeat(batch_size, 1, 1) # [b_size, 98, 2]

        distance_nose = mean_face_cuda_bsize_2[:, 21, :] - x_lm[:, 21, :] # [b_size, 2]
        distance_nose = distance_nose.view(batch_size, 1, 2)
        distance_nose = distance_nose.repeat(1, x_lm.size()[1], 1)


        x_lm_same_nose = x_lm + distance_nose
        lm_2d = x_lm_same_nose.view(x_lm.size()[0], -1)

        self.lm_decoder.eval()

        mean_face_cuda_bsize = mean_face_cuda_bsize_2.view(x_lm.size()[0], -1)

        diff_mean = lm_2d - mean_face_cuda_bsize  # [b_size, 196]

        diff_mean_normed = norm_batch_tensor(diff_mean, type=self.cfg.lm_AE.norm_type)

        _, lm = self.lm_decoder(diff_mean_normed, decoder_id, mean_face_cuda_bsize)

        lm = lm.view(x_lm.size()[0], -1, 2)
        lm -= distance_nose
        return lm



class Lm_autoencoder_mainroad_norm_nomeanface(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad_norm_nomeanface, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_decoder = lmark_decoder_conv_mainroad_norm_nomeanface(decoder_num, cfg.face.num_classes)
        print('this is main road norm nomeanface')

        self.l1_loss_fn = nn.L1Loss()
        try:
            if cfg.lm_AE.lm2d_loss != 0:
                self.l1_loss_fn = compute_2d_loss(cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
                print('lm2d_loss!', cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
        except:
            pass
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)


        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:

                diff_mean_normed = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type)

                xl_lm_rec = self.lm_decoder(diff_mean_normed, decoder_id)

                # lm mean face loss
                # ll_mean_lm_in = self.l1_loss_fn(anti_norm, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        # ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)

            l_total =  (cfg.weights.lm_rec) * (ll_rec_lm_in)
            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(diff_mean_normed, decoder_id_trans)
                    #
                    # xl_lm_trans = xl_lm_trans.view(batch_size, -1, 2)
                    # distance_nose_trans =  mean_face_cuda_bsize_2[:, 21, :] - xl_lm_trans[:, 21, :]
                    # distance_nose_trans = distance_nose_trans.view(batch_size, 1, 2)
                    # distance_nose_trans = distance_nose_trans.repeat(1, x_lm.size()[1], 1)
                    #
                    # xl_lm_trans = xl_lm_trans + distance_nose_trans


                    diff_mean_normed_trans = norm_batch_tensor(xl_lm_trans, type=self.cfg.lm_AE.norm_type)
                    xl_lm_trans_back = self.lm_decoder(diff_mean_normed_trans, decoder_id)

                    # TODO not use ll_mean_transback in original 1210_4 version
                    # ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)

                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_trans_back, lm_2d)
                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            # ll_mean_transback.backward(retain_graph=True)
                            ll_rec_lm_transback.backward()

                l_total =  (cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback)

            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
            else:
                meta_gen['ll_rec_lm_transback'] = torch.FloatTensor([0.0]).cuda()

                # meta_gen['ll_mean_transback'] = ll_mean_transback
            return meta_gen




    def test(self, x_lm, decoder_id):
        x_lm = x_lm[0]
        assert x_lm.size()[-1] == 2

        batch_size = x_lm.size()[0]

        # mean_face_cuda_bsize_2 = self.mean_lm.repeat(batch_size, 1, 1) # [b_size, 98, 2]
        #
        # distance_nose = mean_face_cuda_bsize_2[:, 21, :] - x_lm[:, 21, :] # [b_size, 2]
        # distance_nose = distance_nose.view(batch_size, 1, 2)
        # distance_nose = distance_nose.repeat(1, x_lm.size()[1], 1)
        #
        #
        # x_lm_same_nose = x_lm + distance_nose
        lm_2d = x_lm.view(x_lm.size()[0], -1)

        self.lm_decoder.eval()


        diff_mean_normed = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type)

        lm = self.lm_decoder(diff_mean_normed, decoder_id)

        lm = lm.view(x_lm.size()[0], -1, 2)
        return lm



class Lm_autoencoder_mainroad_0228_1(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad_0228_1, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg


        self.lm_encoder = lmark_encoder_conv_mainroad_0228(decoder_num, cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv_mainroad_norm_nomeanface(decoder_num, cfg.face.num_classes)
        print('this is 200228_1')

        # self.l1_loss_fn = nn.L1Loss()
        try:
            if cfg.lm_AE.lm2d_loss != 0:
                self.l1_loss_fn = compute_2d_loss(cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
                print('lm2d_loss!', cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
        except:
            pass
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:

                lm_target_encoder = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type).view(batch_size, -1)

                diff_mean_normed = self.lm_encoder(lm_2d).view(batch_size, -1)

                ll_encode2norm = self.l1_loss_fn(diff_mean_normed, lm_target_encoder, 2.0, 98)

                xl_lm_rec = self.lm_decoder(diff_mean_normed, decoder_id)

                # lm mean face loss
                # ll_mean_lm_in = self.l1_loss_fn(anti_norm, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)
                ll_within_id = 5.0 * ll_rec_lm_in + ll_encode2norm

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        # ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)

            l_total =  (cfg.weights.lm_rec) * (ll_within_id)
            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(diff_mean_normed, decoder_id_trans)
                    #
                    # xl_lm_trans = xl_lm_trans.view(batch_size, -1, 2)
                    # distance_nose_trans =  mean_face_cuda_bsize_2[:, 21, :] - xl_lm_trans[:, 21, :]
                    # distance_nose_trans = distance_nose_trans.view(batch_size, 1, 2)
                    # distance_nose_trans = distance_nose_trans.repeat(1, x_lm.size()[1], 1)
                    #
                    # xl_lm_trans = xl_lm_trans + distance_nose_trans

                    lm_target_encoder_trans = norm_batch_tensor(xl_lm_trans, type=self.cfg.lm_AE.norm_type).view(batch_size,
                                                                                                            -1)

                    diff_mean_normed_trans = self.lm_encoder(xl_lm_trans).view(batch_size, -1)

                    # diff_mean_normed_trans = norm_batch_tensor(xl_lm_trans, type=self.cfg.lm_AE.norm_type)
                    xl_lm_trans_back = self.lm_decoder(diff_mean_normed_trans, decoder_id)

                    # TODO not use ll_mean_transback in original 1210_4 version
                    # ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)
                    ll_encode2norm_trans = self.l1_loss_fn(diff_mean_normed_trans, lm_target_encoder_trans, 2.0, 98)

                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_trans_back, lm_2d)
                    l_cross_id = ll_encode2norm_trans + 5.0 *  ll_rec_lm_transback
                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            # ll_mean_transback.backward(retain_graph=True)
                            ll_rec_lm_transback.backward()
                if 'lm_rec_trans' in list(cfg.weights.keys()):
                    l_total =  (cfg.weights.lm_rec) * (ll_within_id) + (cfg.weights.lm_rec_trans) * (l_cross_id)
                else:
                    l_total =  (cfg.weights.lm_rec) * (ll_within_id) + (cfg.weights.lm_rec) * (l_cross_id)


            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
            else:
                meta_gen['ll_rec_lm_transback'] = torch.FloatTensor([0.0]).cuda()

                # meta_gen['ll_mean_transback'] = ll_mean_transback
            return meta_gen




    def test(self, x_lm, decoder_id):
        x_lm = x_lm[0]
        assert x_lm.size()[-1] == 2

        batch_size = x_lm.size()[0]

        # mean_face_cuda_bsize_2 = self.mean_lm.repeat(batch_size, 1, 1) # [b_size, 98, 2]
        #
        # distance_nose = mean_face_cuda_bsize_2[:, 21, :] - x_lm[:, 21, :] # [b_size, 2]
        # distance_nose = distance_nose.view(batch_size, 1, 2)
        # distance_nose = distance_nose.repeat(1, x_lm.size()[1], 1)
        #
        #
        # x_lm_same_nose = x_lm + distance_nose
        lm_2d = x_lm.view(x_lm.size()[0], -1)

        self.lm_decoder.eval()
        self.lm_encoder.eval()


        # diff_mean_normed = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type)
        diff_mean_normed = self.lm_encoder(lm_2d)


        lm = self.lm_decoder(diff_mean_normed, decoder_id)

        lm = lm.view(x_lm.size()[0], -1, 2)
        return lm



class Lm_autoencoder_mainroad_0228_2(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad_0228_2, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg


        self.lm_encoder = lmark_encoder_conv_mainroad_0228(decoder_num, cfg.face.num_classes)
        self.lm_decoder = lmark_decoder_conv_mainroad_norm_nomeanface(decoder_num, cfg.face.num_classes)
        print('this is 200228_2')

        # self.l1_loss_fn = nn.L1Loss()
        try:
            if cfg.lm_AE.lm2d_loss != 0:
                self.l1_loss_fn = compute_2d_loss(cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
                print('lm2d_loss!', cfg.lm_AE.lm2d_loss, cfg.lm_AE.lm2d_loss_norm)
        except:
            pass
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:

                lm_target_encoder = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type)

                diff_mean_normed = self.lm_encoder(lm_target_encoder).view(batch_size, -1)

                xl_lm_rec = self.lm_decoder(diff_mean_normed, decoder_id)

                # lm mean face loss
                # ll_mean_lm_in = self.l1_loss_fn(anti_norm, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)
                ll_within_id = ll_rec_lm_in

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        # ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)

            l_total =  (cfg.weights.lm_rec) * (ll_within_id)
            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(diff_mean_normed, decoder_id_trans)
                    #
                    # xl_lm_trans = xl_lm_trans.view(batch_size, -1, 2)
                    # distance_nose_trans =  mean_face_cuda_bsize_2[:, 21, :] - xl_lm_trans[:, 21, :]
                    # distance_nose_trans = distance_nose_trans.view(batch_size, 1, 2)
                    # distance_nose_trans = distance_nose_trans.repeat(1, x_lm.size()[1], 1)
                    #
                    # xl_lm_trans = xl_lm_trans + distance_nose_trans

                    lm_target_encoder_trans = norm_batch_tensor(xl_lm_trans, type=self.cfg.lm_AE.norm_type)

                    diff_mean_normed_trans = self.lm_encoder(lm_target_encoder_trans)

                    # diff_mean_normed_trans = norm_batch_tensor(xl_lm_trans, type=self.cfg.lm_AE.norm_type)
                    xl_lm_trans_back = self.lm_decoder(diff_mean_normed_trans, decoder_id)

                    # TODO not use ll_mean_transback in original 1210_4 version
                    # ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)

                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_trans_back, lm_2d)
                    l_cross_id = ll_rec_lm_transback
                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            # ll_mean_transback.backward(retain_graph=True)
                            ll_rec_lm_transback.backward()
                if 'lm_rec_trans' in list(cfg.weights.keys()):
                    l_total =  (cfg.weights.lm_rec) * (ll_within_id) + (cfg.weights.lm_rec_trans) * (l_cross_id)
                else:
                    l_total =  (cfg.weights.lm_rec) * (ll_within_id) + (cfg.weights.lm_rec) * (l_cross_id)


            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
            else:
                meta_gen['ll_rec_lm_transback'] = torch.FloatTensor([0.0]).cuda()

                # meta_gen['ll_mean_transback'] = ll_mean_transback
            return meta_gen




    def test(self, x_lm, decoder_id):
        x_lm = x_lm[0]
        assert x_lm.size()[-1] == 2

        batch_size = x_lm.size()[0]

        # mean_face_cuda_bsize_2 = self.mean_lm.repeat(batch_size, 1, 1) # [b_size, 98, 2]
        #
        # distance_nose = mean_face_cuda_bsize_2[:, 21, :] - x_lm[:, 21, :] # [b_size, 2]
        # distance_nose = distance_nose.view(batch_size, 1, 2)
        # distance_nose = distance_nose.repeat(1, x_lm.size()[1], 1)
        #
        #
        # x_lm_same_nose = x_lm + distance_nose
        lm_2d = x_lm.view(x_lm.size()[0], -1)

        self.lm_decoder.eval()
        self.lm_encoder.eval()

        lm_target_encoder = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type)

        diff_mean_normed = self.lm_encoder(lm_target_encoder)


        lm = self.lm_decoder(diff_mean_normed, decoder_id)

        lm = lm.view(x_lm.size()[0], -1, 2)
        return lm



class Lm_autoencoder_mainroad_norm222_nomeanface(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Lm_autoencoder_mainroad_norm222_nomeanface, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_decoder = lmark_decoder_conv_mainroad_norm222_nomeanface(decoder_num, cfg.face.num_classes)
        print('this is main road norm nomeanface 2222')

        self.l1_loss_fn = nn.L1Loss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0
        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)


        # print(x_lm[0][0])
        # hm_lm = pts2gau_map(x_lm, 128)
        # hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':
            lm_2d = x_lm.view(x_lm.size()[0], -1)

            lm_dsntnn = norm_cors_2_dsn_cors(x_lm)

            if cfg.lm_AE.is_hm_AE:

                mean_face_dsntnn_cuda_bsize = self.mean_lm_dsntnn.repeat(batch_size, 1, 1)
                gm_lm = pts2gau_map(x_lm)
                gm_lm.requires_grad_()

                xg_lm_input = self.gaumap_encoder(gm_lm)
                ll_dsntnn_input = compute_dsntnn_loss_from_gaumap(lm_dsntnn, xg_lm_input)

                xg_lm_mean = self.lm_encoder(xg_lm_input)
                ll_dsntnn_meanface = compute_dsntnn_loss_from_gaumap(mean_face_dsntnn_cuda_bsize, xg_lm_mean)

                ll_mean_lm_in = (3.0 * ll_dsntnn_input + ll_dsntnn_meanface) / 4.0

                xl_lm_rec, xl_lm_rec_learn, _, xh_lm_rec_learn = self.lm_decoder(xg_lm_mean, decoder_id)
                ll_rec_lm_in = compute_dsntnn_loss(lm_dsntnn, xl_lm_rec_learn, xh_lm_rec_learn)

            else:

                diff_mean_normed = norm2_batch_tensor(x_lm, type=self.cfg.lm_AE.norm_type)

                xl_lm_rec = self.lm_decoder(diff_mean_normed, decoder_id)

                # lm mean face loss
                # ll_mean_lm_in = self.l1_loss_fn(anti_norm, mean_face_cuda_bsize)
                # lm autoencoder loss
                ll_rec_lm_in = self.l1_loss_fn(xl_lm_rec, lm_2d)

                ll_mean_transback = 0
                ll_rec_lm_transback = 0

                if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                    if self.cfg.lm_AE.is_distributed_loss:
                        # ll_mean_lm_in.backward(retain_graph=True)
                        ll_rec_lm_in.backward(retain_graph=True)

            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                if cfg.lm_AE.is_hm_AE:
                    xl_lm_trans, _, xg_lm_trans, _ = self.lm_decoder(xg_lm_mean, decoder_id_trans)

                    _, xl_lm_transback_learn, _, xh_lm_trans_back = self.lm_decoder(self.lm_encoder(xg_lm_trans),
                                                                                    decoder_id)

                    ll_rec_lm_transback = compute_dsntnn_loss(lm_dsntnn, xl_lm_transback_learn, xh_lm_trans_back)

                    ll_rec_lm_in = (2.0 * ll_rec_lm_in + ll_rec_lm_transback) / 3.0
                else:
                    xl_lm_trans = self.lm_decoder(diff_mean_normed, decoder_id_trans)
                    #
                    # xl_lm_trans = xl_lm_trans.view(batch_size, -1, 2)
                    # distance_nose_trans =  mean_face_cuda_bsize_2[:, 21, :] - xl_lm_trans[:, 21, :]
                    # distance_nose_trans = distance_nose_trans.view(batch_size, 1, 2)
                    # distance_nose_trans = distance_nose_trans.repeat(1, x_lm.size()[1], 1)
                    #
                    # xl_lm_trans = xl_lm_trans + distance_nose_trans


                    diff_mean_normed_trans = norm2_batch_tensor(xl_lm_trans, type=self.cfg.lm_AE.norm_type)
                    xl_lm_trans_back = self.lm_decoder(diff_mean_normed_trans, decoder_id)

                    # TODO not use ll_mean_transback in original 1210_4 version
                    # ll_mean_transback = self.l1_loss_fn(xl_lm_trans_mean, mean_face_cuda_bsize)

                    ll_rec_lm_transback = self.l1_loss_fn(xl_lm_trans_back, lm_2d)
                    if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                        if self.cfg.lm_AE.is_distributed_loss:
                            # ll_mean_transback.backward(retain_graph=True)
                            ll_rec_lm_transback.backward()

                l_total =  (cfg.weights.lm_rec) * (ll_rec_lm_in + ll_rec_lm_transback)

            # TODO not use distributed loss in 1210_4 version
            if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if not self.cfg.lm_AE.is_distributed_loss:
                    l_total.backward()
            else:
                l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': ll_rec_lm_in}
            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = ll_rec_lm_transback
                # meta_gen['ll_mean_transback'] = ll_mean_transback
            return meta_gen




    def test(self, x_lm, decoder_id):
        x_lm = x_lm[0]
        assert x_lm.size()[-1] == 2

        batch_size = x_lm.size()[0]

        # mean_face_cuda_bsize_2 = self.mean_lm.repeat(batch_size, 1, 1) # [b_size, 98, 2]
        #
        # distance_nose = mean_face_cuda_bsize_2[:, 21, :] - x_lm[:, 21, :] # [b_size, 2]
        # distance_nose = distance_nose.view(batch_size, 1, 2)
        # distance_nose = distance_nose.repeat(1, x_lm.size()[1], 1)
        #
        #
        # x_lm_same_nose = x_lm + distance_nose
        lm_2d = x_lm.view(x_lm.size()[0], -1)

        self.lm_decoder.eval()


        diff_mean_normed = norm_batch_tensor(lm_2d, type=self.cfg.lm_AE.norm_type)

        lm = self.lm_decoder(diff_mean_normed, decoder_id)

        lm = lm.view(x_lm.size()[0], -1, 2)
        return lm



class Hm_autoencoder_mainroad(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Hm_autoencoder_mainroad, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_encoder = c1heatmap_encoder(cfg.face.num_classes)
        self.lm_decoder = c1heatmap_decoder(decoder_num, cfg.face.num_classes)



        self.l1_loss_fn = nn.L1Loss()
        self.MSE_loss_fn = nn.MSELoss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, cfg.face.num_classes, 2)

        self.mean_lm_hm = pts2gau_map(self.mean_lm)
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0


        self.mean_lm_id_all = []
        self.mean_hm_id_all = []
        for i_id in range(self.decoder_num):
            mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root, str(i_id), 'mean_lm_id_{}.txt'.format(i_id))

            mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes),  2)
            mean_hm_id = pts2gau_map(mean_lm_id)
            self.mean_hm_id_all.append(mean_hm_id)
            mean_lm_id = mean_lm_id.reshape(1, (cfg.face.num_classes) * 2)


            mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()

            self.mean_lm_id_all.append(mean_lm_id)


        if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
            self.margin_mean_general = cfg.lm_AE.margin_mean_general
            self.margin_mean_id = cfg.lm_AE.margin_mean_id
        print('this is HM main road mean id!')

        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        hm_lm = pts2gau_map(x_lm, 256)
        hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':

            self.mean_lm_hm.requires_grad_()

            mean_face_cuda_bsize = self.mean_lm_hm.repeat(batch_size, 1, 1, 1)
            #
            mean_hm_id_self_cuda = self.mean_hm_id_all[decoder_id]
            # mean_hm_id_self_cuda.requires_grad_()
            mean_hm_id_self_cuda = mean_hm_id_self_cuda.repeat(batch_size, 1, 1, 1)

            xh_lm_mean = self.lm_encoder(hm_lm)

            xh_lm_rec = self.lm_decoder(xh_lm_mean, decoder_id)

            # lm mean face loss
            lh_mean_lm_in = self.MSE_loss_fn(xh_lm_mean, mean_face_cuda_bsize)
            # lm autoencoder loss
            lh_rec_lm_in = self.MSE_loss_fn(xh_lm_rec, hm_lm)

            l_dyna_margin = self.MSE_loss_fn(hm_lm, mean_hm_id_self_cuda)



            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                mean_hm_id_cuda = self.mean_hm_id_all[decoder_id_trans]
                mean_hm_id_cuda.requires_grad_()
                mean_face_id_cuda_bsize = mean_hm_id_cuda.repeat(batch_size, 1, 1, 1)

                xh_lm_trans = self.lm_decoder(xh_lm_mean, decoder_id_trans)
                lh_trans_mean_id = self.MSE_loss_fn(xh_lm_trans, mean_face_id_cuda_bsize)

                xh_transback_mean = self.lm_encoder(xh_lm_trans)
                lh_mean_transback = self.MSE_loss_fn(xh_transback_mean, mean_face_cuda_bsize)

                xh_lm_tansback = self.lm_decoder(xh_transback_mean, decoder_id)
                lh_rec_lm_transback = self.MSE_loss_fn(xh_lm_tansback, hm_lm)

                l_rec_total = lh_rec_lm_in + lh_rec_lm_transback
                l_rec_total.backward(retain_graph=True)

                # TODO seems like small batchsize is a must, 1 is best
                if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
                    if self.margin_mean_general < lh_mean_lm_in.mean():
                        lh_mean_lm_in.backward(retain_graph=True)
                    if self.margin_mean_general < lh_mean_transback.mean():
                        lh_mean_transback.backward(retain_graph=True)
                    if l_dyna_margin < lh_trans_mean_id.mean():
                        lh_trans_mean_id.backward()
                else:
                    lh_trans_mean_id.backward(retain_graph=True)
                    lh_mean_total = ll_mean_lm_in + ll_mean_transback
                    lh_mean_total.backward()


                    # ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

                    # ll_mean_lm_in = ll_mean_lm_in + ll_trans_mean_id + ll_mean_transback

            l_total = (cfg.weights.mean_face) * (lh_mean_lm_in + lh_trans_mean_id + lh_mean_transback) + (
                cfg.weights.lm_rec) * (lh_rec_lm_in + lh_rec_lm_transback)



            # if cfg.is_apex:
            #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': lh_rec_lm_in, 'll_mean_lm_in': lh_mean_lm_in}
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = lh_rec_lm_transback
                meta_gen['ll_trans_mean_id'] = lh_trans_mean_id
                meta_gen['ll_mean_transback'] = lh_mean_transback

            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen

    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        hm_lm = pts2gau_map(x_lm[:, 0, :, :], 256)
        lm = self.lm_decoder(self.lm_encoder(hm_lm), decoder_id)
        return lm


class Hm_autoencoder_mean_id_mainroad(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Hm_autoencoder_mean_id_mainroad, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_encoder = c1heatmap_encoder(cfg.face.num_classes)
        self.lm_decoder = c1heatmap_decoder(decoder_num, cfg.face.num_classes)



        self.l1_loss_fn = nn.L1Loss()
        self.MSE_loss_fn = nn.MSELoss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, cfg.face.num_classes, 2)

        self.mean_lm_hm = pts2gau_map(self.mean_lm)
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0


        self.mean_lm_id_all = []
        self.mean_hm_id_all = []
        for i_id in range(self.decoder_num):
            mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root, str(i_id), 'mean_lm_id_{}.txt'.format(i_id))

            mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes),  2)
            mean_hm_id = pts2gau_map(mean_lm_id)
            self.mean_hm_id_all.append(mean_hm_id)
            mean_lm_id = mean_lm_id.reshape(1, (cfg.face.num_classes) * 2)


            mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()

            self.mean_lm_id_all.append(mean_lm_id)


        if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
            self.margin_mean_general = cfg.lm_AE.margin_mean_general
            self.margin_mean_id = cfg.lm_AE.margin_mean_id
        print('this is HM main road mean id!')

        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        hm_lm = pts2gau_map(x_lm, 256)
        hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'train':

            self.mean_lm_hm.requires_grad_()

            mean_face_cuda_bsize = self.mean_lm_hm.repeat(batch_size, 1, 1, 1)
            #
            mean_hm_id_self_cuda = self.mean_hm_id_all[decoder_id]
            # mean_hm_id_self_cuda.requires_grad_()
            mean_hm_id_self_cuda = mean_hm_id_self_cuda.repeat(batch_size, 1, 1, 1)

            xh_lm_mean = self.lm_encoder(hm_lm)

            xh_lm_rec = self.lm_decoder(xh_lm_mean, decoder_id)

            # lm mean face loss
            lh_mean_lm_in = self.MSE_loss_fn(xh_lm_mean, mean_face_cuda_bsize)
            # lm autoencoder loss
            lh_rec_lm_in = self.MSE_loss_fn(xh_lm_rec, hm_lm)

            l_dyna_margin = self.MSE_loss_fn(hm_lm, mean_hm_id_self_cuda)



            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                mean_hm_id_cuda = self.mean_hm_id_all[decoder_id_trans]
                mean_hm_id_cuda.requires_grad_()
                mean_face_id_cuda_bsize = mean_hm_id_cuda.repeat(batch_size, 1, 1, 1)

                xh_lm_trans = self.lm_decoder(xh_lm_mean, decoder_id_trans)
                lh_trans_mean_id = self.MSE_loss_fn(xh_lm_trans, mean_face_id_cuda_bsize)

                xh_transback_mean = self.lm_encoder(xh_lm_trans)
                lh_mean_transback = self.MSE_loss_fn(xh_transback_mean, mean_face_cuda_bsize)

                xh_lm_tansback = self.lm_decoder(xh_transback_mean, decoder_id)
                lh_rec_lm_transback = self.MSE_loss_fn(xh_lm_tansback, hm_lm)

                l_rec_total = lh_rec_lm_in + lh_rec_lm_transback
                l_rec_total.backward(retain_graph=True)

                # TODO seems like small batchsize is a must, 1 is best
                if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
                    if self.margin_mean_general < lh_mean_lm_in.mean():
                        lh_mean_lm_in.backward(retain_graph=True)
                    if self.margin_mean_general < lh_mean_transback.mean():
                        lh_mean_transback.backward(retain_graph=True)
                    if l_dyna_margin < lh_trans_mean_id.mean():
                        lh_trans_mean_id.backward()
                else:
                    lh_trans_mean_id.backward(retain_graph=True)
                    lh_mean_total = ll_mean_lm_in + ll_mean_transback
                    lh_mean_total.backward()


                    # ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

                    # ll_mean_lm_in = ll_mean_lm_in + ll_trans_mean_id + ll_mean_transback

            l_total = (cfg.weights.mean_face) * (lh_mean_lm_in + lh_trans_mean_id + lh_mean_transback) + (
                cfg.weights.lm_rec) * (lh_rec_lm_in + lh_rec_lm_transback)



            # if cfg.is_apex:
            #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': lh_rec_lm_in, 'll_mean_lm_in': lh_mean_lm_in}
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = lh_rec_lm_transback
                meta_gen['ll_trans_mean_id'] = lh_trans_mean_id
                meta_gen['ll_mean_transback'] = lh_mean_transback

            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen

    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        hm_lm = pts2gau_map(x_lm[:, 0, :, :], 256)
        lm = self.lm_decoder(self.lm_encoder(hm_lm), decoder_id)
        return lm


class Hm_autoencoder_mainroad_gan(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Hm_autoencoder_mainroad_gan, self).__init__()
        self.decoder_num = decoder_num
        self.cfg = cfg

        self.lm_encoder = c1heatmap_encoder(cfg.face.num_classes)
        self.lm_decoder = c1heatmap_decoder(decoder_num, cfg.face.num_classes)
        self.dis = Hm_conv_dis(decoder_num, cfg.face.num_classes)



        self.l1_loss_fn = nn.L1Loss()
        self.MSE_loss_fn = nn.MSELoss()
        if cfg.face.num_classes == 51 or cfg.face.num_classes == 68:
            self.mean_lm = mean_face_cuda
        # elif cfg.face.num_classes == 65 or cfg.face.num_classes == 57:
        else:
            self.mean_lm = mean_face_98_cuda
            self.mean_lm = self.mean_lm[0:cfg.face.num_classes, :]

        assert self.mean_lm.size()[0] == cfg.face.num_classes
        self.mean_lm = self.mean_lm.view(1, cfg.face.num_classes, 2)

        self.mean_lm_hm = pts2gau_map(self.mean_lm)
        self.mean_lm = self.mean_lm.view(1, -1)

        self.mean_lm_dsntnn = (self.mean_lm.view(1, cfg.face.num_classes, 2)) * (cfg.face.heatmap_size - 1.0)

        self.mean_lm_dsntnn = (self.mean_lm_dsntnn * 2.0 + 1.0) / (torch.Tensor([cfg.face.heatmap_size]).cuda()) - 1.0


        # self.mean_lm_id_all = []
        # self.mean_hm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root, str(i_id), 'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes),  2)
        #     mean_hm_id = pts2gau_map(mean_lm_id)
        #     self.mean_hm_id_all.append(mean_hm_id)
        #     mean_lm_id = mean_lm_id.reshape(1, (cfg.face.num_classes) * 2)
        #
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


        if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
            self.margin_mean_general = cfg.lm_AE.margin_mean_general
            self.margin_mean_id = cfg.lm_AE.margin_mean_id
        print('this is HM main road mean id and GAN!')

        #
        # self.mean_lm_id_all = []
        # for i_id in range(self.decoder_num):
        #     mean_lm_id_path = osp.join(cfg.lm_AE.mean_lm_id_root.format(cfg.size, cfg.padding),
        #                                'mean_lm_id_{}.txt'.format(i_id))
        #
        #     mean_lm_id = np.loadtxt(mean_lm_id_path).reshape(1, (cfg.face.num_classes) * 2)
        #
        #     mean_lm_id = torch.FloatTensor((mean_lm_id)).cuda()
        #
        #     self.mean_lm_id_all.append(mean_lm_id)


    def forward(self, x_lm, decoder_id, cfg=None, x_img=None, loss_mode='gen_train', optimizer=None, face_lm_model=None):
        cls = torch.LongTensor([decoder_id]).cuda()
        batch_size = x_lm.shape[0]
        cls = cls.repeat(batch_size)

        real_label = torch.LongTensor([1]).cuda()
        real_label = real_label.repeat(batch_size)

        fake_label = torch.LongTensor([0]).cuda()
        fake_label = fake_label.repeat(batch_size)

        lm_2d = x_lm.view(x_lm.size()[0], -1)

        if self.cfg.trans.is_transloss or self.cfg.trans.is_lm_transloss:
            rand_add = np.random.randint(1, cfg.dis.num_classes)
            decoder_id_trans = (decoder_id + rand_add) % cfg.dis.num_classes
            cls_trans = torch.LongTensor([decoder_id_trans]).cuda()
            cls_trans = cls_trans.repeat(batch_size)

        # print(x_lm[0][0])
        hm_lm = pts2gau_map(x_lm, 256)
        hm_lm.requires_grad_()
        # # hhm_lm = dsntnn.flat_softmax(hm_lm)
        # # cors = dsntnn.dsnt(hhm_lm)
        # # print(cors[0][0])
        # plot_heatmap(hm_lm)
        #
        # # plot_heatmap(hhm_lm)
        # xl_lm2gau_map2lm, _, _1 = gau_map2lm(hm_lm)
        # print(xl_lm2gau_map2lm[0][0])
        #
        # chazhi = x_lm - xl_lm2gau_map2lm
        # print(chazhi.max())
        # print(chazhi.min())

        # plot_heatmap(hm_lm)
        # plot_heatmap(_1)

        if loss_mode == 'gen_train':

            self.mean_lm_hm.requires_grad_()

            mean_face_cuda_bsize = self.mean_lm_hm.repeat(batch_size, 1, 1, 1)
            #
            # mean_hm_id_self_cuda = self.mean_hm_id_all[decoder_id]
            # mean_hm_id_self_cuda.requires_grad_()
            # mean_hm_id_self_cuda = mean_hm_id_self_cuda.repeat(batch_size, 1, 1, 1)

            xh_lm_mean = self.lm_encoder(hm_lm)

            xh_lm_rec = self.lm_decoder(xh_lm_mean, decoder_id)

            # lm mean face loss
            lh_mean_lm_in = self.MSE_loss_fn(xh_lm_mean, mean_face_cuda_bsize)
            # lm autoencoder loss
            lh_rec_lm_in = self.MSE_loss_fn(xh_lm_rec, hm_lm)

            acc_mean = 2.0
            # lh_mean_gan, acc_mean = self.dis.calc_encoder_loss(decoder_id, xh_lm_mean, fake_label)
            lh_rec_gan, acc_rec = self.dis.calc_decoder_loss(decoder_id, xh_lm_rec, fake_label)

            lh_mean_transback = 0
            lh_rec_lm_transback = 0

            # if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
            if self.cfg.lm_AE.is_distributed_loss:
                lh_mean_lm_in.backward(retain_graph=True)
                lh_rec_lm_in.backward(retain_graph=True)
                # lh_mean_gan.backward(retain_graph=True)
                lh_rec_gan.backward(retain_graph=True)


            # translation loss in lm autoencoder
            if self.cfg.trans.is_lm_transloss:
                # mean_hm_id_cuda = self.mean_hm_id_all[decoder_id_trans]
                # mean_hm_id_cuda.requires_grad_()
                # mean_face_id_cuda_bsize = mean_hm_id_cuda.repeat(batch_size, 1, 1, 1)

                xh_lm_trans = self.lm_decoder(xh_lm_mean, decoder_id_trans)
                # lh_trans_mean_id = self.MSE_loss_fn(xh_lm_trans, mean_face_id_cuda_bsize)

                xh_transback_mean = self.lm_encoder(xh_lm_trans)
                lh_mean_transback = self.MSE_loss_fn(xh_transback_mean, mean_face_cuda_bsize)

                xh_lm_tansback = self.lm_decoder(xh_transback_mean, decoder_id)
                lh_rec_lm_transback = self.MSE_loss_fn(xh_lm_tansback, hm_lm)

                acc_mean_trans = 2.0
                # lh_mean_gan_trans, acc_mean_trans = self.dis.calc_encoder_loss(decoder_id_trans, xh_transback_mean,fake_label)
                lh_rec_gan_trans, acc_rec_trans = self.dis.calc_decoder_loss(decoder_id_trans, xh_lm_trans, fake_label)

                # if 'is_distributed_loss' in list(self.cfg.lm_AE.keys()):
                if self.cfg.lm_AE.is_distributed_loss:
                    lh_mean_transback.backward(retain_graph=True)
                    # lh_mean_gan_trans.backward(retain_graph=True)
                    lh_rec_gan_trans.backward(retain_graph=True)
                    lh_rec_lm_transback.backward()


                # l_rec_total = lh_rec_lm_in + lh_rec_lm_transback
                # l_rec_total.backward(retain_graph=True)

                # TODO seems like small batchsize is a must, 1 is best
                # if 'margin_mean_general' in list(self.cfg.lm_AE.keys()):
                #     if self.margin_mean_general < lh_mean_lm_in.mean():
                #         lh_mean_lm_in.backward(retain_graph=True)
                #     if self.margin_mean_general < lh_mean_transback.mean():
                #         lh_mean_transback.backward(retain_graph=True)
                #     if l_dyna_margin < lh_trans_mean_id.mean():
                #         lh_trans_mean_id.backward()
                # else:
                #     lh_trans_mean_id.backward(retain_graph=True)
                #     lh_mean_total = ll_mean_lm_in + ll_mean_transback
                #     lh_mean_total.backward()


                    # ll_rec_lm_in = ll_rec_lm_in + ll_rec_lm_transback

                    # ll_mean_lm_in = ll_mean_lm_in + ll_trans_mean_id + ll_mean_transback

            # l_total = (cfg.weights.mean_face) * (lh_mean_lm_in + lh_mean_transback) + (cfg.weights.lm_rec) * (lh_rec_lm_in + lh_rec_lm_transback) + (cfg.weights.gan_w_lm) * (lh_mean_gan+lh_rec_gan+lh_mean_gan_trans+lh_rec_gan_trans)
            l_total = (cfg.weights.mean_face) * (lh_mean_lm_in + lh_mean_transback) + (cfg.weights.lm_rec) * (lh_rec_lm_in + lh_rec_lm_transback) + (cfg.weights.gan_w_lm) * (lh_rec_gan+lh_rec_gan_trans)

            if not self.cfg.lm_AE.is_distributed_loss:
                l_total.backward()



            # if cfg.is_apex:
            #     with amp.scale_loss(l_total, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     l_total.backward()

            meta_gen = {'l_total': l_total, 'll_rec_lm_in': lh_rec_lm_in, 'll_mean_lm_in': lh_mean_lm_in, 'll_rec_gan': lh_rec_gan, 'acc_rec': acc_rec, 'acc_mean': acc_mean}
            if self.cfg.trans.is_lm_transloss:
                meta_gen['ll_rec_lm_transback'] = lh_rec_lm_transback
                # meta_gen['ll_trans_mean_id'] = lh_trans_mean_id
                meta_gen['ll_mean_transback'] = lh_mean_transback

                # meta_gen['ll_mean_gan_trans'] = lh_mean_gan_trans
                meta_gen['ll_rec_gan_trans'] = lh_rec_gan_trans

                meta_gen['acc_mean_trans'] = acc_mean_trans
                meta_gen['acc_rec_trans'] = acc_rec_trans

            if cfg.lm_AE.is_hm_AE:
                meta_gen['ll_dsntnn_input'] = ll_dsntnn_input

            return meta_gen

        elif loss_mode == 'dis_train':
            l_real_dis, acc_real_dis = self.dis.calc_dis_loss(decoder_id, hm_lm, real_label)
            l_real_dis.backward(retain_graph=True)

            l_fake_dis, acc_fake_dis = 2.0, 2.0

            with torch.no_grad():
                xl_rec = self.lm_decoder(self.lm_encoder(hm_lm), decoder_id)
            l_fake_dis, acc_fake_dis = self.dis.calc_dis_loss(decoder_id, xl_rec, fake_label)
            l_fake_dis.backward(retain_graph=True)


            l_fake_dis_other, acc_fake_dis_other= self.dis.calc_dis_loss(decoder_id_trans, hm_lm, fake_label)
            l_fake_dis_other.backward(retain_graph=True)

            meta_dis = {'l_real_dis': l_real_dis, 'l_fake_dis': l_fake_dis, 'acc_real_dis':acc_real_dis, 'acc_fake_dis':acc_fake_dis, 'l_fake_dis_other': l_fake_dis_other, 'acc_fake_dis_other': acc_fake_dis_other}
            return meta_dis


    def test(self, x_lm, decoder_id):
        self.lm_encoder.eval()
        self.lm_decoder.eval()
        hm_lm = pts2gau_map(x_lm[:, 0, :, :], 256)
        lm = self.lm_decoder(self.lm_encoder(hm_lm), decoder_id)
        return lm






class gaumap_encoder(nn.Module):
    def __init__(self, nclasses=51):
        super(gaumap_encoder, self).__init__()
        self.nclasses = nclasses

        self.hm_256 = nn.Sequential(
            nn.Conv2d(in_channels=self.nclasses, out_channels=self.nclasses, kernel_size=5, stride=4, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.nclasses, out_channels=self.nclasses, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.hm_256(x)
        return x


class heatmap_encoder(nn.Module):
    def __init__(self, nclasses=51):
        super(heatmap_encoder, self).__init__()
        self.nclasses = nclasses

        self.hm64_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.nclasses, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=self.nclasses, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.nclasses, out_channels=self.nclasses, kernel_size=1, bias=False)
        )


    def forward(self, x):
        assert x.shape[1] == self.nclasses

        x = self.hm64_encoder(x)

        return x


class heatmap_decoder(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(heatmap_decoder, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.hm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.hm_decoder_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.nclasses, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=self.nclasses, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.nclasses, out_channels=self.nclasses, kernel_size=1, bias=False)
        ))

    def forward(self, x, decoder_id):
        heatmap_size = float(x.shape[-1])
        learn_gaumaps = self.hm_decoder_list[decoder_id](x)

        learn_heatmaps = dsntnn.flat_softmax(learn_gaumaps)
        learn_cors = dsntnn.dsnt(learn_heatmaps)
        cors = dsn_cors_2_norm_cors(learn_cors, heatmap_size)

        return cors, learn_cors, learn_gaumaps, learn_heatmaps



class c1heatmap_encoder(nn.Module):
    def __init__(self, nclasses=51):
        super(c1heatmap_encoder, self).__init__()
        self.nclasses = nclasses

        self.hm_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
            # nn.Conv2d(in_channels=self.nclasses, out_channels=self.nclasses, kernel_size=1, bias=False)
        )


    def forward(self, x):
        # assert x.shape[1] == self.nclasses

        x = self.hm_encoder(x)

        return x


class c1heatmap_decoder(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(c1heatmap_decoder, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.hm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.hm_decoder_list.append(nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
            # nn.Conv2d(in_channels=self.nclasses, out_channels=self.nclasses, kernel_size=1, bias=False)
        ))

    def forward(self, x, decoder_id):
        x = self.hm_decoder_list[decoder_id](x)

        return x




class lmark_encoder_conv(nn.Module):
    def __init__(self, nclasses=51):
        super(lmark_encoder_conv, self).__init__()

        self.nclasses = nclasses

        self.lm_encoder = nn.Sequential(
            nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        )
        self.conv_meanface = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
        )


    def forward(self, x):
        if self.nclasses == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]

        x = x.view(x.size()[0], -1, 1, 1)
        x = self.lm_encoder(x)
        x_meanface = self.conv_meanface(x)
        x_meanface = x_meanface.view(x.size()[0], -1)
        return x, x_meanface


class lmark_decoder_conv(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(lmark_decoder_conv, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.lm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
            ))

    def forward(self, x, decoder_id):

        if self.nclasses == 51:

            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
                    x = x.view(x.size()[0], -1)
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)

        x = x.view(x.size()[0], -1, 1, 1)
        x = self.lm_decoder_list[decoder_id](x)
        x = x.view(x.size()[0], -1)
        # x = x.view(x.size()[0], 68, -1)
        return x


class lmark_encoder_conv_mainroad(nn.Module):
    def __init__(self, nclasses=51, mode='1d'):
        super(lmark_encoder_conv_mainroad, self).__init__()


        self.nclasses = nclasses
        self.mode = mode
        if mode == '1d':
            self.lm_encoder = nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
            )
        elif mode == '2d':
            self.lm_encoder = nn.Sequential(
                nn.Conv2d(in_channels=self.nclasses, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=self.nclasses, kernel_size=1, stride=1)
            )
        # self.conv_meanface = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
        # )


    def forward(self, x):
        if self.nclasses == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
        if self.mode == '1d':
            x = x.view(x.size()[0], -1, 1, 1)
        elif self.mode == '2d':
            x = x.view(x.size()[0], -1, 2, 1)
        x = self.lm_encoder(x)
        # x_meanface = self.conv_meanface(x)
        # x_meanface = x_meanface.view(x.size()[0], -1)
        return x


class lmark_decoder_conv_mainroad(nn.Module):
    def __init__(self, decoder_num, nclasses=51, mode='1d'):
        super(lmark_decoder_conv_mainroad, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.mode = mode
        self.lm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            if mode == '1d':
                self.lm_decoder_list.append(nn.Sequential(
                    nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
                ))
            elif mode == '2d':
                self.lm_decoder_list.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.nclasses, out_channels=512, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(in_channels=512, out_channels=self.nclasses, kernel_size=1, stride=1)
                ))

    def forward(self, x, decoder_id):

        if self.nclasses == 51:

            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
                    x = x.view(x.size()[0], -1)
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)
        if self.mode == '1d':
            x = x.view(x.size()[0], -1, 1, 1)
        elif self.mode == '2d':
            x = x.view(x.size()[0], -1, 2, 1)

        x = self.lm_decoder_list[decoder_id](x)
        x = x.view(x.size()[0], -1)
        # x = x.view(x.size()[0], 68, -1)
        return x


class lmark_decoder_conv_mainroad_norm(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(lmark_decoder_conv_mainroad_norm, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.lm_decoder_list = nn.ModuleList()
        self.lm_decoder_list_2 = nn.ModuleList()

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
            ))

        for i in range(decoder_num):
            self.lm_decoder_list_2.append(nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
            ))


    def forward(self, x, decoder_id, mean_lm=None):

        if self.nclasses == 51:

            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
                    x = x.view(x.size()[0], -1)
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)

        x = x.view(x.size()[0], -1, 1, 1)
        mean_lm = mean_lm.view(x.size())
        x_anti_norm = self.lm_decoder_list[decoder_id](x)
        x = x_anti_norm + mean_lm
        x = self.lm_decoder_list_2[decoder_id](x)
        x = x.view(x.size()[0], -1)
        # x = x.view(x.size()[0], 68, -1)
        return x_anti_norm, x


class lmark_decoder_conv_mainroad_norm_nomeanface(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(lmark_decoder_conv_mainroad_norm_nomeanface, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.lm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
            ))

    def forward(self, x, decoder_id):

        if self.nclasses == 51:

            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
                    x = x.view(x.size()[0], -1)
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)

        x = x.view(x.size()[0], -1, 1, 1)
        x = self.lm_decoder_list[decoder_id](x)
        x = x.view(x.size()[0], -1)
        # x = x.view(x.size()[0], 68, -1)
        return x


class lmark_encoder_conv_mainroad_0228(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(lmark_encoder_conv_mainroad_0228, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num

        self.lm_encoder = nn.Sequential(
                nn.Conv2d(in_channels=(2 * self.nclasses), out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=(2 * self.nclasses), kernel_size=1, stride=1)
            )

    def forward(self, x):

        x = x.view(x.size()[0], -1, 1, 1)
        x = self.lm_encoder(x)
        x = x.view(x.size()[0], -1)
        # x = x.view(x.size()[0], 68, -1)
        return x


class lmark_decoder_conv_mainroad_norm222_nomeanface(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(lmark_decoder_conv_mainroad_norm222_nomeanface, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.lm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                nn.Conv2d(in_channels=self.nclasses, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_channels=512, out_channels=self.nclasses, kernel_size=1, stride=1)
            ))


    def forward(self, x, decoder_id):

        if self.nclasses == 51:

            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
                    x = x.view(x.size()[0], -1)
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)

        x = x.view(x.size()[0], -1, 2, 1)
        x = self.lm_decoder_list[decoder_id](x)
        x = x.view(x.size()[0], -1)
        # x = x.view(x.size()[0], 68, -1)
        return x



class lmark_encoder(nn.Module):
    def __init__(self, nclasses=51):
        super(lmark_encoder, self).__init__()
        self.nclasses = nclasses

        self.lm_encoder = nn.Sequential(
            nn.Linear(in_features= (2*self.nclasses), out_features=512),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=512, out_features=(2*self.nclasses))
        )

    def forward(self, x):
        if self.nclasses == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]

        x = x.view(x.size()[0], -1)
        x = self.lm_encoder(x)
        # x = x.view(x.size()[0], 68, -1)
        return x


class lmark_decoder(nn.Module):
    def __init__(self, decoder_num, nclasses=51):
        super(lmark_decoder, self).__init__()
        self.nclasses = nclasses
        self.decoder_num = decoder_num
        self.lm_decoder_list = nn.ModuleList()

        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                nn.Linear(in_features=(2 * self.nclasses), out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=(2 * self.nclasses))
            ))

    def forward(self, x, decoder_id):
        if self.nclasses == 51:
            if len(x.size()) == 3:
                if x.size()[1] == 68:
                    x = x[:, 17:68, :]
                    x = x.view(x.size()[0], -1)
            elif x.size()[1] == 136:
                x = x.view(x.size()[0], 68, -1)
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)

        x = self.lm_decoder_list[decoder_id](x)
        # x = x.view(x.size()[0], 68, -1)
        return x


class lmark_decoder_102(nn.Module):
    def __init__(self, decoder_num):
        super(lmark_decoder_102, self).__init__()
        self.decoder_num = decoder_num
        self.lm_decoder_list = nn.ModuleList()


        for i in range(decoder_num):
            self.lm_decoder_list.append(nn.Sequential(
                nn.Linear(in_features=102, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=512),
                nn.LeakyReLU(0.1),
                nn.Linear(in_features=512, out_features=102)
            ))
    def forward(self, x, decoder_id):
        if len(x.size()) == 3:
            if x.size()[1] == 68:
                x = x[:, 17:68, :]
                x = x.view(x.size()[0], -1)
        elif x.size()[1] == 136:
            x = x.view(x.size()[0], 68, -1)
            x = x[:, 17:68, :]
            x = x.view(x.size()[0], -1)
        x = self.lm_decoder_list[decoder_id](x)
        # x = x.view(x.size()[0], 68, -1)
        return x


class UpScale(nn.Module):

    def __init__(self, n_in, n_out):
        super(UpScale, self).__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
        )

    def forward(self, input):
        return self.upscale(input)


class AmplifyNet(nn.Module):

    def __init__(self, block_num=1):
        super(AmplifyNet, self).__init__()
        self.block_num = block_num
        self.block = nn.ModuleList()
        for i in range(block_num):
            self.block.append(nn.Sequential(
                Concat(),
                UpScale(n_in=64, n_out=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ))
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = self.conv(x)
        for i in range(self.block_num):
            x2 = self.block[i](x2)
        x3 = self.out(x2)
        x3 = self.activate(x3)
        return x1 + x3


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x1)
        x3 = self.conv1(x2)
        return x1 + x2 + x3
