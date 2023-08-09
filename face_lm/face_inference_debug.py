from __future__ import print_function
import os
import torch
import torch.nn as nn

from face_lm.fan import FAN
# import imutils
import matplotlib.pyplot as plt


import numpy as np
from easydict import EasyDict as edict
import yaml
import cv2
from utils.transforms import generate_target
# from apex.fp16_utils import *
import sys



mean_face_65_828 = np.array([[156, 219],
[194, 192],
[241, 181],
[286, 186],
[329, 196],
[328, 218],
[288, 220],
[241, 211],
[194, 220],
[498, 196],
[541, 186],
[587, 181],
[635, 192],
[671, 220],
[631, 218],
[587, 211],
[541, 216],
[497,222],
[414, 282],
[414, 333],
[414, 384],
[414, 437],
[360, 477],
[383, 483],
[414, 487],
[444, 483],
[469, 477],
[220, 284],
[241, 271],
[270, 263],
[295, 269],
[336, 288],
[301, 300],
[276, 303],
[242, 302],
[495, 290],
[533, 269],
[558, 263],
[586, 271],
[607, 284],
[586, 300],
[553, 303],
[528, 298],
[299, 589],
[339, 560],
[388, 542],
[414, 547],
[438, 542],
[488, 561],
[528, 589],
[485, 607],
[452, 618],
[414, 621],
[377, 618],
[344, 606],
[311, 584],
[379, 572],
[414, 572],
[449, 572],
[519, 586],
[447, 586],
[414, 585],
[380, 585],
[270, 283],
[558, 283]], dtype=float)

mean_face_65_828 = np.array([[156, 219], [194, 192], [241, 181], [286, 186], [329, 196], [328, 218], [288, 220], [241, 211], [194, 220], [498, 196], [541, 186], [587, 181], [635, 192], [671, 220], [631, 218], [587, 211], [541, 216], [497,222], [414, 282], [414, 333], [414, 384], [414, 437], [360, 477], [383,  483], [414, 487], [444, 483], [469, 477], [220, 284], [241, 271], [270, 263], [295, 269], [336, 288], [301, 300], [276, 303], [242, 302], [495,  290], [533, 269], [558, 263], [586, 271], [607, 284], [586, 300], [553, 303], [528, 298], [299, 589], [339, 560], [388, 542], [414, 547], [438, 542], [488, 561], [528, 589], [485, 607], [452, 618], [414, 621], [377, 618], [344, 606], [311, 584], [379, 572], [414, 572], [449, 572], [519, 586], [447, 586], [414, 585], [380, 585], [270, 283], [558, 283]], dtype=float)

mean_face_x = mean_face_65_828[:, 0]
mean_face_y = mean_face_65_828[:, 1]

mean_face_x -= 126.0
mean_face_x /= 574.0

mean_face_y -= 126.0
mean_face_y /= 574.0

meanface_65_1 = np.zeros(mean_face_65_828.shape)

meanface_65_1[:, 0] = mean_face_x
meanface_65_1[:, 1] = mean_face_y




def plt_circle(img, cors, point_size=1, color=(0, 255, 0)):
    for i in range(len(cors)):
        cv2.circle(img, tuple(cors[i]), point_size, color, -1 )
        # cv2.putText(img, '{}'.format(i), tuple(cors[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

def bgr2rgb(image):
    if len(image.shape) == 4:
        return image[:, :, :, [2, 1, 0]]
    elif len(image.shape) == 3:
        return image[:, :, [2, 1, 0]]

def plt_show(image, cv_channel=False, scale255=True, sshow=False):

    try:
        img_show = image.copy()
    except:
        img_show = image.clone()
    if len(img_show.shape) == 4:
        img_show = img_show[0].cpu().detach().numpy()

    if scale255:
        img_show *= 255.0
    if img_show.shape[0] == 3:
        img_show = np.transpose(img_show, (1, 2, 0))
    if cv_channel:
        img_show = bgr2rgb(img_show)
    img_show = np.uint8(img_show)
    if sshow:
        plt.imshow(img_show)
        plt.axis("off")
        plt.show()
    return img_show


def pts2gau_map(pts, heatmap_size=256):
    '''
    :param pts: Tensor cuda coordinates, [batch, n_classes, 2], ranging from 0 to 1
    :return: Tensor cuda heatmaps, [b_size, n_classes, 64, 64] Gaussian map, ranging from 0 to 1
    '''
    pts_numpy = pts.clone()
    pts_numpy = pts_numpy.cpu().detach().numpy()
    pts_numpy = pts_numpy * float(heatmap_size-1)
    # b_size = pts_numpy.shape[0]
    target = np.zeros((pts_numpy.shape[1], heatmap_size, heatmap_size))

    # for i_b in range(b_size):
    for i_c in range(pts_numpy.shape[1]):
        target[i_c] = generate_target(target[i_c], pts_numpy[i_c])


    return torch.FloatTensor(target).cuda()

def plot_heatmap(hm, sample=0, index=0, sshow=False):
    if not sshow:
        return None
    try:
        hm_show = hm.copy()
    except:
        hm_show = hm.clone()

    if len(hm.shape) == 4:
        hm_show = hm_show[sample].cpu().detach().numpy()
    else:
        hm_show = hm_show.cpu().detach().numpy()


    hm_show_1 = hm_show[index]

    plt.figure("heatmap")
    plt.imshow(hm_show_1, cmap='bwr')
    plt.axis('on')
    plt.title('heatmap')
    plt.show()

    hm_show_1 = hm_show[27]

    plt.figure("heatmap")
    plt.imshow(hm_show_1, cmap='bwr')
    plt.axis('on')
    plt.title('heatmap')
    plt.show()

    hm_show_1 = hm_show[22]

    plt.figure("heatmap")
    plt.imshow(hm_show_1, cmap='bwr')
    plt.axis('on')
    plt.title('heatmap')
    plt.show()


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class FaceAlignment(nn.Module):
    def __init__(self, cfg):
        super(FaceAlignment, self).__init__()
        self.cfg = cfg

        self.device = cfg.face.device
        self.heatmap_size = cfg.face.heatmap_size

        self.face_alignment_net = FAN(cfg.face.nStack, is_train=False, nModules=1, nHgDepth=4, num_feats=256, num_classes=cfg.face.num_classes)
        self.model_path = cfg.face.modelfilename

        if not os.path.isfile(self.model_path):
            print("model:%s not exists." % self.model_path)
            return None



        # TODO fix apex support
        if cfg.is_apex:
            self.face_alignment_net = network_to_half(self.face_alignment_net)
            self.face_alignment_net.train()
            self.face_alignment_net.apply(fix_bn)





    def forward(self, image_tensor):
        self.face_alignment_net.eval()


        batch_size, C, H, W = image_tensor.size()

        if self.cfg.is_lm_debug:
            # img_show = image_tensor
            img_show = plt_show(image_tensor)


        learn_cors, learn_gaumap, learn_heatmaps = self.face_alignment_net(image_tensor) # [bsize, 68,  2]



        cors = ((learn_cors[-1]+1) * self.heatmap_size - 1) / 2.0
        if self.cfg.face.norm2one:
            cors = cors / (self.heatmap_size - 1) # 0~1
            # TODO padding align
            # cors = cors - (padding / H)

        # TODO delete
        if self.cfg.is_lm_debug:
            # blank_img_path = '/media/disk/Backup/04drive/01pusun/03data/tedata/white_256.jpg'
            # blank_img = cv2.imread(blank_img_path)
            # blank_points_img = self.plt_blank(blank_img, cors)
            # mean_cors = torch.unsqueeze(torch.FloatTensor(meanface_65_1), 0).cuda()
            # blank_img = cv2.imread(blank_img_path)
            # mean_points = self.plt_blank(blank_img, mean_cors)
            # lm_img = self.plt_lm(image_tensor, cors)
            # self.draw_gaumap(cors)
            # plot_heatmap(learn_heatmaps[0])
            # return cors, lm_img, blank_points_img, mean_points
            return cors

        return cors, learn_cors, learn_gaumap, learn_heatmaps

    def load_model(self):

        self.face_alignment_net.load_state_dict(torch.load(self.model_path))
        self.face_alignment_net.to(self.device)
        self.face_alignment_net.eval()

    def plt_lm(self, image_tensor, preds):
        img = image_tensor[0].clone()
        pred = preds[0].clone()
        img = img.permute(1, 2, 0)
        img = img.cpu().detach().numpy()
        img = img[:, :, [2, 1, 0]]
        pred = pred.cpu().detach().numpy()

        img *= 255.0
        pred *= 255.0
        img = np.uint8(img)
        img = img.copy()
        # img = cv2.resize(img, (828, 822))
        plt_circle(img, pred)
        return plt_show(img, True, False)

    def plt_blank(self, img, cors):
        pred = cors[0].clone()
        pred = pred.cpu().detach().numpy()
        pred *= 255.0
        plt_circle(img, pred, 3, (255, 0, 0))
        return plt_show(img, True, False, False)

    def draw_gaumap(self, cors):
        x_lm = cors[0].clone()

        print(x_lm[0][0])
        hm_lm = pts2gau_map(x_lm, 256)

        plot_heatmap(hm_lm)



#
# if __name__ == '__main__':
#
#     with open('/media/disk/Backup/04drive/01pusun/04codes/03yuezun/04_git_df_v3_patchgan/model/cfgs/FSNet.yml', 'rb') as f:
#         cfg = edict(yaml.load(f))
#
#     fa = FaceAlignment(cfg)
#     img_path = '/media/disk/Backup/04drive/01pusun/03data/07_01dfv3/0000_train_videos/11_14_train/1min/images_s206p25/11_1min/00001.jpg'
#     img = cv2.imread(img_path)
#     img_tensor = torch.from_numpy(img[:, :,  [2, 1, 0]]).type(torch.FloatTensor) / 255.
#     img_tensor = img_tensor.permute(0, 3, 1, 2).cuda()
#
#
#     preds = fa(img_tensor)
#     plt_circle(img_cv, preds[0])
#     # for k,d in enumerate(detected_faces):
#     #     cv2.rectangle(img_in,(d[0],d[1]),(d[2],d[3]),(255,255,255))
#     #     landmark = preds[k]
#     #     for i in range(landmark.shape[0]):
#     #         pts = landmark[i]
#     #         cv2.circle(img_in, (pts[0], pts[1]),5,(0,255,0), -1, 8)
#     #         cv2.putText(img_in,str(i),(pts[0],pts[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,2555,255))
#     plt_show(img_cv, True, True)
#     io.imsave('res.jpg',img_in)
