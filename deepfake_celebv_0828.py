
import numpy as np
import os
from model.net import *
import torch
import sys
from easydict import EasyDict as edict
import yaml, pdb

from model.trainer import Trainer as gan_trainer
from model.trainer_face_lm import Trainer as lmconsis_trainer
from model.trainer_patchgan_lmconsis import Trainer as patchgan_lmconsis_trainer
from model.trainer_patchgan_lmconsis_one_decoder import Trainer as trainer_one_decoder
from model.trainer_patchgan_lmconsis_lmautoencoder import Trainer as trainer_lm_AE


class Lmark_DeepFake(object):
    def __init__(self, size=256, ctl_cfg=None, net_cfg=None, is_cuda=True, model_index=-1):
        self.size = size

        self.is_cuda = is_cuda
        self.model_index = model_index
        self.iterations = 0
        print('==================>')

        with open(net_cfg['decoder_cfg_path'], 'rb') as f:
            self.decoder_cfg = edict(yaml.safe_load(f))
        with open(net_cfg['lm_AE_cfg_path'], 'rb') as f:
            self.lm_AE_cfg = edict(yaml.safe_load(f))

        self.net_cfg = net_cfg

        self.decoder_ids = net_cfg['decoder_ids']
        self.decoder_num = len(self.decoder_ids)
        self.lm_AE_ids = net_cfg['lm_AE_ids']
        self.lm_AE_num = len(self.lm_AE_ids)



        if net_cfg['model_type'] == '1216_gan':
            self.lm_AE = trainer_lm_AE_gan(self.lm_AE_num, self.lm_AE_cfg, None)
        else:
            self.lm_AE = trainer_lm_AE(self.lm_AE_num, self.lm_AE_cfg, None)
        self.lm_AE.resume(self.net_cfg['lm_AE_model_name'], self.lm_AE_cfg, multigpus=False, index=net_cfg['lm_AE_ckpt_index'], decoder_id=net_cfg['lm_AE_ckpt_index'])
        self.lm_AE.eval().cuda()

        self.decoder_dict = {}

        for i in range(self.decoder_num):
            decoder_id = self.decoder_ids[i]

            trainer = trainer_one_decoder(1, self.decoder_cfg, None)
            iterations = trainer.resume(self.net_cfg['decoder_model_name'], self.decoder_cfg, multigpus=False, index=net_cfg['decoder_ckpt_index'], decoder_id=decoder_id)
            print('model for {} loaded'.format(decoder_id))
            trainer.eval().cuda()
            self.decoder_dict['{}'.format(decoder_id)] = trainer


    def GAN_convert(self, face_lmarks, id_specific=-1):
        face_lmarks = np.expand_dims(face_lmarks, 0)
        lm_tensor = torch.from_numpy(face_lmarks).float().cuda()

        face_dict = {}
        if id_specific == -1:
            for i in range(self.decoder_num):
                decoder_id = self.decoder_ids[i]
                lm_decoder_id = self.net_cfg['map_de_id2AE_id'][str(decoder_id)]
                if self.net_cfg['skip_lm_AE'] == True:
                    lm = lm_tensor
                    face = self.decoder_dict['{}'.format(decoder_id)].test(lm, 0)

                    new_face = np.uint8(face.data.cpu().numpy()[0] * 255)
                    new_face = np.transpose(new_face, (1, 2, 0))
                    face_dict['{}'.format(decoder_id)] = new_face
                elif self.net_cfg['skip_lm_AE'] == False or self.net_cfg['skip_lm_AE'] == 'full':
                    lm = self.lm_AE.test(lm_tensor, lm_decoder_id)
                    face = self.decoder_dict['{}'.format(decoder_id)].test(lm, 0)

                    new_face = np.uint8(face.data.cpu().numpy()[0] * 255)
                    new_face = np.transpose(new_face, (1, 2, 0))
                    face_dict['{}'.format(decoder_id)] = new_face
                elif self.net_cfg['skip_lm_AE'] == 'skip':
                    lm = lm_tensor
                    face = self.decoder_dict['{}'.format(decoder_id)].test(lm, 0)

                    new_face = np.uint8(face.data.cpu().numpy()[0] * 255)
                    new_face = np.transpose(new_face, (1, 2, 0))
                    face_dict['{}'.format(decoder_id)] = new_face
                elif self.net_cfg['skip_lm_AE'] == 'only_AE':
                    lm = self.lm_AE.test(lm_tensor, lm_decoder_id).detach().cpu().numpy()[0] #[98, 2]
                    face_dict['{}'.format(decoder_id)] = lm
            return face_dict
        else:
            decoder_id = self.decoder_ids[id_specific]
            lm_decoder_id = self.net_cfg['map_de_id2AE_id'][str(decoder_id)]
            if self.net_cfg['skip_lm_AE'] == True:
                lm = lm_tensor
                face = self.decoder_dict['{}'.format(decoder_id)].test(lm, 0)

                new_face = np.uint8(face.data.cpu().numpy()[0] * 255)
                new_face = np.transpose(new_face, (1, 2, 0))
                face_dict['{}'.format(decoder_id)] = new_face
            elif self.net_cfg['skip_lm_AE'] == False or self.net_cfg['skip_lm_AE'] == 'full':
                lm = self.lm_AE.test(lm_tensor, lm_decoder_id)
                face = self.decoder_dict['{}'.format(decoder_id)].test(lm, 0)

                new_face = np.uint8(face.data.cpu().numpy()[0] * 255)
                new_face = np.transpose(new_face, (1, 2, 0))
                face_dict['{}'.format(decoder_id)] = new_face
            elif self.net_cfg['skip_lm_AE'] == 'skip':
                lm = lm_tensor
                face = self.decoder_dict['{}'.format(decoder_id)].test(lm, 0)

                new_face = np.uint8(face.data.cpu().numpy()[0] * 255)
                new_face = np.transpose(new_face, (1, 2, 0))
                face_dict['{}'.format(decoder_id)] = new_face
            elif self.net_cfg['skip_lm_AE'] == 'only_AE':
                lm = self.lm_AE.test(lm_tensor, lm_decoder_id).detach().cpu().numpy()[0]  # [98, 2]
                face_dict['{}'.format(decoder_id)] = (lm)
            return face_dict
