
from torch import nn
import torch
import math
import numpy as np
import sys, copy
import os, sys, pdb
import torch.nn.init as init
from torch.optim import lr_scheduler

from model.net import Lm_autoencoder, Lm_autoencoder_mainroad, Lm_autoencoder_mean_id_mainroad, Hm_autoencoder_mean_id_mainroad, Lm_autoencoder_mainroad_norm, Lm_autoencoder_mainroad_norm_nomeanface, Lm_autoencoder_mainroad_norm222_nomeanface, Lm_autoencoder_mainroad_0228_1, Lm_autoencoder_mainroad_0228_2


def ct2np(cudatensor):
    return cudatensor.cpu().detach().numpy()


def modelsize(params):
    return sum(p.numel() for p in params)


class Trainer(nn.Module):
    def __init__(self, decoder_num, cfg, face_lm_model):
        super(Trainer, self).__init__()
        self.cfg = cfg

        self.model = Lm_autoencoder_mainroad(decoder_num, cfg)




        if 'is_meanface_mainroad' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_meanface_mainroad:
                self.model = Lm_autoencoder_mainroad(decoder_num, cfg)

        if cfg.lm_AE.is_mean_lm_id_loss:
            self.model = Lm_autoencoder_mean_id_mainroad(decoder_num, cfg)
        if 'is_hm_AE_v2' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE_v2:
                self.model = Hm_autoencoder_mean_id_mainroad(decoder_num, cfg)

        if 'is_lm_norm' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_lm_norm == 1:
                self.model = Lm_autoencoder_mainroad_norm(decoder_num, cfg)
            elif cfg.lm_AE.is_lm_norm == 2:
                self.model = Lm_autoencoder_mainroad_norm_nomeanface(decoder_num, cfg)
            elif cfg.lm_AE.is_lm_norm == 3:
                self.model = Lm_autoencoder_mainroad_norm222_nomeanface(decoder_num, cfg)
            elif cfg.lm_AE.is_lm_norm == 4:
                self.model = Lm_autoencoder_mainroad_0228_1(decoder_num, cfg)
            elif cfg.lm_AE.is_lm_norm == 5:
                # self.model = (decoder_num, cfg)
                self.model = Lm_autoencoder_mainroad_0228_2(decoder_num, cfg)



        # if 'is_multi_scale_loss' in list(cfg.gen.keys()):
        #     if cfg.gen.is_multi_scale_loss:
        #         self.model = FSNet_lmarks_multiscale_patchgan_lmconsis_256(decoder_num, cfg)
        #
        # if 'is_mean_lm_id_loss' in list(cfg.lm_AE.keys()):
        #     if cfg.lm_AE.is_mean_lm_id_loss:
        #         self.model = FSNet_lmarks_mean_lm_id_patchgan_lmconsis_256(decoder_num, cfg)


        lr_lm_decoder = cfg.train.lr
        if cfg.lm_AE.is_pretrained_lm_encoder:
            lr_lm_encoder = 1e-5
        else:
            lr_lm_encoder = cfg.train.lr

        # gen_params = list(self.model.gen.parameters())
        # lm_encoder_params = list(self.model.lm_encoder.parameters())
        # lm_decoder_params = list(self.model.lm_decoder.parameters())

        if 'is_lm_norm' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_lm_norm == 2:
                self.lm_AE_opt = torch.optim.RMSprop(
                    [{'params': self.model.lm_decoder.parameters(), 'lr': lr_lm_decoder}
                     ], weight_decay=cfg['weight_decay'])
            elif cfg.lm_AE.is_lm_norm >= 4:
                self.lm_AE_opt = torch.optim.RMSprop(
                    [{'params': self.model.lm_decoder.parameters(), 'lr': lr_lm_decoder},
                     {'params': self.model.lm_encoder.parameters(), 'lr': lr_lm_encoder},
                     ], weight_decay=cfg['weight_decay'])

        else:

            self.lm_AE_opt = torch.optim.RMSprop(
                [{'params': self.model.lm_decoder.parameters(), 'lr': lr_lm_decoder},
                 {'params': self.model.lm_encoder.parameters(), 'lr': lr_lm_encoder},
                 ], weight_decay=cfg['weight_decay'])


        if 'is_hm_AE' in list(cfg.lm_AE.keys()):
            if cfg.lm_AE.is_hm_AE:
                self.lm_AE_opt = torch.optim.RMSprop(
                    [{'params': self.model.lm_decoder.parameters(), 'lr': lr_lm_decoder},
                     {'params': self.model.lm_encoder.parameters(), 'lr': lr_lm_encoder},
                     {'params': self.model.gaumap_encoder.parameters(), 'lr': lr_lm_encoder}
                     ], weight_decay=cfg['weight_decay'])



        #
        # assert (modelsize(self.model.gen.parameters()) == modelsize(self.gen_lm_AE_opt.param_groups[0]['params'])) and (modelsize(self.model.lm_decoder.parameters()) == modelsize(self.gen_lm_AE_opt.param_groups[1]['params'])) and  (modelsize(self.model.lm_encoder.parameters()) == modelsize(self.gen_lm_AE_opt.param_groups[2]['params']))

        self.lm_AE_scheduler = get_scheduler(self.lm_AE_opt, cfg)

        # if self.cfg.lm_AE.is_pretrained_lm_encoder:
        #     self.gen_lm_AE_opt.param_groups[2]['lr'] = 1e-5
        self.apply(weights_init(cfg['init']))
        # self.face_lm_model = face_lm_model

    def lm_AE_update(self, x_lm, x_img, decoder_id, cfg):
        self.lm_AE_opt.zero_grad()
        # al, ad, cr, ac, lf_rec_face, ll_rec_lm_in, ll_rec_facelm, ll_mean_lm_in = self.model(x_lm,  decoder_id, cfg, x_img, 'gen_update', None, self.face_lm_model)

        meta_lm_AE = self.model(x_lm, decoder_id, cfg, x_img=None, loss_mode='train')

        # self.loss_gen_total = torch.mean(al)
        # self.loss_gen_recon_c = torch.mean(cr)
        # self.loss_gen_recon_s = torch.mean(sr)
        # self.loss_gen_adv = torch.mean(ad)
        # self.accuracy_gen_adv = torch.mean(ac)

        self.lm_AE_opt.step()
        # this_model = self.model.module if multigpus else self.model
        # update_average(this_model.gen_test, this_model.gen)
        # return ct2np(lf_rec_face), self.accuracy_gen_adv.item(), ct2np(ll_rec_lm_in), ct2np(ll_rec_facelm), ct2np(ll_mean_lm_in)
        return meta_lm_AE
    def dis_update(self, x_lm, x_img, decoder_id, cfg):
        self.dis_opt.zero_grad()
        meta_dis = self.model(x_lm, decoder_id, cfg, x_img, 'dis_update', self.dis_opt)

        # al, lfa, lre, acc, acc_f, acc_r = self.model(x_lm, decoder_id, cfg, x_img, 'dis_update', self.dis_opt)
        # self.loss_dis_total = torch.mean(al)
        # self.loss_dis_fake_adv = torch.mean(lfa)
        # self.loss_dis_real_adv = torch.mean(lre)
        # self.loss_dis_reg = torch.mean(reg)
        # self.accuracy_dis_adv = torch.mean(acc)
        self.dis_opt.step()
        return meta_dis

    def test(self, x_lm, decoder_id, multigpus=False):
        this_model = self.model.module if multigpus else self.model
        return this_model.test(x_lm, decoder_id)

    def resume(self, checkpoint_dir, cfg, multigpus=False, index=-1, decoder_id=1000):
        this_model = self.model.module if multigpus else self.model
        # pdb.set_trace()
        try:
            last_model_name = get_model_list(checkpoint_dir, "lm_encoder_", index)
            state_dict = torch.load(last_model_name)
            this_model.lm_encoder.load_state_dict(state_dict['lm_encoder'])
            # this_model.gen_test.load_state_dict(state_dict['gen_test'])
        except:
            pass

        last_model_name = get_model_list(checkpoint_dir, "lm_decoder_", index)
        state_dict = torch.load(last_model_name)
        this_model.lm_decoder.load_state_dict(state_dict['lm_decoder'])
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(checkpoint_dir, "lm_AE_opt_", index)
        state_dict = torch.load(last_model_name)
        # state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        # self.dis_opt.load_state_dict(state_dict['dis'])
        self.lm_AE_opt.load_state_dict(state_dict['lm_AE'])



        # self.dis_scheduler = get_scheduler(self.dis_opt, cfg)
        self.lm_AE_scheduler = get_scheduler(self.lm_AE_opt, cfg)

        # if self.cfg.lm_AE.is_pretrained_lm_encoder:
        #     self.gen_lm_AE_opt.param_groups[2]['lr'] = 1e-5

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, multigpus, decoder_id=1000):
        this_model = self.model.module if multigpus else self.model
        # Save generators, discriminators, and optimizers
        lm_encoder_name = os.path.join(snapshot_dir, 'lm_encoder_{:0>8d}.pt'.format((iterations + 1)))
        lm_decoder_name = os.path.join(snapshot_dir, 'lm_decoder_{:0>8d}.pt'.format((iterations + 1)))

        # dis_name = os.path.join(snapshot_dir, 'dis_{}_{:0>8d}.pt'.format(decoder_id, (iterations + 1)))
        # lm_encoder_name = os.path.join(snapshot_dir, 'lm_encoder_%08d.pt' % (iterations + 1))
        # lm_decoder_name = os.path.join(snapshot_dir, 'lm_decoder_%08d.pt' % (iterations + 1))
        # gaumap_encoder_name = os.path.join(snapshot_dir, 'gaumap_coder_%08d.pt' % (iterations + 1))

        opt_name = os.path.join(snapshot_dir, 'lm_AE_opt_{:0>8d}.pt'.format((iterations + 1)))
        try:
            torch.save({'lm_encoder': this_model.lm_encoder.state_dict()}, lm_encoder_name)
        except:
            pass
        torch.save({'lm_decoder': this_model.lm_decoder.state_dict()}, lm_decoder_name)
        # torch.save({'lm_encoder': this_model.lm_encoder.state_dict()}, lm_encoder_name)
        # torch.save({'lm_decoder': this_model.lm_decoder.state_dict()}, lm_decoder_name)
        torch.save({'lm_AE': self.lm_AE_opt.state_dict()}, opt_name)

        # if 'is_hm_AE' in list(self.cfg.lm_AE.keys()):
        #     if self.cfg.lm_AE.is_hm_AE:
        #         torch.save({'gaumap_encoder': this_model.gaumap_encoder.state_dict()}, gaumap_encoder_name)

    def load_ckpt(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['gen'])

    def forward(self, *inputs):
        print('Forward function not implemented.')
        pass


def get_model_list(dirname, key, index=-1):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[index]
    print('load model {} from {}:'.format(key, last_model_name))
    return last_model_name


def get_scheduler(optimizer, cfg, it=0, fixed=False):
    if fixed:
        lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    times = (it) // cfg.train.lr_decay_step
    lr = cfg.train.lr
    for i in range(times):
        lr = lr * cfg.train.lr_decay_weight
    lr = np.maximum(1e-5, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun
