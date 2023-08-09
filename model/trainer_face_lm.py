
from torch import nn
import torch
import math
import numpy as np
import sys, copy

# import copy
import os
# import math

# import torch
# import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

from model.net import FSNet_lmarks_consistency_1767_256

# from apex import amp

def ct2np(cudatensor):
    return cudatensor.cpu().detach().numpy()

class Trainer(nn.Module):
    def __init__(self, decoder_num, cfg, face_lm_model):
        super(Trainer, self).__init__()
        self.model = FSNet_lmarks_consistency_1767_256(decoder_num, cfg)

        lr = cfg.train.lr
        # lr_lm_MLP = cfg.train.lr
        # dis_params = list(self.model.dis.parameters())
        # gen_params = list(self.model.gen.parameters())
        # lm_MLP_params = list(self.model.lm_MLP.parameters())
        # gen_params = list(self.model.gen.parameters())
        params = list(self.model.parameters())
        if cfg.optim == 'RMSprop':
            self.opt = torch.optim.RMSprop(
                [p for p in params if p.requires_grad],
                lr=lr,  weight_decay=cfg['weight_decay'])

        elif cfg.optim == 'Adam':
            self.opt = torch.optim.Adam(
                [p for p in params if p.requires_grad],
                lr=lr,
                betas=[cfg.train.beta_1, cfg.train.beta_2],
                # weight_decay=0.01
            )
        # self.gen_opt = torch.optim.RMSprop(
        #     [p for p in gen_params if p.requires_grad],
        #     lr=lr_gen, weight_decay=cfg['weight_decay'])
        #
        # self.lm_MLP_scheduler = get_scheduler(self.lm_MLP_opt, cfg)
        self.scheduler = get_scheduler(self.opt, cfg)
        self.apply(weights_init(cfg['init']))
        self.face_lm_model = face_lm_model
        # self.model.gen_test = copy.deepcopy(self.model.gen)

    def update(self, x_lm, x_img, decoder_id, cfg):
        self.opt.zero_grad()

        if cfg.is_transloss:
            l_total, lf_rec_face, ll_rec_lm_in, ll_rec_lm_transback, ll_rec_facelm, ll_face_trans_lm_transback, ll_mean_lm_in = self.model(x_lm,  decoder_id, cfg, x_img, 'train', self.opt, self.face_lm_model)
        else:
            l_total, lf_rec_face, ll_rec_lm_in, ll_rec_facelm, ll_mean_lm_in = self.model(x_lm,  decoder_id, cfg, x_img, 'train', self.opt, self.face_lm_model)
        # self.loss_gen_total = torch.mean(al)
        # self.loss_gen_recon_x = torch.mean(xr)
        # self.loss_gen_recon_c = torch.mean(cr)
        # # self.loss_gen_recon_s = torch.mean(sr)
        # self.loss_gen_adv = torch.mean(ad)
        # self.accuracy_gen_adv = torch.mean(ac)

        self.opt.step()
        # this_model = self.model.module if multigpus else self.model
        # update_average(this_model.gen_test, this_model.gen)
        if cfg.is_transloss:
            return ct2np(l_total), ct2np(lf_rec_face), ct2np(ll_rec_lm_in), ct2np(ll_rec_lm_transback), ct2np(ll_rec_facelm), ct2np(ll_face_trans_lm_transback), ct2np(ll_mean_lm_in)

        else:
            return ct2np(l_total), ct2np(lf_rec_face), ct2np(ll_rec_lm_in), ct2np(ll_rec_facelm), ct2np(ll_mean_lm_in)



    def test(self, x_lm, decoder_id, multigpus=False):
        this_model = self.model.module if multigpus else self.model
        return this_model.test(x_lm, decoder_id)

    def resume(self, checkpoint_dir, cfg, multigpus):
        this_model = self.model.module if multigpus else self.model

        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        this_model.gen.load_state_dict(state_dict['gen'])
        # this_model.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(checkpoint_dir, "lm_encoder")
        state_dict = torch.load(last_model_name)
        this_model.lm_encoder.load_state_dict(state_dict['lm_encoder'])

        last_model_name = get_model_list(checkpoint_dir, "lm_decoder")
        state_dict = torch.load(last_model_name)
        this_model.lm_decoder.load_state_dict(state_dict['lm_decoder'])

        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        # state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.opt.load_state_dict(state_dict['opt'])
        # self.gen_opt.load_state_dict(state_dict['gen'])

        # self.dis_scheduler = get_scheduler(self.dis_opt, cfg, iterations)
        self.scheduler = get_scheduler(self.opt, cfg, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, multigpus):
        this_model = self.model.module if multigpus else self.model
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        lmen_name = os.path.join(snapshot_dir, 'lm_encoder_%08d.pt' % (iterations + 1))
        lmde_name = os.path.join(snapshot_dir, 'lm_decoder_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % (iterations + 1))

        torch.save({'gen': this_model.gen.state_dict() }, gen_name)
        torch.save({'lm_encoder': this_model.lm_encoder.state_dict()}, lmen_name)
        torch.save({'lm_decoder': this_model.lm_decoder.state_dict()}, lmde_name)
        torch.save({'opt': self.opt.state_dict()}, opt_name)

    def load_ckpt(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['gen'])
        self.model.lm_encoder.load_state_dict(state_dict['lm_encoder'])
        self.model.lm_decoder.load_state_dict(state_dict['lm_decoder'])

        # self.model.gen_test.load_state_dict(state_dict['gen_test'])

    # def translate(self, co_data, cl_data):
    #     return self.model.translate(co_data, cl_data)
    #
    # def translate_k_shot(self, co_data, cl_data, k, mode):
    #     return self.model.translate_k_shot(co_data, cl_data, k, mode)

    def forward(self, *inputs):
        print('Forward function not implemented.')
        pass


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, cfg, it=0):
    times = (it+1) // cfg.train.lr_decay_step
    lr = cfg.train.lr
    for i in range(times):
        lr = np.maximum(1e-5, lr * cfg.train.lr_decay_weight)
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
