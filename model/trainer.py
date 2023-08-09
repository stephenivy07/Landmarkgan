
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

from model.net import FSNet_lmarks_1767_patchgan_256
# from apex import amp

class Trainer(nn.Module):
    def __init__(self, decoder_num, cfg):
        super(Trainer, self).__init__()
        self.model = FSNet_lmarks_1767_patchgan_256(decoder_num, cfg)
        lr_gen = cfg.train.lr
        lr_dis = cfg.train.lr
        # dis_params = list(self.model.dis.parameters())
        # gen_params = list(self.model.gen.parameters())
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.RMSprop(
            [p for p in dis_params if p.requires_grad],
            lr=lr_gen, weight_decay=cfg['weight_decay'])
        self.gen_opt = torch.optim.RMSprop(
            [p for p in gen_params if p.requires_grad],
            lr=lr_dis, weight_decay=cfg['weight_decay'])

        # self.gen_opt = torch.optim.RMSprop(
        #     [p for p in gen_params if p.requires_grad],
        #     lr=lr_dis, weight_decay=cfg['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, cfg)
        self.gen_scheduler = get_scheduler(self.gen_opt, cfg)
        self.apply(weights_init(cfg['init']))
        # self.model.gen_test = copy.deepcopy(self.model.gen)

    def gen_update(self, x_lm, x_img, decoder_id, cfg):
        self.gen_opt.zero_grad()
        al, ad, xr, cr, ac = self.model(x_lm,  decoder_id, cfg, x_img, 'gen_update', self.gen_opt)
        self.loss_gen_total = torch.mean(al)
        self.loss_gen_recon_x = torch.mean(xr)
        self.loss_gen_recon_c = torch.mean(cr)
        # self.loss_gen_recon_s = torch.mean(sr)
        self.loss_gen_adv = torch.mean(ad)
        self.accuracy_gen_adv = torch.mean(ac)
        self.gen_opt.step()
        # this_model = self.model.module if multigpus else self.model
        # update_average(this_model.gen_test, this_model.gen)
        return self.loss_gen_recon_x, self.accuracy_gen_adv.item()

    def dis_update(self, x_lm, x_img, decoder_id, cfg):
        self.dis_opt.zero_grad()
        al, lfa, lre, acc = self.model(x_lm, decoder_id, cfg, x_img, 'dis_update', self.dis_opt)
        self.loss_dis_total = torch.mean(al)
        self.loss_dis_fake_adv = torch.mean(lfa)
        self.loss_dis_real_adv = torch.mean(lre)
        # self.loss_dis_reg = torch.mean(reg)
        self.accuracy_dis_adv = torch.mean(acc)
        self.dis_opt.step()
        return self.accuracy_dis_adv.item()

    def test(self, x_lm, decoder_id, multigpus):
        this_model = self.model.module if multigpus else self.model
        return this_model.test(x_lm, decoder_id)

    def resume(self, checkpoint_dir, cfg, multigpus):
        this_model = self.model.module if multigpus else self.model

        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        this_model.gen.load_state_dict(state_dict['gen'])
        # this_model.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        this_model.dis.load_state_dict(state_dict['dis'])

        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        # state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        self.dis_scheduler = get_scheduler(self.dis_opt, cfg, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, cfg, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, multigpus):
        this_model = self.model.module if multigpus else self.model
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % (iterations + 1))
        torch.save({'gen': this_model.gen.state_dict() }, gen_name)
        torch.save({'dis': this_model.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)

    def load_ckpt(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['gen'])
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
