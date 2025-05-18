import torch
from torch import optim
import torch.nn.functional as F
from .generator import Generator
from .discriminator import Discriminator
from .mappingmlp import MappingMLP
from .ema import EMA
from .util import PathLengthPenalty, GradientPenalty, calculate_r_d
from .losses import DiscriminatorLoss, GeneratorLoss
from .diffusion import Diffusion
import math
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torchvision.utils as vutils
import torchvision.transforms as T
import time
import numpy as np

from tqdm import tqdm


class diffusiongan(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.batch_size = config.batch_size
        self.channels = config.channels
        self.res = config.res
        self.style_mixing_prob = config.mixing_prob
        self.w_dims = config.dim_w

        self.grad_pen_interval = config.grad_pen_interval
        self.grad_pen_coef = config.grad_pen_coef
        self.path_pen_interval = config.path_pen_interval
        self.path_pen_after = config.path_pen_after

        self.generator = Generator(int(math.log2(config.res)),
                                   config.dim_w).to(config.device)
        self.discriminator = Discriminator(int(math.log2(config.res))).to(
            config.device)
        self.mapping_net = MappingMLP(
            config.dim_w, config.mappingnet_layers).to(config.device)
        #self.ema = EMA(self.generator)

        self.num_blocks = int(math.log2(config.res)) - 1
        self.disc_loss = DiscriminatorLoss().to(config.device)
        self.gen_loss = GeneratorLoss().to(config.device)

        self.path_len_pen = PathLengthPenalty(0.99).to(config.device)
        self.r1_pen = GradientPenalty()

        self.t_update_interval = config.t_update_interval
        self.diffusion_p = 0  # initial value of diffusion (bounded [0,1] adjusting T_max)
        self.diffusion = Diffusion()
        self.diffusion.p = self.diffusion_p

        self.d_target = config.d_target
        self.update_kimg = config.update_kimg  # KImgs to change p from 0,1

        #self.g_optim = optim.Adam(self.generator.parameters(), lr=config.lr, betas=(0.0, 0.99))
        #self.d_optim = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(0.0, 0.99))
        #self.mlp_optim = optim.Adam(self.mapping_net.parameters(), lr=config.mapping_lr, betas=(0.0, 0.99))

    def forward(self):
        w = self.get_w()
        noise = self.get_noise()
        imgs = self.generator(w, noise)
        return imgs, w

    def train_step(self, real_images, d_optim, g_optim, mlp_optim, batch_idx,
                   epoch, num_batches):
        d_optim.zero_grad()
        self.train()
        real_images, t_real = self.diffusion(real_images)
        #real_images = real_images.to(device)
        #d_optim.zero_grad()

        fake_images, _ = self.gen_images()
        fake_images, t_fake = self.diffusion(fake_images)
        fake_output = self.discriminator(fake_images.detach(), t_fake)
        # requires.grad if reaches gradient penalty interval (set to 4)
        if (batch_idx + 1) % self.grad_pen_interval == 0:
            real_images.requires_grad_()

        real_output = self.discriminator(real_images, t_real)

        real_loss, fake_loss = self.disc_loss(real_output, fake_output)
        d_loss = real_loss + fake_loss

        if (batch_idx + 1) % self.t_update_interval == 0:
            adjust = np.sign(
                calculate_r_d(real_output).item() -
                self.d_target) * (self.batch_size * self.t_update_interval) / (
                    self.update_kimg * 1000)
            self.diffusion.p = (self.diffusion.p + adjust).clip(min=0., max=1.)
            self.diffusion.update_T()

        if (batch_idx + 1) % self.grad_pen_interval == 0:
            r1 = self.r1_pen(real_images, real_output)
            d_loss = d_loss + 0.5 * self.grad_pen_coef * r1 * self.grad_pen_interval
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        d_optim.step()

        #self.d_losses.append(d_loss.item())

        g_optim.zero_grad()
        mlp_optim.zero_grad()

        fake_images, w = self.gen_images()
        fake_images, t_fake = self.diffusion(fake_images)
        fake_output = self.discriminator(fake_images, t_fake)

        g_loss = self.gen_loss(fake_output)

        if (batch_idx + 1) % self.path_pen_interval == 0 and batch_idx + (
                epoch * num_batches) > self.path_pen_after:
            path_len_penalty = self.path_len_pen(w, fake_images)
            if not torch.isnan(path_len_penalty):
                g_loss = g_loss + path_len_penalty

        g_loss.backward()

        nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.mapping_net.parameters(), max_norm=1.0)

        g_optim.step()
        mlp_optim.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
        }
        #ema.update(generator)

    def get_w(self):
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.num_blocks)
            z2 = torch.randn(self.batch_size, self.w_dims).to(self.device)
            z1 = torch.randn(self.batch_size, self.w_dims).to(self.device)

            w1 = self.mapping_net(z1)
            w2 = self.mapping_net(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.num_blocks - cross_over_point, -1,
                                       -1)
            return torch.cat((w1, w2), dim=0)

        else:

            z = torch.randn(self.batch_size, self.w_dims).to(self.device)

            w = self.mapping_net(z)

            return w[None, :, :].expand(self.num_blocks, -1, -1)

    def get_noise(self):
        noise = []
        resolution = 4

        for i in range(self.num_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(self.batch_size,
                                 1,
                                 resolution,
                                 resolution,
                                 device=self.device)

            n2 = torch.randn(self.batch_size,
                             1,
                             resolution,
                             resolution,
                             device=self.device)
            noise.append((n1, n2))
            resolution *= 2

        return noise

    def gen_images(self):
        w = self.get_w()
        noise = self.get_noise()
        imgs = self.generator(w, noise)
        return imgs, w
