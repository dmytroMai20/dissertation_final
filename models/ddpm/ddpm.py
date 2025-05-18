import torch
import torch.nn.functional as F
from .network import Unet
from .schedules import linear_beta_schedule
from tqdm import tqdm


class ddpm(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.timesteps = config.timesteps
        self.batch_size = config.batch_size
        self.channels = config.channels
        self.res = config.res

        self.model = Unet(dim=config.res,
                          channels=config.channels,
                          dim_mults=(
                              1,
                              2,
                              4,
                          )).to(self.device)
        self.betas = linear_beta_schedule(timesteps=config.timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0),
                                         value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. -
                                                        self.alphas_cumprod)

        self.posterior_variance = self.betas * (
            1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        #print("betas initialized")

    def forward(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_start=x_0, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    def train_step(self, imgs, optimizer):
        optimizer.zero_grad()
        self.train()
        t = torch.randint(0,
                          self.timesteps, (imgs.shape[0], ),
                          device=self.device).long()
        loss = self(imgs.to(self.device), t)
        loss.backward()
        optimizer.step()
        return loss.item()

    """def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss"""

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size,
                           *((1, ) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t,
                                             x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample_loop(self, z=None):
        #device = next(self.parameters()).device

        #b = shape[0]
        #shape=(batch_size, channels, image_size, image_size)
        # start from pure noise (for each example in the batch)
        if z == None:
            img = torch.randn(
                (self.batch_size, self.channels, self.res, self.res),
                device=self.device)
        else:
            img = z
        #imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)),
                      desc='sampling loop time step',
                      total=self.timesteps):
            img = self.p_sample(
                img,
                torch.full((self.batch_size, ),
                           i,
                           device=self.device,
                           dtype=torch.long), i)
            #imgs.append(img.cpu().numpy())
        return img.detach()

    @torch.no_grad()
    def p_sample(self, x, t, t_index):

        betas_t = self.extract(self.betas, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t,
                                                x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, z=None):
        return self.p_sample_loop(z=z)
