from models.ddpm.ddpm import ddpm
from torch.optim import Adam
import torch
from collections import defaultdict
from tqdm import tqdm
import time
import numpy as np
from .util import load_real_images, compute_fid, compute_kid, save_metrics, save_generated_images


class DDPMTrainer():

    def __init__(self, config, dataloader):
        self.config = config

        self.model = ddpm(config)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.config.lr)  # 1e-3
        self.data_loader = dataloader

        self.history = defaultdict(list)

    def train(self):
        evaluation_real_imgs = load_real_images(self.data_loader,
                                                self.config.device, 500)

        for epoch in range(self.config.epochs):
            start_time = time.time()

            for batch_idx, (imgs, _) in enumerate(tqdm(self.data_loader)):
                imgs = imgs.to(self.config.device)
                #t = torch.randint(0, self.config.timesteps, (self.config.batch_size,), device=self.config.device).long()
                loss = self.model.train_step(imgs, self.optimizer)

                gpu_mb_alloc = torch.cuda.memory_allocated() / (
                    1024**2
                )  # potentially track every few batches rather than every batch
                gpu_mb_reserved = torch.cuda.memory_reserved() / (1024**2)

                self.history['loss'].append(loss)
                self.history['gpu_mb_alloc'].append(gpu_mb_alloc)
                self.history['gpu_mb_reserved'].append(gpu_mb_reserved)

            time_per_epoch = time.time() - start_time
            self.history['times_per_epoch'].append(time_per_epoch)

            generated_images = self.gen_eval_imgs()
            save_generated_images(
                generated_images,
                f'ddpm_{self.config.dataset}_{str(self.config.res)}/epoch_{str(epoch)}'
            )
            metrics = self.calc_metrics(evaluation_real_imgs, generated_images)

            print(f"Finished epoch {epoch}/30, FID: {metrics['fid_score']}")

        self.history['cum_times'] = np.cumsum(
            np.array(self.history['times_per_epoch']))
        time_per_kimg = (
            (sum(self.history['times_per_epoch']) /
             len(self.history['times_per_epoch'])) /
            (len(self.data_loader) * self.config.batch_size)) * 1000

        print(f"Time per 1 KIMG: {time_per_kimg:.4f}")

        self.save_model(f"stylegan2_{self.config.dataset}_{self.config.res}")
        save_metrics(self.history,
                     f"stylegan2_{self.config.dataset}_{self.config.res}")

    @torch.no_grad()
    def gen_eval_imgs(self):
        generated_images = []
        num_batches = 500 // self.config.batch_size  # there is going to be slight imbalance (5000 real vs 5024 fake) but should not affect results and still yeild in accurate result
        for _ in range(num_batches):
            gen_images = self.model.sample()
            generated_images.append(gen_images)
        generated_images = torch.cat(generated_images, dim=0)
        return generated_images

    def save_model(self, name):
        save_path = f"data/{name}.pt"
        torch.save({'model': self.model.state_dict()}, save_path)

    @torch.no_grad()
    def calc_metrics(self, real, fake):

        fid_score = compute_fid(real, fake, self.config.device, 500)
        kid_mean, kid_std = compute_kid(real, fake, self.config.device, 500)

        self.history['fid_scores'].append(fid_score)
        self.history['kid_means'].append(kid_mean)
        self.history['kid_stds'].append(kid_std)

        return {
            'fid_score': fid_score,
            'kid_mean': kid_mean,
            'kid_std': kid_std
        }
