from collections import defaultdict
from models.diff_gan.diffgan import diffusiongan
from torch.optim import Adam
from models.diff_gan.ema import EMA
import torch
from tqdm import tqdm
import time
import numpy as np
from .util import load_real_images, compute_fid, compute_kid, save_metrics, save_generated_images

class DiffusionGANTrainer():
    def __init__(self, config, dataloader):
        self.config = config

        self.model = diffusiongan(config)

        self.data_loader = dataloader

        self.g_optim = Adam(self.model.generator.parameters(), lr=config.lr, betas=(0.0, 0.99))
        self.d_optim = Adam(self.model.discriminator.parameters(), lr=config.lr, betas=(0.0, 0.99))
        self.mlp_optim = Adam(self.model.mapping_net.parameters(), lr=config.mapping_lr, betas=(0.0, 0.99))

        self.ema = EMA(self.model.generator)

        self.history = defaultdict(list)
        
    def train(self):
        evaluation_real_imgs = load_real_images(self.dataloader, self.config.device, 500)
        num_batches = len(self.data_loader)

        for epoch in range(self.config.epochs):
            start_time = time.time()

            for batch_idx, (imgs, _) in enumerate(tqdm(self.data_loader)):
                
                imgs = imgs.to(self.config.device)
                
                loss = self.model.train_step(imgs, self.d_optim, self.g_optim, self.mlp_optim, batch_idx, epoch, num_batches)

                gpu_mb_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                gpu_mb_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

                self.ema.update(self.model.generator)

                self.history['loss_d'].append(loss['d_loss'])
                self.history['loss_g'].append(loss['g_loss'])

                self.history['gpu_mb_alloc'].append(gpu_mb_alloc)
                self.history['gpu_mb_reserved'].append(gpu_mb_reserved)

            time_per_epoch = time.time()-start_time
            ###
            # Eval
            ###
            generated_images = self.gen_eval_imgs()
            save_generated_images(generated_images,f'ddpm_{self.config.dataset}_{str(self.config.res)}/epoch_{str(epoch)}')
            metrics = self.calc_metrics(evaluation_real_imgs,generated_images)
            
            print(f"Finished epoch {epoch}/30, FID: {metrics['fid_score']}")
            self.history['times_per_epoch'].append(time_per_epoch)
        
        self.history['cum_times'] = np.cumsum(np.array(self.history['times_per_epoch']))
        time_per_kimg = ((sum(self.history['times_per_epoch'])/len(self.history['times_per_epoch']))/
                         (len(self.data_loader)*self.config.batch_size))*1000
        
        print(f"Time per 1 KIMG: {time_per_kimg:.4f}")

        self.save_model(f"stylegan2_{self.config.dataset}_{self.config.res}")
        save_metrics(self.history, f"stylegan2_{self.config.dataset}_{self.config.res}")

    
    def save_model(self, name):
        save_path = f"data/{name}.pt"
        torch.save({'generator':self.model.generator.state_dict(),
                    'mapping_net':self.model.mapping_net.state_dict(),
                    'ema':self.ema.ema_model.state_dict()}, save_path)
        
    @torch.no_grad()    
    def calc_metrics(self, real, fake):

        fid_score = compute_fid(real,fake, self.config.device, 500)
        kid_mean, kid_std = compute_kid(real, fake, self.config.device, 500)

        self.history['fid_scores'].append(fid_score)
        self.history['kid_means'].append(kid_mean)
        self.history['kid_stds'].append(kid_std)

        return {
            'fid_score': fid_score,
            'kid_mean': kid_mean,
            'kid_std': kid_std
        }

    @torch.no_grad()
    def gen_eval_imgs(self):
        generated_images = []
        num_batches = 500 // self.config.batch_size # there is going to be slight imbalance (5000 real vs 5024 fake) but should not affect results and still yeild in accurate result
        for _ in range(num_batches):
            gen_images = self.model.gen_images()
            generated_images.append(gen_images)
        generated_images = torch.cat(generated_images, dim=0)
        return generated_images
