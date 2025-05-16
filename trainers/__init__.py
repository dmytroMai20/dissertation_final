from .ddpm_trainer import DDPMTrainer
from .stylegan2_trainer import StyleGAN2Trainer
from .diffgan_trainer import DiffusionGANTrainer

def get_trainer(name, config, dataloader):
    if name == "ddpm":
        return DDPMTrainer(config, dataloader)
    elif name == "stylegan2":
        return StyleGAN2Trainer(config, dataloader)
    elif name == "diffgan":
        return DiffusionGANTrainer(config, dataloader)
    else:
        raise ValueError(f"Model {name} not implemented")