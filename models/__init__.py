from .ddpm.ddpm import ddpm
from .stylegan2.stylegan2 import stylegan2
from .diff_gan.diffgan import diffusiongan


def get_model(name, config):
    if name == "ddpm":
        return ddpm(config)
    elif name == "stylegan2":
        return stylegan2(config)
    elif name == "diffusion_gan":
        return diffusiongan(config)
    else:
        raise ValueError(f"Unknown model: {name}")
