
#from torch_fidelity import calculate_metrics
import torch 
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def interp_noise_list(noise1, noise2, alpha):
    """
    Given noise1, noise2: lists of (n1, n2) tuples,
    return a new interpolated list
    """
    mixed = []
    for (n1_a, n2_a), (n1_b, n2_b) in zip(noise1, noise2):
        # block‐0’s n1 is None; keep it None in mixed too
        if n1_a is None or n1_b is None:
            n1_m = None
        else:
            n1_m = (1 - alpha) * n1_a + alpha * n1_b

        # always have real n2
        n2_m = (1 - alpha) * n2_a + alpha * n2_b
        mixed.append((n1_m, n2_m))
    return mixed
@torch.no_grad()
def interpolate_latent_space_ddpm(model, device, res, num_steps=7):
    batch_size = 1
    shape = (batch_size, 3, res, res)

    z1 = torch.randn(shape, device=device)
    z2 = torch.randn(shape, device=device)

    images = []
    for lambd in torch.linspace(0, 1, num_steps, device=device):
        # w1, w2 are both shape: (num_blocks, batch_size, w_dims)
        z_lambd = (1 - lambd) * z1 + lambd * z2
        img = model.sample(z=z_lambd)   # exactly the same call as gen_images()
        images.append(img[0].cpu())

    # 4) Visualize
    grid = make_grid(images, nrow=num_steps, normalize=True, scale_each=True)
    plt.figure(figsize=(num_steps * 2, 2))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()

@torch.no_grad()
def interpolate_w_and_generate(
    model,
    mapping_network,
    synthesis_network,
    iteration,
    num_blocks: int,
    w_dims: int,
    device: torch.device,
    num_steps: int = 7,
    epsilon_std: float = 5,
    max_l2_dist: float = 2.0
):
    batch_size=1
    style_mixing_prob=0.9
    w1 = model.get_w()
    w2 = model.get_w()

    # 2) Generate one fixed noise list once
    fixed_noise = model.get_noise()
    fixed_noise_2 = model.get_noise()

    # 3) Linearly interpolate per-layer:
    images = []
    for alpha in torch.linspace(0, 1, num_steps, device=device):
        # w1, w2 are both shape: (num_blocks, batch_size, w_dims)
        w_interp = (1 - alpha) * w1 + alpha * w2
        noise_interp = interp_noise_list(fixed_noise, fixed_noise_2, alpha)
        img = synthesis_network(w_interp, noise_interp)   # exactly the same call as gen_images()
        images.append(img[0].cpu())

    # Visualize with torchvision grid and matplotlib
    grid = make_grid(images, nrow=num_steps, normalize=True, scale_each=True)
    plt.figure(figsize=(num_steps * 2, 2))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Interpolation in W space")
    plt.savefig(f"interpolation_plot_sgan2_{iteration}.png")
    plt.show()