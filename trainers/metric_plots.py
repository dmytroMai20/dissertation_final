import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def plot_d_g_loss(d_losses, g_losses,name):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.title("Training Loss Curves")
    plt.savefig(f"log_loss_plot_{name}.png")
    plt.show()

def plot_loss(losses, name):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.title("Training Loss (log) Curve")
    plt.savefig(f"loss_plot_{name}.png")
    plt.show()

def plot_mem_usage(gpu_mb_alloc, gpu_mb_reserved, name):
    plt.figure(figsize=(10, 5))
    plt.plot(gpu_mb_alloc, label="CUDA Memory Allocated (MB)")
    plt.plot(gpu_mb_reserved, label="CUDA Memory Reserved (MB)")
    plt.xlabel("Iteration")
    plt.ylabel("MB")
    plt.legend()
    plt.title("Training Memory Usage")
    plt.savefig(f"memory_plot_{name}.png")
    plt.show()

def create_image_grid(folder_path, output_path, grid_size=(8, 8), image_size=(64, 64)):
    # List all PNG images in the folder
    folder_path = os.path.join(os.getcwd(), folder_path)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort()  # Sort to keep order consistent
    image_files = image_files[:grid_size[0] * grid_size[1]]  # Take only first 64 images

    # Create a blank canvas
    grid_width = grid_size[0] * image_size[0]
    grid_height = grid_size[1] * image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, file_name in enumerate(image_files):
        img = Image.open(os.path.join(folder_path, file_name)).resize(image_size)
        row = idx // grid_size[0]
        col = idx % grid_size[0]
        grid_image.paste(img, (col * image_size[0], row * image_size[1]))

    grid_image.save(output_path)
    print(f"Saved grid image to {output_path}")


if __name__ == "__main__":
    create_image_grid('epoch_29', 'output_grid_diff_gan_29.png')

def plot_fid_kid(times_per_epoch, fid_scores, kid_means, kid_stds, name):
    # plot FID and KID against time 
    cum_times = np.cumsum(np.array(times_per_epoch))
    fig, ax1 = plt.subplots()   # may need to fix figure size to (10,5) too

    # FID line (left y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('FID', color=color)
    ax1.plot(cum_times, fid_scores, color=color, label='FID')
    ax1.tick_params(axis='y', labelcolor=color)

    # KID line with error bars (right y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('KID', color=color)
    ax2.errorbar(cum_times, kid_means, yerr=kid_stds, color=color, linestyle='--', marker='o', label='KID ± std')
    ax2.tick_params(axis='y', labelcolor=color)

    # Layout and title
    fig.tight_layout()
    plt.title("FID and KID over Training Time")
    plt.savefig(f"fid_vs_kid_{name}.png")
    plt.show()

