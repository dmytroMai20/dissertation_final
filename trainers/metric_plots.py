import matplotlib.pyplot as plt
import numpy as np


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

