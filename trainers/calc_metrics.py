import torch
import torchvision.transforms as T
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms
from torchvision.io import read_image
import math
import os
from torch_fidelity import calculate_metrics
from dataset import CustomDataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""def load_images_from_folder(folder_path, device):
    images = []
    transform = T.Compose([
        transforms.Resize((299, 299)),  # Inception expects 299x299
        transforms.ConvertImageDtype(torch.float32),  # Convert to float [0,1]
    ])
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png')):
            img_path = os.path.join(folder_path, filename)
            img = read_image(img_path)  # shape: (C, H, W), dtype uint8 [0,255]
            img = transform(img)  # shape: (C, 299, 299), dtype float32 [0,1]
            images.append(img)
    
    if images:
        imgs_tensor = torch.stack(images).to(device)
        return imgs_tensor
    else:
        return None
"""
@torch.no_grad()
def save_real_images(real_dataloader, to_save = 5000):
    os.makedirs("real", exist_ok=True)
    imgs_saved = 0
    for batch in tqdm(real_dataloader, desc= "Saving real images"):
        if imgs_saved >= to_save: 
            break
        imgs, _ = batch
        for img in imgs:
            if imgs_saved >= to_save:
                break
            path = os.path.join("real", f"img_{imgs_saved}.png")
            save_image(img, path, normalize=True)
            imgs_saved+=1
    print("Real images saved.")
    
@torch.no_grad()
def compute_recall_precision(real_imgs, fake_imgs, device):
    prc_dict = calculate_metrics(
        input1=f'./{real_imgs}', 
        input2=f'./{fake_imgs}', 
        cuda=True, 
        isc=False, 
        fid=False, 
        kid=False, 
        prc=True, 
        verbose=False
    )
    inception_dict = calculate_metrics(
        input1=f'./{fake_imgs}', 
        cuda=True, 
        isc=True, 
        fid=False, 
        kid=False, 
        prc=False, 
        verbose=False
    )
    prc_dict['inception_score_mean'] = inception_dict['inception_score_mean']
    return prc_dict
@torch.no_grad()
def compute_metrics(dataset_name,fake_path, res):
    precisions = []
    recalls = []
    inceptions = []
    dataset = CustomDataLoader(32,res,dataset_name)
    real_loader = dataset.get_loader()
    save_real_images(real_loader)
    real_imgs = "real"
    for i in tqdm(range(0,30), desc="Calculating metrics per epoch"):
        fake_imgs = os.path.join(fake_path,f"epoch_{i}")
        metrics = compute_recall_precision(real_imgs, fake_imgs, device)
        print(metrics)
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        inceptions.append(metrics['inception_score_mean'])
    return precisions, recalls, inceptions
"""
def main():
    root_dir = './'  # Change this

    results = {}

    for i in range(0, 30):
        folder = os.path.join(root_dir, f"epoch_{str(i)}")
        if os.path.isdir(folder):
            imgs = load_images_from_folder(folder, device)
            if imgs is not None:
                score, std = compute_inception_score(imgs, device)
                results[i] = score
                print(f"Folder {i}: Inception Score = {score:.4f}, std = {std:.4f}")
            else:
                print(f"Folder {i}: No images found.")
    
    # Optionally save results
    #with open('inception_scores.txt', 'w') as f:
    #    for k, v in results.items():
    #        f.write(f"Folder {k}: {v:.4f}\n")
"""

def plot_graphs(precisions, recalls, inceptions, model_name, dataset_name, res):
    x_vals = range(1,31)
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals,precisions, label="Precision")
    plt.plot(x_vals,recalls, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Precision and Recall")
    plt.savefig(f"prc_plot_{model_name}_{dataset_name}_{res}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals,precisions, label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Precision")
    plt.savefig(f"precision_plot_{model_name}_{dataset_name}_{res}.png")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_vals,inceptions, label="Inception Score")
    plt.xlabel("Epoch")
    plt.ylabel("Inception Score")
    plt.legend()
    plt.title("Inception Score")
    plt.savefig(f"inception_plot_{model_name}_{dataset_name}_{res}.png")
    plt.show()
if __name__=="__main__":
    prs, recalls, inceptions = compute_metrics("FashionMNIST","stylegan_FashionMNIST_64", 64)
    plot_graphs(prs, recalls, inceptions, "stylegan", "FashionMNIST", 64)