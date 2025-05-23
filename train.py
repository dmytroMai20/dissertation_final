import argparse
from config_loader import load_config
from dataset import CustomDataLoader
from trainers import get_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        required=False,
                        choices=['ddpm', 'stylegan2', 'diffusiongan'],
                        default="stylegan2")
    parser.add_argument('--dataset',
                        type=str,
                        required=False,
                        choices=['celeba', 'fashionmnist', 'stl10'],
                        default="stl10")
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    model_name = args.model.lower()
    dataset_name = args.dataset.lower()
    config_path = args.config or f"configs/{model_name}_{dataset_name}.yaml"

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print(f"Preparing dataloader for dataset: {config.dataset}")
    print(config.batch_size)
    dataloader = CustomDataLoader(batch_size=config.batch_size,
                                  resolution=config.res,
                                  dataset_name=config.dataset.lower())
    data_loader = dataloader.get_loader()

    print(f"Starting training for model: {model_name}")
    trainer = get_trainer(model_name, config, data_loader)
    trainer.train()


if __name__ == "__main__":
    main()
