import argparse
#from config_loader import load_config
from dataset import CustomDataLoader
#from trainers import get_trainer
from torch_fidelity import calculate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rpath', type=str,
                        required=True)  # path to real pictures
    parser.add_argument('--gpath', type=str,
                        required=True)  # path to generated pictures
    args = parser.parse_args()

    real_path = args.rpath
    g_path = args.gpath

    prc_dict = calculate_metrics(input1=real_path,
                                 input2=g_path,
                                 cuda=True,
                                 isc=False,
                                 fid=True,
                                 kid=True,
                                 prc=True,
                                 verbose=False)
    inception_dict = calculate_metrics(input1=g_path,
                                       cuda=True,
                                       isc=True,
                                       fid=False,
                                       kid=False,
                                       prc=False,
                                       verbose=False)
    prc_dict['inception_score_mean'] = inception_dict['inception_score_mean']
    for key, value in prc_dict.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
