import argparse


def str2bool(s):
    return s.lower() == "true"


def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100", "Caltech101", "Caltech256", "OxfordPets", "OxfordFlowers"])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)

    # model
    parser.add_argument("--model", type=str, default="MLP", choices=["MLP", "AdamBNN", "BCDNet"])
    parser.add_argument("--freeze_num", type=int, default=0, choices=[0,2,4,6,8,10,12,14])
    parser.add_argument("--xnor", type=str2bool, default=False)
    parser.add_argument("--approx", type=str, default="STE", choices=["STE", "ReSTE", "ApproxSign"])

    # for MLP
    parser.add_argument("--hid_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    
    # training algorithm
    parser.add_argument("--algorithm", type=str, default="real", choices=["binary", "real", "bop"])
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])

    # for BinaryTuner
    parser.add_argument("--mask_type", type=str, default="EWM", choices=["EWM", "WPM", "RAND"])
    parser.add_argument("--scheduler_type", type=str, default="scaled-gaussian", choices=["const", "gaussian", "scaled-gaussian"])

    # for Binary optimzer
    parser.add_argument("--bop_threshold", type=float, default=1e-6)

    # others
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    # wandb
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--agent", action="store_true")

    args = parser.parse_args()

    return args
