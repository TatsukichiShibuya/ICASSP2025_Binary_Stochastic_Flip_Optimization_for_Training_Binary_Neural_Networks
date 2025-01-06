import os
import wandb
import numpy as np
import torch
from torch import nn


def calc_accuracy(pred, label, k=1, sum=False):
    topk_indices = torch.topk(pred, k=k).indices.T
    count = 0
    for i in range(k):
        count += (topk_indices[i] == label).sum().item()
    return count if sum else count / label.shape[0]


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_wandb(args, params):
    config = args.copy()
    for param in params["forward_function"]:
        config["forward_function_" + param] = params["forward_function"][param]
    if "backward_function" in params:
        for param in params["backward_function"]:
            config["backward_function_" + param] = params["backward_function"][param]
    config["last_activation"] = params["last_activation"]
    wandb.init(config=config)


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        os.environ['OMP_NUM_THREADS'] = '1'
    return device


def send_log(results, wandb_flag):
    if wandb_flag:
        wandb.log(results)
    else:
        for key in results:
            if results[key] is not None:
                print(f"\t{key}\t: {results[key]}")