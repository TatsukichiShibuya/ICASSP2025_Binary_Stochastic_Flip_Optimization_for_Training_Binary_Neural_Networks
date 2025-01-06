from utils import *
from datasets import get_dataset
from get_args import get_args
from models.mask_generator import get_mask_generator
from models.temp_scheduler import *

from models.MLP import *
from models.AdamBNN import *
from models.BCDNet import *
from models.base_memory_efficient import sign_function

import os
import gc
import sys
import wandb
import torch
import numpy as np

from torch import nn
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


DATA_INFO = {
    "MNIST"        : {"input": (64,1),  "output": 10},
    "CIFAR10"      : {"input": (32,3),  "output": 10},
    "CIFAR100"     : {"input": (32,3),  "output": 100},
    "Caltech101"   : {"input": (256,3), "output": 101},
    "Caltech256"   : {"input": (256,3), "output": 256},
    "OxfordPets"   : {"input": (256,3), "output": 37},
    "OxfordFlowers": {"input": (256,3), "output": 102},
}

with open(".strage_path.txt") as f:
    STRAGE_PATH = f.read()

N_GPUS_PER_NODE = 4
RANK = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
LOCAL_RANK = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
WORLD_SIZE = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
DEVICE = torch.device('cuda', LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")


def main(kwargs):
    if kwargs["log"] and RANK==0:
        wandb.init(config=kwargs)

    fix_seed(kwargs["seed"])
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv('MASTER_PORT', default='8888')
    method = "tcp://{}:{}".format(master_addr, master_port)

    dist.init_process_group("nccl", init_method=method, rank=RANK, world_size=WORLD_SIZE)
    
    train_set, valid_set, loss_function = get_dataset(kwargs["dataset"], kwargs["test"])
    train_sampler = DistributedSampler(train_set, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=kwargs["seed"])
    test_sampler = DistributedSampler(valid_set, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=kwargs["batch_size"]//WORLD_SIZE, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=kwargs["batch_size"]//WORLD_SIZE, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, sampler=test_sampler)
    
    model = get_model(model=kwargs["model"], depth=kwargs["depth"], hid_dim=kwargs["hid_dim"], dataset=kwargs["dataset"], xnor=kwargs["xnor"], approx=kwargs["approx"], freeze_num=kwargs["freeze_num"]).to(DEVICE)
    model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    if kwargs["optimizer"] == "Adam":
        train_adam(model, train_loader, valid_loader, kwargs)
    else:
        train(model, train_loader, valid_loader, kwargs)


def train_adam(model, train_loader, valid_loader, kwargs):
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    loss_function_sum = nn.CrossEntropyLoss(reduction="sum")

    optimizer = optim.Adam(model.module.parameters(), lr=kwargs["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["epochs"])
    
    for e in range(kwargs["epochs"]):
        if RANK==0:
            print(f"Epoch: {e+1}")
        scheduler.step()

        if kwargs["approx"]=="ReSTE":
            model.module.update_rectified_scale(e/kwargs["epochs"])

        for i, (x, y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            gc.collect()
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_pred = model(x)
            loss = loss_function(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        results = {}
        results.update(evaluate(model, train_loader, loss_function_sum, topk=1, valid=False))
        results.update(evaluate(model, valid_loader, loss_function_sum, topk=[1,5]))
        
        if RANK==0:
            send_log(results, wandb_flag=kwargs["log"])
    

def train(model, train_loader, valid_loader, kwargs):
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    loss_function_sum = nn.CrossEntropyLoss(reduction="sum")
    mask_generator = {}
    momentum_bop = {}

    for e in range(kwargs["epochs"]):
        if RANK==0:
            print(f"Epoch: {e+1}")

        if kwargs["approx"]=="ReSTE":
            model.module.update_rectified_scale(e/kwargs["epochs"])

        for i, (x, y) in enumerate(train_loader):
            torch.cuda.empty_cache()
            gc.collect()
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_pred = model(x)
            loss = loss_function(y_pred, y)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    
                    grad = param.grad.data
                    coef = (x.shape[0]/96)**0.5
                    s_name = name.split(".")

                    ########################################################################################################################

                    if kwargs["model"] in ["AdamBNN", "BCDNet"]:
                        binary_flag = "binary" in name
                        real_flag = ~binary_flag

                        if real_flag:
                            param.data -= kwargs["lr"]*coef * param.grad.data

                        coef_b = coef*0.1
                        if binary_flag:
                            # Real-space SGD
                            if kwargs["algorithm"]=="real":
                                param.data -= kwargs["lr"] * coef_b * param.grad.data

                            # BinaryTuner
                            elif kwargs["algorithm"]=="binary":
                                if name not in mask_generator:
                                    if kwargs["scheduler_type"] in ["gaussian", "scaled-gaussian", "const"]:
                                        scheduler_params = {}
                                        scheduler_params["accumulator"] = param.data.std()
                                        scheduler_params["lr"] = kwargs["lr"]*coef_b
                                        param.data = sign_function(param.data)
                                        mask_generator[name] = get_mask_generator(mask_type=kwargs["mask_type"], 
                                                                                  scheduler_type=kwargs["scheduler_type"], 
                                                                                  scheduler_params=scheduler_params)
                                    else:
                                        raise NotImplementedError()
                                else:
                                    mask_generator[name].scheduler.lr.update(kwargs["lr"]*coef_b)

                                mask = mask_generator[name].get_mask(param.grad.data, sign_function(param.data))
                                param_target = sign_function(-param.grad.data)
                                param.data[mask] = param_target[mask]

                            # Binary optimizer
                            elif kwargs["algorithm"]=="bop":
                                if name not in momentum_bop:
                                    momentum_bop[name] = -param.data
                                    param.data = sign_function(param.data)

                                momentum_bop[name] = (1-kwargs["lr"]*coef_b)*momentum_bop[name] + kwargs["lr"]*coef_b*grad
                                mask = (momentum_bop[name].abs()>kwargs["bop_threshold"])&(sign_function(param.data)==sign_function(momentum_bop[name]))
                                param.data[mask] = -param.data[mask]

                    ########################################################################################################################

                    elif kwargs["model"]=="MLP":
                        # Real-space SGD
                        if kwargs["algorithm"]=="real":
                            param.data -= kwargs["lr"]*coef * grad

                        # BinaryTuner
                        elif kwargs["algorithm"]=="binary":
                            if name not in mask_generator:
                                scheduler_params = {}
                                scheduler_params["accumulator"] = 0.01
                                scheduler_params["lr"] = kwargs["lr"]*coef
                                mask_generator[name] = get_mask_generator(mask_type=kwargs["mask_type"], 
                                                                          scheduler_type=kwargs["scheduler_type"], 
                                                                          scheduler_params=scheduler_params)
                            else:
                                mask_generator[name].scheduler.lr.update(kwargs["lr"]*coef)
                            
                            mask = mask_generator[name].get_mask(grad, sign_function(param.data))
                            param_target = sign_function(-grad)
                            param.data[mask] = param_target[mask].to(torch.float32)

                        # Binary optimizer
                        elif kwargs["algorithm"]=="bop":
                            if name not in momentum_bop:
                                momentum_bop[name] = torch.zeros_like(param).to(DEVICE)
                                param.data = sign_function(param.data)

                            momentum_bop[name] = (1-kwargs["lr"]*coef)*momentum_bop[name] + kwargs["lr"]*coef*grad
                            mask = (momentum_bop[name].abs()>kwargs["bop_threshold"])&(sign_function(param.data)==sign_function(momentum_bop[name]))
                            param.data[mask] = -param.data[mask]
                    
                    ########################################################################################################################
        
        results = {}
        results.update(evaluate(model, train_loader, loss_function_sum, topk=1, valid=False))
        results.update(evaluate(model, valid_loader, loss_function_sum, topk=[1,5]))
        
        if RANK==0:
            send_log(results, wandb_flag=kwargs["log"])


def evaluate(model, data_loader, loss_function, topk, valid=True):
    if not isinstance(topk, list):
        topk = [topk]

    with torch.no_grad():
        total_count, loss, acc = test(model, data_loader, loss_function=loss_function, topk=topk)
    model.train()

    total_tensor = torch.tensor([total_count]).to(DEVICE)
    loss_tensor = torch.tensor([loss]).to(DEVICE)
    acc_tensor = torch.tensor(acc).to(DEVICE)
        
    torch.distributed.reduce(total_tensor, dst=0)
    torch.distributed.reduce(loss_tensor, dst=0)
    torch.distributed.reduce(acc_tensor, dst=0)

    results = {}
    if RANK==0:
        results[f"{'valid' if valid else 'train'} loss"] = (loss_tensor[0] / total_tensor[0]).item()
        for i, k in enumerate(topk):
            results[f"{'valid' if valid else 'train'} accuracy (top-{k})"] = (acc_tensor[i] / total_tensor[0]).item()
    return results


def test(model, data_loader, loss_function, topk):
    loss, acc = 0, [0 for k in topk]
    total_count = 0

    for i, (x, y) in enumerate(data_loader):
        torch.cuda.empty_cache()
        gc.collect()

        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x)
        loss += loss_function(y_pred, y)
        for j, k in enumerate(topk):
            acc[j] += calc_accuracy(y_pred, y, k=k, sum=True)
        total_count += len(x)

    return total_count, loss, acc


def get_model(model, depth, hid_dim, dataset, xnor, approx, freeze_num):

    in_shape = DATA_INFO[dataset]["input"]
    out_dim = DATA_INFO[dataset]["output"]

    if model=="MLP":
        model = get_mlp(in_shape=in_shape, hid_dim=hid_dim, out_dim=out_dim, depth=depth, approx=approx)

    elif model=="AdamBNN":
        model = get_adambnn(out_dim=out_dim, xnor=xnor, approx=approx, freeze_num=freeze_num)

    elif model=="BCDNet":
        model = get_bcdnet(out_dim=out_dim, xnor=xnor, approx=approx)

    return model


if __name__ == '__main__':
    FLAGS = vars(get_args())

    main(FLAGS)

    dist.destroy_process_group()