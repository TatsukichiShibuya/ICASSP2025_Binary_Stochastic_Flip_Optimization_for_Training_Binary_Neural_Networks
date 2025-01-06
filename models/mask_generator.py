import torch
from torch import erf
import numpy as np
from abc import ABCMeta, abstractmethod
from models.temp_scheduler import *


ERFINV = {0.5: 0.4769362762044698733814, 
         -0.5: -0.4769362762044698733814}
    
class MaskGenerator(metaclass=ABCMeta):
    def __init__(self, mask_type, scheduler_type, scheduler_params):
        self.mask_type = mask_type
        self.scheduler_type = scheduler_type
        self._set_scheduler(self.scheduler_type, scheduler_params)

    def _set_scheduler(self, scheduler_type, scheduler_params):
        if self.scheduler_type == "const":
            self.scheduler = ConstScheduler(scheduler_params["accumulator"], 
                                            scheduler_params["lr"])
        elif self.scheduler_type == "gaussian":
            self.scheduler = GaussianScheduler(scheduler_params["accumulator"], 
                                               scheduler_params["lr"])
        elif self.scheduler_type == "scaled-gaussian":
            self.scheduler = ScaledGaussianScheduler(scheduler_params["accumulator"], 
                                                     scheduler_params["lr"])
        else:
            raise NotImplementedError(f"SCHEDULER ERROR: [{self.scheduler_type}] is not available.")

    @abstractmethod
    def _get_prob(self, grad, b_weight, temp):
        pass

    def get_temp(self):
        return self.scheduler.get_temp(update=False)

    def get_mask(self, grad, b_weight):
        temp = self.scheduler.get_temp(update=True, update_step=(grad.std() if self.scheduler_type in ["gaussian", "scaled-gaussian"] else 1))
        prob = self._get_prob(grad, b_weight, temp)
        mask = torch.bernoulli(prob).bool()

        return mask

class EWMGenerator(MaskGenerator):
    def _get_prob(self, grad, b_weight, temp):
        prob = torch.zeros_like(grad, device=grad.device)
        prob_pos = erf(temp*grad) - erf(torch.min(torch.zeros_like(grad, device=grad.device), temp*grad))
        prob_neg = erf(torch.max(torch.zeros_like(grad, device=grad.device), temp*grad)) - erf(temp*grad)
        prob[b_weight>0] = prob_pos[b_weight>0]
        prob[b_weight<0] = prob_neg[b_weight<0]
        return prob

class WPMGenerator(MaskGenerator):
    def _get_prob(self, grad, b_weight, temp):
        prob = torch.zeros_like(grad, device=grad.device)
        prob[(b_weight>0)&(grad>=ERFINV[0.5]/temp)] = 1
        prob[(b_weight<0)&(grad<=ERFINV[-0.5]/temp)] = 1
        prob[(b_weight>0)&(grad<ERFINV[0.5]/temp)] = 0
        prob[(b_weight<0)&(grad>ERFINV[-0.5]/temp)] = 0
        return prob

class RANDGenerator(MaskGenerator):
    def _get_prob(self, grad, b_weight, temp):
        prob = torch.ones_like(grad, device=grad.device)
        prob *= temp
        prob[prob>1] = 1
        prob[prob<0] = 0
        return prob


def get_mask_generator(mask_type, scheduler_type, scheduler_params, th=None):
    if mask_type=="EWM":
        mask_generator = EWMGenerator(mask_type=mask_type, scheduler_type=scheduler_type, scheduler_params=scheduler_params)
    elif mask_type=="WPM":
        mask_generator = WPMGenerator(mask_type=mask_type, scheduler_type=scheduler_type, scheduler_params=scheduler_params)
    elif mask_type=="RAND":
        mask_generator = RANDGenerator(mask_type=mask_type, scheduler_type=scheduler_type, scheduler_params=scheduler_params)
    else:
        raise NotImplementedError(f"MASK ERROR: [{mask_type}] is not available.")
    return mask_generator