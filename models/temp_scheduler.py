import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def get_temp(self, update, update_step):
        raise NotImplementedError()


class ManualScheduler(Scheduler):
    def __init__(self, lr):
        self.lr = lr
    
    def get_temp(self):
        return self.lr
    
    def update(self, lr):
        self.lr = lr

class AdaptiveScheduler(Scheduler):
    def __init__(self, accumulator, lr):
        self.accumulator = accumulator
        if isinstance(lr, float):
            self.lr = ManualScheduler(lr)
        else:
            raise RuntimeError()

class GaussianScheduler(AdaptiveScheduler):
    def get_temp(self, update=True, update_step=None):
        lr = self.lr.get_temp()
        temp = lr/(2**0.5 * self.accumulator)

        if update:
            self.accumulator = ((self.accumulator**2 + (lr*update_step)**2)**0.5).item()
        
        return temp

class ScaledGaussianScheduler(AdaptiveScheduler):
    def get_temp(self, update=True, update_step=None):
        lr = self.lr.get_temp()
        temp = lr*np.pi**0.5/(2 * self.accumulator**2)

        if update:
            self.accumulator = ((self.accumulator**2 + np.pi/2 * lr**2 * (update_step/self.accumulator)**2)**0.5).item()

        return temp

class ConstScheduler(AdaptiveScheduler):
    def get_temp(self, update=True, update_step=None):
        lr = self.lr.get_temp()
        return lr