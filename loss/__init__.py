from importlib import import_module
from .submodule import *
import torch
import torch.nn as nn


def get(args):
    loss_name = 'Loss'
    module_name = 'loss.' + loss_name.lower()
    module = import_module(module_name)

    return getattr(module, loss_name)


class BaseLoss:
    def __init__(self, args, loss):

        self.args = args
        self.loss_dict = {}
        self.loss_module = nn.ModuleList()

        # Loss configuration : w1*l1+w2*l2+w3*l3+...
        # Ex : 1.0*L1+0.5*L2+...
        for loss_item in loss.split('+'):
            weight, loss_type = loss_item.split('*')

            module_name = 'loss.submodule.' + loss_type.lower() + 'loss'
            module = import_module(module_name)
            loss_func = getattr(module, loss_type + 'Loss')()

            loss_tmp = {
                'weight': float(weight),
                'func': loss_func
            }

            self.loss_dict.update({loss_type: loss_tmp})
            self.loss_module.append(loss_func)

        self.loss_dict.update({'Total': {'weight': 1.0, 'func': None}})

    def __call__(self, output, sample):
        return self.compute(output, sample)

    def cuda_gpu(self, gpu):
        self.loss_module.cuda(gpu)

    def cuda(self):
        self.loss_module.cuda()

    def compute(self, output, sample):
        loss_val = []
        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is not None:
                loss_tmp = loss['weight'] * loss_func(output, sample)
                loss_val.append(loss_tmp)

        loss_val = torch.cat(loss_val, dim=1)
        loss_sum = torch.sum(loss_val)

        return loss_sum, loss_val
