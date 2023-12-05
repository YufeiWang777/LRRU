"""
    Some of useful functions are defined here.
"""

import os
import shutil

import torch
from collections import OrderedDict

import math
import torch.distributed

from prettytable import PrettyTable

import cv2
import numpy as np


# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    # depth_map = np.squeeze(depth_map, axis=-1)

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]


    # Median blur
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)


    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


def outlier_removal(lidar, thre):

    sparse_lidar = np.squeeze(lidar)
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)

    lidar_sum = cv2.filter2D(sparse_lidar, -1, FULL_KERNEL_3)
    lidar_count = cv2.filter2D(valid_pixels, -1, FULL_KERNEL_3)

    lidar_aveg = lidar_sum / (lidar_count + 0.00001)
    outliers_mask = (sparse_lidar - lidar_aveg) < thre

    return outliers_mask


def remove_moudle(remove_dict):
    for k, v in remove_dict.items():
        if 'module' in k :
            print("==> model dict with addtional module, remove it...")
            removed_dict = { k[7:]: v for k, v in remove_dict.items()}
        else:
            removed_dict = remove_dict
        break
    return removed_dict

def update_conv_spn_model(out_dict, in_dict):
    in_dict = {k: v for k, v in in_dict.items() if k in out_dict}
    return in_dict

def compare_dicts(dict1, dict2):
    for i, j in zip(dict1.items(), dict2.items()):
        if i == j:
            print(i[0])


def freeze_partmodel(model, selected_layers):

    for name, p in model.named_parameters():
        for layername in selected_layers:
            if layername in name:
                p.requires_grad = False
                print(name)
                break

def select_partmodel(pretrained_dict, selected_layers):

    partmodel = OrderedDict()
    for k, v in pretrained_dict.items():
        for layername in selected_layers:
            if layername in k:
                partmodel[k] = v
                break

    return partmodel


def replace_relu2leaky(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.ReLU:
            print('replaced: ', name, attr_str)
            setattr(m, attr_str, torch.nn.LeakyReLU(0.2, inplace=True))
    for n, ch in m.named_children():
        replace_relu2leaky(ch, n)

def replace_relu2elu(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.ReLU:
            print('replaced: ', name, attr_str)
            setattr(m, attr_str,torch.nn.ELU(inplace=True))
    for n, ch in m.named_children():
        replace_relu2elu(ch, n)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Params:{total_params}")
    return total_params

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*", '*wandb*', '*test*', '*val*')

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

def check_args(args):
    if args.batch_size < args.num_gpus:
        print("batch_size changed : {} -> {}".format(args.batch_size,
                                                     args.num_gpus))
        args.batch_size = args.num_gpus

    new_args = args
    if not args.pretrain == '':
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)
            new_args.defrost()
            new_args.start_epoch = checkpoint['epoch']
            new_args.freeze()
            pre_args = checkpoint['args']
            # check if the important parametes setting is same as the pre setting
            # * dataset
            assert new_args.data_name == pre_args.data_name
            assert new_args.patch_height == pre_args.patch_height
            assert new_args.patch_width == pre_args.patch_width
            assert new_args.top_crop == pre_args.top_crop
            assert new_args.max_depth == pre_args.max_depth
            assert new_args.augment == pre_args.augment
            assert new_args.num_sample == pre_args.num_sample
            assert new_args.test_crop == pre_args.test_crop
            # * loss
            assert new_args.loss == pre_args.loss
            # * apex level
            assert new_args.opt_level == pre_args.opt_level
            # * training
            assert new_args.epochs == pre_args.epochs
            # assert new_args.batch_size == pre_args.batch_size
            # * optimizer
            # assert new_args.lr == pre_args.lr
            assert new_args.optimizer == pre_args.optimizer
            assert new_args.momentum == pre_args.momentum
            assert new_args.betas == pre_args.betas
            assert new_args.epsilon == pre_args.epsilon
            assert new_args.weight_decay == pre_args.weight_decay
            assert new_args.scheduler == pre_args.scheduler
            assert new_args.decay_step == pre_args.decay_step
            assert new_args.decay_factor == pre_args.decay_factor

    return new_args

def count_validpoint(x: torch.Tensor):
    mask = x > 0.001
    num_valid = mask.sum()
    return int(num_valid.data.cpu())

def pad_rep(image, ori_size):
    h, w = image.shape
    (oh, ow) = ori_size
    pl = (ow - w) // 2
    pr = ow - w - pl
    pt = oh - h
    image_pad = np.pad(image, pad_width=((pt, 0), (pl, pr)), mode='edge')
    return image_pad
