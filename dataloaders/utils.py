import os
import cv2
import torch
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

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

# 9x9 diamond kernel
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

# 13x13 diamond kernel
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

def outlier_removal(lidar):
    # sparse_lidar = np.squeeze(lidar)
    threshold = 1.0
    sparse_lidar = lidar
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)

    lidar_sum_7 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_7)
    lidar_count_7 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_7)

    lidar_aveg_7 = lidar_sum_7 / (lidar_count_7 + 0.00001)
    potential_outliers_7 = ((sparse_lidar - lidar_aveg_7) > threshold).astype(np.float)

    lidar_sum_9 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_9)
    lidar_count_9 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_9)

    lidar_aveg_9 = lidar_sum_9 / (lidar_count_9 + 0.00001)
    potential_outliers_9 = ((sparse_lidar - lidar_aveg_9) > threshold).astype(np.float)

    lidar_sum_13 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_13)
    lidar_count_13 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_13)

    lidar_aveg_13 = lidar_sum_13 / (lidar_count_13 + 0.00001)
    potential_outliers_13 = ((sparse_lidar - lidar_aveg_13) > threshold).astype(np.float)

    potential_outliers = potential_outliers_7 + potential_outliers_9 + potential_outliers_13
    lidar_cleared = (sparse_lidar * (1 - potential_outliers)).astype(np.float32)
    # lidar_cleared = np.expand_dims(lidar_cleared, -1)

    # potential_outliers = np.expand_dims(potential_outliers, -1)

    return lidar_cleared, potential_outliers

# def outlier_removal(lidar):
#     # sparse_lidar = np.squeeze(lidar)
#     threshold = 0.01
#     sparse_lidar = lidar
#     valid_pixels = (sparse_lidar > 0.1).astype(np.float)
#
#     lidar_sum_3 = cv2.filter2D(sparse_lidar, -1, FULL_KERNEL_3)
#     lidar_count_3 = cv2.filter2D(valid_pixels, -1, FULL_KERNEL_3)
#
#     lidar_aveg_3 = lidar_sum_3 / (lidar_count_3 + 0.00001)
#     potential_outliers_3 = ((sparse_lidar - lidar_aveg_3) > threshold).astype(np.float)
#
#
#     lidar_sum_7 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_7)
#     lidar_count_7 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_7)
#
#     lidar_aveg_7 = lidar_sum_7 / (lidar_count_7 + 0.00001)
#     potential_outliers_7 = ((sparse_lidar - lidar_aveg_7) > threshold).astype(np.float)
#
#     lidar_sum_9 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_9)
#     lidar_count_9 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_9)
#
#     lidar_aveg_9 = lidar_sum_9 / (lidar_count_9 + 0.00001)
#     potential_outliers_9 = ((sparse_lidar - lidar_aveg_9) > threshold).astype(np.float)
#
#     lidar_sum_13 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_13)
#     lidar_count_13 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_13)
#
#     lidar_aveg_13 = lidar_sum_13 / (lidar_count_13 + 0.00001)
#     potential_outliers_13 = ((sparse_lidar - lidar_aveg_13) > threshold).astype(np.float)
#
#     potential_outliers = potential_outliers_3 + potential_outliers_7 + potential_outliers_9 + potential_outliers_13
#     lidar_cleared = (sparse_lidar * (1 - potential_outliers)).astype(np.float32)
#     # lidar_cleared = np.expand_dims(lidar_cleared, -1)
#
#     # potential_outliers = np.expand_dims(potential_outliers, -1)
#
#     return lidar_cleared, potential_outliers

def get_sparse_depth(dep, num_spot):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_spot]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp
#
# def get_sparse_depth(dep, num_sample):
#     channel, height, width = dep.shape
#
#     assert channel == 1
#
#     idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
#
#     num_idx = len(idx_nnz)
#     idx_sample = torch.randperm(num_idx)[:num_sample]
#
#     idx_nnz = idx_nnz[idx_sample[:]]
#
#     mask = torch.zeros((channel * height * width))
#     mask[idx_nnz] = 1.0
#     mask = mask.view((channel, height, width))
#
#     dep_sp = dep * mask.type_as(dep)
#
#     return dep_sp


def get_sparse_depth_grid(dep):
    """
    Simulate pincushion distortion:
    --stride:
    It controls the distance between neighbor spots7
    Suggest stride value:       5~10

    --dist_coef:
    It controls the curvature of the spot pattern
    Larger dist_coef distorts the pattern more.
    Suggest dist_coef value:    0 ~ 5e-5

    --noise:
    standard deviation of the spot shift
    Suggest noise value:        0 ~ 0.5
    """

    # Generate Grid points
    channel, img_h, img_w = dep.shape
    assert channel == 1

    stride = np.random.randint(5, 7)

    dist_coef = np.random.rand() * 4e-5 + 1e-5
    noise = np.random.rand() * 0.3

    x_odd, y_odd = np.meshgrid(np.arange(stride // 2, img_h, stride * 2), np.arange(stride // 2, img_w, stride))
    x_even, y_even = np.meshgrid(np.arange(stride // 2 + stride, img_h, stride * 2), np.arange(stride, img_w, stride))
    x_u = np.concatenate((x_odd.ravel(), x_even.ravel()))
    y_u = np.concatenate((y_odd.ravel(), y_even.ravel()))
    x_c = img_h // 2 + np.random.rand() * 50 - 25
    y_c = img_w // 2 + np.random.rand() * 50 - 25
    x_u = x_u - x_c
    y_u = y_u - y_c

    # Distortion
    r_u = np.sqrt(x_u ** 2 + y_u ** 2)
    r_d = r_u + dist_coef * r_u ** 3
    num_d = r_d.size
    sin_theta = x_u / r_u
    cos_theta = y_u / r_u
    x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
    idx_mask = (x_d < img_h) & (x_d > 0) & (y_d < img_w) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    spot_mask = np.zeros((img_h, img_w))
    spot_mask[x_d, y_d] = 1

    dep_sp = torch.zeros_like(dep)
    dep_sp[:, x_d, y_d] = dep[:, x_d, y_d]

    return dep_sp


def cut_mask(dep):
    _, h, w = dep.size()
    c_x = np.random.randint(h / 4, h / 4 * 3)
    c_y = np.random.randint(w / 4, w / 4 * 3)
    r_x = np.random.randint(h / 4, h / 4 * 3)
    r_y = np.random.randint(h / 4, h / 4 * 3)

    mask = torch.zeros_like(dep)
    min_x = max(c_x - r_x, 0)
    max_x = min(c_x + r_x, h)
    min_y = max(c_y - r_y, 0)
    max_y = min(c_y + r_y, w)
    mask[0, min_x:max_x, min_y:max_y] = 1

    return dep * mask


def get_sparse_depth_prop(dep, prop):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    num_sample = int(num_idx * prop)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp



def get_sparse_depthv2(dep, num_sample):
    channel, height, width = dep.shape

    assert channel == 1

    num_sample

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel * height * width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp

def read_rgb(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # img_file.close()
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    # rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    return img_file

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    img_file = Image.open(file_name)
    image_depth = np.array(img_file, dtype=int)
    # img_file.close()

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    depth = Image.fromarray(image_depth.astype('float32'), mode='F')
    return depth

# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def Crop(img, h_init, w_init, crop_h, crop_w):
    return TF.crop(img, h_init, w_init, crop_h, crop_w)

def Hflip(img, flip):
    if flip > 0.5:
        return TF.hflip(img)
    else:
        return img

def ColorJitter(img):
    # METHOD1
    # brightness = torch.FloatTensor(1).uniform_(0.6, 1.4)
    # contrast = torch.FloatTensor(1).uniform_(0.6, 1.4)
    # saturation = torch.FloatTensor(1).uniform_(0.6, 1.4)
    #
    # img = TF.adjust_brightness(img, brightness)
    # img = TF.adjust_contrast(img, contrast)
    # img = TF.adjust_saturation(img, saturation)

    # METHOD2
    # borrow from https://github.com/kujason/avod/blob/master/avod/datasets/kitti/kitti_aug.py
    img_np = np.array(img)
    pca = compute_pca(img_np)
    img_np = add_pca_jitter(img_np, pca)
    img = Image.fromarray(img_np, 'RGB')

    return img

def compute_pca(image):
    """
    calculate PCA of image
    """

    reshaped_data = image.reshape(-1, 3)
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)
    pca = np.sqrt(e_vals) * e_vecs
    return pca

def add_pca_jitter(img_data, pca):
    """
    add a multiple of principal components with Gaussian noise
    """
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data

# def Rotation(img, degree, mode):
#     return TF.rotate(img, angle=degree, resample=mode)

def Rotation(img, degree):
    return TF.rotate(img, angle=degree)

# def Resize(img, scale, mode):
#     return TF.resize(img, scale, mode)

def Resize(img, size, mode=None):
    if mode:
        return TF.resize(img, size, mode)
    else:
        return TF.resize(img, size)

