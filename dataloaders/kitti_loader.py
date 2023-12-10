import torch.utils.data as data

from dataloaders.paths_and_transform import *
from dataloaders.NNfill import *


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        self.paths = get_kittipaths(split, args)
        self.transforms = kittitransforms
        self.ipfill = fill_in_fast

    def __getraw__(self, index):
        dep = read_depth(self.paths['dep'][index]) if \
            (self.paths['dep'][index] is not None) else None
        gt = read_depth(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        rgb = read_rgb(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None) else None

        if self.paths['K'][index] is not None:
            if self.split == 'train' or (self.split == 'val' and self.args.val == 'full'):
                calib = read_calib_file(self.paths['K'][index])
                K_cam = None
                if 'image_02' in self.paths['rgb'][index]:
                    K_cam = np.reshape(calib['P_rect_02'], (3, 4))
                elif 'image_03' in self.paths['rgb'][index]:
                    K_cam = np.reshape(calib['P_rect_03'], (3, 4))
                K = [K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2]]
            else:
                f_calib = open(self.paths['K'][index], 'r')
                K_cam = f_calib.readline().split(' ')
                f_calib.close()
                K = [float(K_cam[0]), float(K_cam[4]), float(K_cam[2]),
                     float(K_cam[5])]
        else:
            K = None
        return dep, gt, K, rgb, self.paths['dep'][index]

    def __getitem__(self, index):

        dep, gt, K, rgb, paths = self.__getraw__(index)
        dep, gt, K, rgb = self.transforms(self.split, self.args, dep, gt, K, rgb)

        dep_np = dep.numpy().squeeze(0)
        dep_clear, _ = outlier_removal(dep_np)
        dep_clear = np.expand_dims(dep_clear, 0)
        dep_clear_torch = torch.from_numpy(dep_clear)

        # ip_basic fill
        dep_np = dep.numpy().squeeze(0)
        dep_np_ip = np.copy(dep_np)

        dep_ip = self.ipfill(dep_np_ip, max_depth=100.0,
                            extrapolate=True, blur_type='gaussian')

        dep_ip_torch = torch.from_numpy(dep_ip)
        dep_ip_torch = dep_ip_torch.to(dtype=torch.float32)

        candidates = {'dep': dep, 'dep_clear':dep_clear_torch, 'gt': gt, 'rgb': rgb, 'ip': dep_ip_torch}
        items = {
            key: val
            for key, val in candidates.items() if val is not None
        }
        if self.args.debug_dp or self.args.test:
            items['d_path'] = paths

        return items

    def __len__(self):
        if self.args.toy_test:
            return self.args.toy_test_number
        else:
            return len(self.paths['gt'])
