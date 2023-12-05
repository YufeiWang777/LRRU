import glob
import os.path

from dataloaders.utils import *
import torchvision.transforms.functional as TF


glob_dep, glob_gt, glob_K, glob_rgb = None, None, None, None
get_rgb_paths, get_K_paths = None, None

def get_kittipaths(split, args):
    global glob_dep, glob_gt, glob_K, glob_rgb
    global get_rgb_paths, get_K_paths

    if split == 'train':
        glob_dep = os.path.join(
            args.data_folder,
            'train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        )
        glob_gt = os.path.join(
            args.data_folder,
            'train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
        )
        def get_rgb_paths(p):
            if 'image_02' in p:
                tmp = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
            elif 'image_03' in p:
                tmp = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
            else:
                raise ValueError('ERROR')
            return tmp
        def get_K_paths(p):
            return p.split('proj_depth')[0] + 'calibration/calib_cam_to_cam.txt'
    elif split == 'val':
        if args.val == 'full':
            glob_dep = os.path.join(
                args.data_folder,
                'val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                if 'image_02' in p:
                    tmp = p.replace('proj_depth/velodyne_raw/image_02', 'image_02/data')
                elif 'image_03' in p:
                    tmp = p.replace('proj_depth/velodyne_raw/image_03', 'image_03/data')
                else:
                    raise ValueError('ERROR')
                return tmp
            def get_K_paths(p):
                return p.split('proj_depth')[0] + 'calibration/calib_cam_to_cam.txt'
        elif args.val == 'select':
            glob_dep = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/velodyne_raw/*.png'
            )
            glob_gt = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/groundtruth_depth/*.png'
            )
            glob_K = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/intrinsics/*.txt'
            )
            glob_rgb = os.path.join(
                args.data_folder,
                'depth_selection/val_selection_cropped/image/*.png'
            )
    elif split == "test_completion":

        glob_dep = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png'
        )
        glob_gt = None
        glob_K = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/intrinsics/*.txt'
        )
        glob_rgb = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_completion_anonymous/image/*.png'
        )
    elif split == "test_prediction":

        glob_dep, glob_gt = None, None
        glob_K = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_prediction_anonymous/intrinsics/*.txt'
        )
        glob_rgb = os.path.join(
            args.data_folder,
            'depth_selection/test_depth_prediction_anonymous/image/*.png'
        )

    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_dep = sorted(glob.glob(glob_dep))
        paths_gt = sorted(glob.glob(glob_gt))
        if split == 'train' or (split == 'val' and args.val == 'full'):
            paths_rgb = [get_rgb_paths(p) for p in paths_dep]
            paths_K = [get_K_paths(p) for p in paths_dep]
        else:
            paths_rgb = sorted(glob.glob(glob_rgb))
            paths_K = sorted(glob.glob(glob_K))
    else:
        # test only has dep or rgb
        paths_K = sorted(glob.glob(glob_K))
        paths_rgb = sorted(glob.glob(glob_rgb))
        if split == "test_prediction":
            paths_dep = [None] * len(paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_dep = sorted(glob.glob(glob_dep))
        paths_gt = [None] * len(paths_dep)

    if len(paths_dep) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0 and len(paths_K) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_dep) == 0:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_K) == 0:
        raise (RuntimeError("Requested structure images but no structure was found"))

    if len(paths_rgb) != len(paths_dep) or len(paths_rgb) != len(paths_gt) or len(paths_gt) != len(paths_K):
        print(len(paths_dep), len(paths_gt), len(paths_rgb), len(paths_K))

    paths = {'dep':paths_dep,
             'gt': paths_gt,
             'K': paths_K,
             'rgb': paths_rgb}

    items = {}
    for key, val in paths.items():
        if key not in args.dataset:
            items[key] = [None] * len(paths_rgb)
        else:
            items[key] = val

    return items

def kittitransforms(split, args, dep, gt, K, rgb,
                    bottom_crop=False, hflip=False, colorjitter=False,
                    rotation=False, resize=False, random_crop=False,
                    normalize=False, scale_depth=False, noise_num=0.0, rgb_noise_num=0.0):
    if split == 'train':
        bottom_crop = args.train_bottom_crop
        hflip = args.hflip
        colorjitter = args.colorjitter
        rotation = args.rotation
        resize = args.resize
        random_crop = args.train_random_crop
        normalize = args.normalize
        scale_depth = args.scale_depth
        noise_num = args.noise
        rgb_noise_num = args.rgb_noise
    elif split == 'val':
        normalize = args.normalize
        bottom_crop = args.val_bottom_crop
        random_crop = args.val_random_crop
    elif split == 'test_completion':
        normalize = args.normalize
        bottom_crop = args.test_bottom_crop
        random_crop = args.test_random_crop
    else:
        raise ValueError('split is not supported!')

    width, height = dep.size

    flip = torch.FloatTensor(1).uniform_(0, 1).item()
    degree = torch.FloatTensor(1).uniform_(-5.0, 5.0).item()
    _scale = torch.FloatTensor(1).uniform_(1.0, 1.5).item()

    if bottom_crop:
        oheight, owidth = args.val_h, args.val_w
        h_init = height - oheight
        w_init = (width - owidth) // 2
        dep = Crop(dep, h_init, w_init, oheight, owidth) if (dep is not None) else None
        gt = Crop(gt, h_init, w_init, oheight, owidth)if (gt is not None) else None
        rgb = Crop(rgb, h_init, w_init, oheight, owidth) if (rgb is not None) else None
        if K is not None:
            K[3] = K[3] - h_init
            K[2] = K[2] - w_init

    if colorjitter:
        rgb = ColorJitter(rgb) if (rgb is not None) else None

    if hflip:
        dep = Hflip(dep, flip) if (dep is not None) else None
        gt = Hflip(gt, flip) if (gt is not None) else None
        rgb = Hflip(rgb, flip) if (rgb is not None) else None
        if K is not None and flip > 0.5:
            K[2] = width - K[2]

    if rotation:
        dep = Rotation(dep, degree) if (dep is not None) else None
        gt = Rotation(gt, degree)  if (gt is not None) else None
        rgb = Rotation(rgb, degree) if (rgb is not None) else None

    if resize:
        scale = np.int(height * _scale)
        dep = Resize(dep, scale, Image.NEAREST) if (dep is not None) else None
        gt = Resize(gt, scale, Image.NEAREST)  if (gt is not None) else None
        rgb = Resize(rgb, scale, Image.BICUBIC) if (rgb is not None) else None

        if K is not None:
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            K[2] = K[2] * _scale
            K[3] = K[3] * _scale

    if random_crop:
        width_, height_ = dep.size
        rwidth, rheight = args.random_crop_width, args.random_crop_height
        h_init = height_ - rheight
        w_init = int(torch.FloatTensor(1).uniform_(0, (width_-rwidth)//2).item())
        dep = Crop(dep, h_init, w_init, rheight, rwidth) if (dep is not None) else None
        gt = Crop(gt, h_init, w_init, rheight, rwidth) if (gt is not None) else None
        rgb = Crop(rgb, h_init, w_init, rheight, rwidth) if (rgb is not None) else None
        if K is not None:
            K[3] = K[3] - h_init
            K[2] = K[2] - w_init

    dep = TF.to_tensor(np.array(dep)) if (dep is not None) else None
    gt = TF.to_tensor(np.array(gt)) if (gt is not None) else None
    rgb = TF.to_tensor(np.array(rgb, dtype=np.float32)) if (rgb is not None) else None

    if normalize:
        if rgb is not None:
            rgb = TF.normalize(rgb, (90.995, 96.2278, 94.3213), (79.2382, 80.5267, 82.1483), inplace=True)

    if scale_depth:
        dep = dep / _scale if (dep is not None) else None
        gt = gt / _scale if (gt is not None) else None

    rgb_n = torch.FloatTensor(1).uniform_(0, 1).item()
    if rgb_n > 0.2 and rgb_noise_num > 0:
        rgb_noise = torch.normal(mean=torch.zeros_like(rgb), std=args.rgb_noise * torch.FloatTensor(1).uniform_(0.5, 1.5).item())
        rgb = rgb + rgb_noise

    if noise_num:
        reflection = np.clip(np.random.normal(1, scale=0.333332, size=(1, 1)), 0.01, 3)[0, 0]
        noise = torch.normal(mean=0.0, std=dep * reflection * noise_num)
        dep_noise = dep + noise
        dep_noise[dep_noise < 0] = 0
    else:
        dep_noise = dep.clone()
    dep = dep_noise

    return dep, gt, K, rgb

