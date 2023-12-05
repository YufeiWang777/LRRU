### LRRU: Long-short Range Recurrent Updating Networks for Depth Completion (ICCV 2023)

[Project Page](https://npucvr.github.io/LRRU/), [arXiv](https://arxiv.org/abs/2310.08956.pdf)

# I'll be perfecting this github repository in a week!

### Environment
```
CUDA 11.7
CUDNN 8.5.0
torch 1.13.0
torchvision 0.14.0
pip install -r LRRU/requirements.txt
pip3 install opencv-python
pip3 install opencv-python-headless
```

### Dataset

We used two datasets for training and evaluation.

#### KITTI Depth Completion (KITTI DC)

KITTI DC dataset is available at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) and the data structure is:

```
.
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```


#### NYU Depth V2 (NYUv2)

We used preprocessed NYUv2 HDF5 dataset provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).

```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```
After that, you will get a data structure as follows:

```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```


### Pretrained models

#### Models on the KITTI validate dataset.
|   Methods  | Pretrained Model  |   Loss  | RMSE[mm] | MAE[mm] | iRMSE[1/km] | iMAE[1/km] |
|:----------:|-------------------|:-------:|:--------:|:-------:|:-----------:|:----------:|
|  LRRU-Mini | [download link](https://drive.google.com/file/d/18je8eR_EqgtS8IM5dKvr0uy9jBoiMZe6/view?usp=sharing) | L1 + L2 |   806.3  |  210.0  |     2.3     |     0.9    |
|  LRRU-Tiny | [download link](https://drive.google.com/file/d/1nEoC1eUkvB_eZF-t6V_ykogwo0YXoA2l/view?usp=sharing) | L1 + L2 |   763.8  |  198.9  |     2.1     |     0.8    |
| LRRU-Small | [download link](https://drive.google.com/file/d/1YtldwyFsTUwmii4H2_fk8z9OiRLdZniI/view?usp=sharing) | L1 + L2 |   745.3  |  195.7  |     2.0     |     0.8    |
|  LRRU-Base | [download link](https://drive.google.com/file/d/10WTVS7a_5Hjo4f5iNgY0v_KsYuftoDZk/view?usp=sharing) | L1 + L2 |   729.5  |  188.8  |     1.9     |     0.8    |

### Acknowledgments

Thanks the ACs and the reviewers for their insightful comments, which are very helpful to improve our paper!
Thanks for all open source projects that have effectively promoted the development of the depth completion communities!

Supervised methods：
    <a href="https://github.com/fangchangma/self-supervised-depth-completion" target="_blank">S2D</a>, 
    <a href="https://github.com/XinJCheng/CSPN" target="_blank">CSPN</a>, 
    <a href="https://github.com/zzangjinsun/NLSPN_ECCV20" target="_blank">NLSPN</a>, 
    <a href="https://github.com/JUGGHM/PENet_ICRA2021" target="_blank">PENet</a>, 
    <a href="https://github.com/sshan-zhao/ACMNet" target="_blank">ACMNet</a>, 
    <a href="https://github.com/kakaxi314/GuideNet" target="_blank">GuideNet</a>, 
    <a href="https://github.com/USTC-Keyanjie/MDANet_ICRA2021" target="_blank">MDANet</a>, 
    <a href="https://github.com/JiaxiongQ/DeepLiDAR" target="_blank">DeepLiDAR</a>, 
    <a href="https://github.com/anglixjtu/msg_chn_wacv20" target="_blank">MSG-CHN</a>, 
    <a href="https://github.com/wvangansbeke/Sparse-Depth-Completion" target="_blank">Sparse-Depth-Completion</a>, 
    <a href="https://github.com/Wenchao-Du/GAENet" target="_blank">GAENet</a>,    
    <a href="https://github.com/yurimjeon1892/ABCD" target="_blank">ABCD</a>,
    <a href="https://github.com/danishnazir/SemAttNet" target="_blank">SemAttNet</a>, 
    <a href="https://github.com/youmi-zym/CompletionFormer" target="_blank">CompletionFormer</a>, 
    <a href="https://github.com/Kyakaka/DySPN" target="_blank">DySPN</a>, 
    <a href="https://github.com/AlexSunNik/ReDC" target="_blank">ReDC</a>.

Unsupervised methods：
    <a href="https://github.com/fangchangma/self-supervised-depth-completion" target="_blank">S2D</a>, 
    <a href="https://github.com/alexklwong/learning-topology-synthetic-data" target="_blank">ScaffFusion-SSL</a>, 
    <a href="https://github.com/alexklwong/calibrated-backprojection-network" target="_blank">KBNet</a>, 
    <a href="https://github.com/alexklwong/learning-topology-synthetic-data" target="_blank">ScaffFusion</a>, 
    <a href="https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry">VOICED</a>.

If I have accidentally forgotten your work, please contact me to add.



### Citation
```
@InProceedings{LRRU_ICCV_2023,
  author    = {Wang, Yufei and Li, Bo and Zhang, Ge and Liu, Qi and Gao Tao and Dai, Yuchao},
  title     = {LRRU: Long-short Range Recurrent Updating Networks for Depth Completion},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year      = {2023},
}
```
