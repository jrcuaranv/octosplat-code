<!-- PROJECT LOGO -->
Code implementation for our paper [OctoSplat: Hybrid OctoMap-Gaussian Splatting for Active Semantic Mapping and Phenotyping with Horticultural Robots](https://jrcuaranv.github.io/octosplat/)


## Table of Contents
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Overview</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#download-dataset">Download Dataset</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#saving-and-visualization">Saving and Visualization</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>


## Installation
We build our framework upon [SGS-SLAM](https://github.com/ShuhongLL/SGS-SLAM). Please refer to its website for further installation details if necessary.

```bash
conda create -n sgs-slam python=3.9
conda activate sgs-slam
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cudatoolkit=11.8 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Download Dataset

[Simulation experiments](https://uofi.box.com/shared/static/j4ekey2bo0k27c5rtrf91sr5867ngpnb.zip)

[Laboratory experiments](https://uofi.box.com/shared/static/8noydzxkpwqdhj995vj4bdloi5uyu9ob.zip)

[Real greenhouse experiments](https://uofi.box.com/shared/static/aisgg0w9jufvccw3i7dmu0l5xrh5j5gk.zip)


<details>
  <summary>[Directory structure of training and evaluation scenes]</summary>

```
  SCENE_DIR
  └── images
        └── 2026xxxx.png
        └── 2026xxxx.png
  └── semantics
        └── 2026xxxx.png
        └── 2026xxxx.png
  └── confidences
        └── 2026xxxx.png
        └── 2026xxxx.png
  └── depth
        └── 2026xxxx.png
        └── 2026xxxx.png
  └── poses
        └── 2026xxxx.txt
        └── 2026xxxx.txt
  └── intrinsics.txt
```
</details>

## Usage

To run OctoSplat on a specific scene, provide the training and testing data paths:

```bash
train_data_path="/mnt/ssd2T/datasets/clean_octosplat_data/gazebo_noisy_seg_noisy_depth/2026-06-25_13-18-05_g1_row3"
test_data_path="/mnt/ssd2T/datasets/clean_octosplat_data/gazebo_eval_data_folders/greenhouse_1/row_3"
python3 scripts/octosplat.py configs/octosplat/slam.py --train_data_path $train_data_path --test_data_path $test_data_path
```

## Saving and Visualization

By default, the system stores the reconstructed scenes in `.npz` format, which includes both appearance and semantic features. Additionally, we save the final RGB and semantic maps in `.ply` format for easier visualization. You can view the scenes using any 3DGS viewer, such as [SuperSplat](https://playcanvas.com/supersplat/editor/) and [WebGL 3D Gaussian Splat Viewer](https://antimatter15.com/splat/).

## Acknowledgement

Our work is based on [SGS-SLAM](https://github.com/ShuhongLL/SGS-SLAM), and by using or modifying this work further, you agree to adhere to their terms of usage and include the license file. Many thanks to the SGS-SLAM team for their excellent contributions and for helping make this work possible.

