#!/bin/bash

openMVG_bin_dir=/opt/openMVG_Build/install/bin
dataset_dir=../data/360Roam_dataset

python3 ./360roam_to_openmvg_train.py --openMVG_bin_dir $openMVG_bin_dir --dataset_dir $dataset_dir --scene_list ./360roam_scene_list.txt --img_width 1520 --img_height 760
python3 ./360roam_to_openmvg_test.py --openMVG_bin_dir $openMVG_bin_dir --dataset_dir $dataset_dir --scene_list ./360roam_scene_list.txt --img_width 1520 --img_height 760
