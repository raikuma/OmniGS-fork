import os
import sys
import numpy as np
import json
import argparse
import subprocess
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser(description="")
parser.add_argument("--openMVG_bin_dir", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--scene_list", type=str, default=None)
args = parser.parse_args(sys.argv[1:])

openMVG_bin_dir = args.openMVG_bin_dir
dataset_dir = args.dataset_dir
scene_list_path = args.scene_list
ptr_wrapper_id = 2147483649
polymorphic_id = 1073741824

num_cpu_threads = 12

scene_list = []
with open(scene_list_path, 'r') as fin:
    for line in fin:
        scene_list.append(line.strip())

for scene in scene_list:
    openMVG_output_dir = os.path.join(dataset_dir, scene, "openMVG")
    os.makedirs(openMVG_output_dir, exist_ok=True)
    openMVG_init_path = os.path.join(openMVG_output_dir, "data_openmvg_test.json")

    # Gather scene information
    test_indices = []
    with open(os.path.join(dataset_dir, scene, "test.txt"), 'r') as test_fin:
        for line in test_fin:
            test_indices.append(line.strip())

    img_dir = os.path.join(dataset_dir, scene, "imgs")
    colmap_dir = os.path.join(dataset_dir, scene, "output_dir", "colmap")

    camera_file = os.path.join(colmap_dir, "cameras.txt")
    frames_file = os.path.join(colmap_dir, "images.txt")

    # Camera information
    with open(camera_file, 'r') as camera_fin:
        for line in camera_fin:
            if line[0] == "0" or line[0] == "1":
                img_width = int(line.split(" ")[2])
                img_height = int(line.split(" ")[3])


    # Frames information
    json_views = []
    json_extrs = []
    frame_idx = 0
    with open(frames_file, 'r') as frames_fin:
        is_pose_line = True
        for line in frames_fin:
            if line.strip() and line[0] != "#":
                if is_pose_line or True:
                    is_pose_line = False
                    line_content = line.split(" ")
                    image_filename = line_content[9].strip()
                    image_id = os.path.splitext(image_filename)[0]
                    if not image_id in test_indices:
                        continue
                    qw = float(line_content[1])
                    qx = float(line_content[2])
                    qy = float(line_content[3])
                    qz = float(line_content[4])
                    tx = float(line_content[5])
                    ty = float(line_content[6])
                    tz = float(line_content[7])

                    json_view = {
                                "key" : frame_idx,
                                "value": {
                                    "polymorphic_id": polymorphic_id,
                                    "ptr_wrapper": {
                                        "id": (ptr_wrapper_id + frame_idx),
                                        "data": {
                                            "local_path": "",
                                            "filename" : image_filename,
                                            "width": img_width,
                                            "height": img_height,
                                            "id_view": frame_idx,
                                            "id_intrinsic": 0,
                                            "id_pose": frame_idx
                                        }
                                    }
                                }
                            }

                    tcw = np.array([tx, ty, tz])
                    Rcw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    Rwc = np.linalg.inv(Rcw)
                    twc = -Rwc @ tcw
                    ccw = twc
                
                    rotation_matrix = Rcw.tolist()
                    center = ccw.tolist()

                    json_extr = {
                        "key" : frame_idx,
                        "value": {
                            "rotation": rotation_matrix,
                            "center": center
                        }
                    }
                    json_views.append(json_view)
                    json_extrs.append(json_extr)
                    frame_idx = frame_idx + 1
                else:
                    is_pose_line = True

    json_intrs = []
    json_intrs.append({
        "key" : 0,
        "value": {
                "polymorphic_id": ptr_wrapper_id,
                "polymorphic_name": "spherical",
                "ptr_wrapper": {
                    "id": (ptr_wrapper_id + frame_idx),
                    "data": {
                        "value0": {
                            "width": img_width,
                            "height": img_height
                        }
                    }
                }
            }
        })

    json_content = {
        "sfm_data_version": "0.3",
        "root_path" : img_dir,
        "views" : json_views,
        "intrinsics": json_intrs,
        "extrinsics": json_extrs,
        "structure": [],
        "control_points": []
    }

    with open(openMVG_init_path, 'w') as fout:
        json.dump(json_content, fout)
