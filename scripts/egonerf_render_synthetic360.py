import os
import subprocess

# synthetic360
dataset_dir = "../data/synthetic360"
scene_list_path = "./egonerf_scene_list_synthetic360.txt"
exp_loops = [10]
test_iters = [32000]

training_cfg_path = "../cfg/lonlat/egonerf_lonlat_synthetic360.yaml"
result_root = "../results/lonlat/EgoNeRF/synthetic360"

scene_list = []
with open(scene_list_path, 'r') as fin:
    for line in fin:
        scene_list.append(line.strip())

for loop in exp_loops:
    result_root_loop = result_root + "_" + loop.__str__()
    for scene in scene_list:
        scene_root = os.path.join(dataset_dir, scene)
        result_dir = os.path.join(result_root_loop, scene)

        # subprocess.run(["../bin/train_egonerf_ricoh360",
        #                 training_cfg_path,
        #                 scene_root,
        #                 result_dir,
        #                 "no_viewer"])

        for test_iter in test_iters:
            subprocess.run(["../bin/test_openmvg_lonlat",
                            training_cfg_path,
                            scene_root,
                            os.path.join(result_dir, test_iter.__str__(), "ply", "point_cloud", "iteration_" + test_iter.__str__(), "point_cloud.ply"),
                            os.path.join(result_dir, test_iter.__str__() + "_test")])