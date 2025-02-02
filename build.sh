cd ./third_party/Sophus

# Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Photo-SLAM
cd ../../..
mkdir build
cd build
# cmake .. # add Torch_DIR and/or OpenCV_DIR definitions if needed, example:
cmake .. -DTorch_DIR=/workspace/libs/libtorch/share/cmake/Torch -DOpenCV_DIR=/workspace/libs/opencv4/lib/cmake/opencv4
make -j4
