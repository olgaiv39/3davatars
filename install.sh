#!/bin/bash
apt install libgl1-mesa-glx
pip install opencv-python numpy pandas matplotlib scikit-image Pillow
cd /usr/local/lib/python3.6/dist-packages/tensorflow_core
ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
cd 
cd 3davatars
echo "compiling rasterizer"
TF_INC=/usr/local/lib/python3.6/dist-packages/tensorflow_core/include
TF_LIB=/usr/local/lib/python3.6/dist-packages/tensorflow_core
# you might need the following to successfully compile the third-party library
tf_mesh_renderer_path=$(pwd)/third_party/kernels
g++ -std=c++11 \
    -shared $tf_mesh_renderer_path/rasterize_triangles_grad.cc $tf_mesh_renderer_path/rasterize_triangles_op.cc $tf_mesh_renderer_path/rasterize_triangles_impl.cc $tf_mesh_renderer_path/rasterize_triangles_impl.h \
    -o $tf_mesh_renderer_path/rasterize_triangles_kernel.so -fPIC  -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I$TF_INC -L$TF_LIB -ltensorflow_framework -O2

if [ "$?" -ne 0 ]; then echo "compile rasterizer failed"; exit 1; fi
