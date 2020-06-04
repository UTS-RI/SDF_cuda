# SDF_cuda
SDF computation on CUDA (look at my [other implementation on CPU](https://github.com/rFalque/voxelization_and_sdf) if you want some illustration -- as a side note kdtree are really efficient ... faster than this code but that method is not exact for meshes with very large triangles).

## What is it?
Example of distance function computation using CUDA. The code is pretty minimalistic and the main files to look at are:
* app/test_sdf.cpp
* includes/compute_sdf.h
* src/compute_sdf.cu

The specifiers `__global__`, `__device__`, and `__host__` are defining if the function will be compiled on the host (CPU) or on the device (GPU).

## Installation
As dependencies, you need [cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile), [cmake](https://cgold.readthedocs.io/en/latest/first-step/installation.html), and [eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) to be installed.

To compile the code and run an example (in ubuntu):
```bash
mkdir build
cd build
cmake ..
make
./test_sdf
```

This will compute a regular distance function grid of size [58, 35, 100] from of mesh with 100k faces, 50k vertices. All together 58 * 35 * 100 * 100.000 distance computations are performed (20.3 billion inference), hence the importance of GPU parrallelization.

The output of the demo will be saved in `data/image_stack/` in the form of images.

## TODO
* There is a weird bug with the sign of the distance function which is flipped is some rare occasions. In the demo, it happens at the tensor value [25, 0, 19] on Lucy100k.
* Time each step of the program, I suspect most of the time is spent in transfering the data to the GPU memory
* Add visualization

## Further CUDA doc to read
* [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/): I can not recommend this enough, this is a very simple and straighforward introduction to GPU programming with CUDA
* [CUDA syntax](http://www.icl.utk.edu/~mgates3/docs/cuda.html): CUDA cheat sheet kind of style
* [Actual CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): More torough documentation
* [andyzeng/tsdf-fusion](https://github.com/andyzeng/tsdf-fusion): Other related git repo
