docker build --build-arg SGL_BRANCH=v0.5.15.post1 --build-arg GPU_ARCH=gfx950-rocm720 --build-arg ENABLE_MORI=1 -t sabreshao/sglang:v0.5.15.post1-rocm720-mi35x -f rocm.Dockerfile .
