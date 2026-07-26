docker build \
  --build-arg SGL_BRANCH=refactor/amd-glm5x-cp-eagle-20260723 --build-arg SGL_REPO=https://github.com/AFDEAPAC/sglang --build-arg GPU_ARCH=gfx950-rocm720 --build-arg ENABLE_MORI=1 -t sabreshao/sglang:cp-eagle-0725 -f rocm.Dockerfile .
