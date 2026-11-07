# GPU engine timing image: singlet --engine gpu built from a git ref of the public repository, with Clipper's recipe (build.sh), plus the
# bundle's fetch.sh and run.sh. Clipper builds in docker://nvidia/cuda:12.4.1-devel-ubuntu22.04 (as an Apptainer image); so does this.
#
#   docker build -t singlet-gpu-bench --build-arg SINGLET_REF=gpu/frontier bench/gpu/portable
#   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /data:/data singlet-gpu-bench fetch.sh
#   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /data:/data singlet-gpu-bench run.sh
#
# --ipc=host matters: the engine passes reads between processes through /dev/shm, and Docker's default /dev/shm is 64 MB.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG SINGLET_REPO=https://github.com/Singlet-Bio/singlet
ARG SINGLET_REF=gpu/frontier
ARG CUDA_ARCHS=80;90
ARG GCC_SPEC=
ARG JOBS=

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends git curl ca-certificates time procps python3 util-linux pciutils \
 && rm -rf /var/lib/apt/lists/*

COPY build.sh fetch.sh run.sh README.md /opt/singlet/bundle/
RUN chmod +x /opt/singlet/bundle/*.sh \
 && SINGLET_REPO="$SINGLET_REPO" SINGLET_REF="$SINGLET_REF" CUDA_ARCHS="$CUDA_ARCHS" GCC_SPEC="$GCC_SPEC" JOBS="${JOBS:-$(nproc)}" \
    PREFIX=/opt/singlet bash /opt/singlet/bundle/build.sh

ENV PATH=/opt/singlet/bundle:/opt/singlet/bin:/opt/singlet/env/bin:$PATH \
    SINGLET_BIN=/opt/singlet/bin/singlet \
    SINGLET_SRC=/opt/singlet/src \
    SINGLET_WHITELIST_DIR=/opt/singlet/whitelists \
    DATA=/data
WORKDIR /data
CMD ["bash"]
