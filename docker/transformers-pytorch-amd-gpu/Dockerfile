FROM rocm/pytorch:rocm6.4_ubuntu24.04_py3.12_pytorch_release_2.6.0
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
ENV MIOPEN_FIND_MODE=1
ENV AMD_DIRECT_DISPATCH=0

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-dev python3-pip python3-dev ffmpeg git-lfs && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN rm -rf $(pip show numpy | grep Location: | awk '{print $2}')/numpy*
RUN python3 -m pip install --no-cache-dir --upgrade pip numpy

RUN python3 -m pip uninstall -y torch torchvision torchaudio
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

RUN python3 -m pip install --no-cache-dir --upgrade importlib-metadata setuptools ninja git+https://github.com/facebookresearch/detectron2.git pytesseract "itsdangerous<2.1.0"

RUN git clone --recurse https://github.com/ROCm/bitsandbytes && \
    cd bitsandbytes && \
    git checkout rocm_enabled_multi_backend && \
    pip install -r requirements-dev.txt && \
    cmake -DCOMPUTE_BACKEND=hip -S .  && \
    make && \
    pip install .

ARG REF=main
WORKDIR /

# Invalidate docker cache from here if new commit is available.
#ADD https://api.github.com/repos/huggingface/transformers/git/refs/heads/main version.json
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

RUN python3 -m pip uninstall -y tensorflow flax

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Remove nvml and nvidia-ml-py as it is not compatible with ROCm. apex is not tested on NVIDIA either.
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y
RUN pip install phonemizer pyctcdecode
