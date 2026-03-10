# Python 3.12.9 on Debian Bookworm (slim)
FROM python:3.12.9-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps commonly needed by gym/gymnasium rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git curl build-essential pkg-config \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Keep setuptools compatible with older gym
RUN python -m pip install --upgrade pip "setuptools<72" wheel packaging

# Install PyTorch (CUDA 12.1 wheels). Use --gpus all at runtime.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Install your specified NumPy:
# prefer the provided wheel; otherwise fall back to same version from PyPI
RUN python -m pip install "numpy==1.26.4"

# Your pinned packages
RUN python -m pip install \
    "gym==0.23.1" \
    "gym-notices==0.0.8" \
    "gymnasium==1.1.1" \
    "gymnasium-robotics==1.3.1" \
    "tyro==0.9.20" \
    "PyYAML==6.0.2" \
    "stable_baselines3==2.6.0" \
    "tensorboard==2.19.0"

# Extra system libs for headless MuJoCo rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgles2 libopengl0 libx11-6 libxext6 libxrender1 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo + helpers (pin a recent version)
RUN python -m pip install "mujoco>=3.1.0" "glfw>=2.7.0" imageio
ENV MUJOCO_GL=egl

# Add project code into the image
COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["python", "-V"]
