# ─────────────────────────────────────────────────────────────────
#  InstructTune-LLM  |  Docker image
#  Builds a GPU-ready training container with CUDA 11.8 + Python 3.10
# ─────────────────────────────────────────────────────────────────

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install -r requirements.txt \
    && python3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cu118

# Copy project source
COPY . .

# Default: run training with the standard config
# Override with:  docker run ... python3.10 -m src.inference ...
ENTRYPOINT ["python3.10", "-m", "src.train_lora", "--config", "configs/training_config.yaml"]
