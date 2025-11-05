# Modern Dockerfile for PyTorch 2.1+ with CUDA support
# Compatible with AWS, Google Cloud, and TAMU HPRC

# Use Ubuntu 22.04 (LTS) - widely supported on cloud platforms
# Includes Python 3.10 by default
FROM ubuntu:22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA 11.8 support
# This version is compatible with most cloud GPU instances
# For CUDA 12.1, change cu118 to cu121 in the index-url
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining Python dependencies
RUN pip install -r requirements.txt

# Set Python path to include /app directory
# This replicates the original sgan.pth hack
RUN python3 -c "import site; print('/app')" > $(python3 -c "import site; print(site.getsitepackages()[0])")/sgan.pth || \
    echo "/app" > /usr/local/lib/python3/dist-packages/sgan.pth

# Set Python as default python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Set the default command to open a shell
CMD ["/bin/bash"]
