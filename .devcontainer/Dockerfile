# 基础镜像
FROM ubuntu:22.04

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive

# 安装 ca-certificates 包
RUN apt-get update && apt-get install -y ca-certificates


# 设置环境变量
ENV RUSTUP_DIST_SERVER=https://mirrors.aliyun.com/rustup
ENV RUSTUP_UPDATE_ROOT=https://mirrors.aliyun.com/rustup/rustup
ENV HF_ENDPOINT=https://hf-mirror.com

# 更新 apt 源并安装必要的软件包
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    pkg-config \ 
    libicu-dev \
    build-essential \
    zlib1g-dev \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-dev \
    tzdata \
    iputils-ping \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建Python软链接
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python download_models.py

# 安装 Rust 开发环境
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 创建 /app 目录
RUN mkdir -p /app

# 设置工作目录
WORKDIR /app

# 配置并安装 Node.js 20.x 稳定版
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs

# 配置 npm 阿里源
RUN npm config set registry https://registry.npmmirror.com

# 配置pip镜像并安装uvicorn
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install uv && \
    pip install libjemalloc-dev

ENV UV_INDEX=https://mirrors.aliyun.com/pypi/simple

