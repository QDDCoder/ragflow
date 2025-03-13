# 基础镜像
FROM python:3.10-slim

# 删除原有的 apt 源配置
RUN rm -rf /etc/apt/sources.list.d/*

# 设置新的 apt 源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free" > /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list

# 设置环境变量
ENV RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
ENV RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup

# 更新 apt 源并安装必要的软件包
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    pkg-config \ 
    libicu-dev \
    build-essential \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Rust 开发环境
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 创建 /app 目录
RUN mkdir -p /app


# 设置工作目录
WORKDIR /app

# 后续可以添加更多的构建步骤，例如复制文件、安装依赖等
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install uv 
    
ENV UV_INDEX=https://mirrors.aliyun.com/pypi/simple
