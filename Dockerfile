# 1. 基础镜像
FROM python:3.10-slim

# 2. 安装 git 和系统编译基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 设置默认工作目录
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# 4. 从 GitHub 仓库拉取源码
ARG REPO_URL="https://github.com/ovedick2026/babyAgent.git"
ARG BRANCH="main"

RUN git clone --depth=1 -b ${BRANCH} ${REPO_URL} . && rm -rf .git

# 5. 安装依赖
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 6. 放开目录读写权限
RUN chmod -R 777 /app

# 7. 暴露 HF Spaces 默认端口
EXPOSE 7860

# 8. 启动
CMD ["python", "app.py"]
