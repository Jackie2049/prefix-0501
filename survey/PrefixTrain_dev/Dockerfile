FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


# 安装 Python 及常用工具
RUN apt-get update && \
    apt-get install -y python3 pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools 

# 验证安装
CMD ["python3", "--version"]
RUN pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch/
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt

COPY external/apex /workspace/apex
WORKDIR /workspace/apex
RUN python3 -m pip install --upgrade pip && pip3 install -r requirements.txt
RUN pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./

# COPY external/alpa /workspace/alpa
# WORKDIR /workspace/alpa
# RUN pip uninstall -y cupy-cuda115
# RUN pip install -r requirements.txt

# RUN pip install jaxlib==0.3.5+cuda113.cudnn820 -f https://github.com/alpa-projects/alpa/releases/download/v0.1.5/jaxlib-0.3.5%2Bcuda113.cudnn820-cp38-none-manylinux2010_x86_64.whl

RUN apt-get update
RUN apt-get -y install coinor-cbc

WORKDIR /workspace