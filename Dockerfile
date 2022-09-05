# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.9    (apt)
# jupyter       latest (pip)
# pytorch       latest (pip)
# jupyterlab    latest (pip)
# ==================================================================

FROM ubuntu:20.04
ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        cmake \
        xdot \
        nodejs \
	python-is-python3 \
	python3-pip \
        && \

# ==================================================================
# python
# ------------------------------------------------------------------

    $PIP_INSTALL \
        numpy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        numba \
        && \

# ==================================================================
# jupyter
# ------------------------------------------------------------------

    $PIP_INSTALL \
        jupyter \
        jupyterlab \
        && \
        
# ==================================================================
# DL libs
# ------------------------------------------------------------------

    $PIP_INSTALL \
        tensorflow \
        keras \
        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

COPY requirements.txt /
RUN cd / \
  && python -m pip --default-timeout=300 --no-cache-dir install --upgrade \
    --requirement requirements.txt  \
  && rm /requirements.txt
