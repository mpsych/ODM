# Use an official Tensorflow runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

# Set the working directory in the container to /app
WORKDIR /app

# Set environment varibles
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_AUTO_UPDATE_CONDA=false

# Add missing GPG key for CUDA repo
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install wget and git
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cuda-drivers \
    wget \
    git \
    vim \
    ack \
    tree \
    build-essential \
    python3-setuptools \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm -f Miniconda3-py39_23.3.1-0-Linux-x86_64.sh

RUN conda --version

# clone repo
RUN git clone https://github.com/mpsych/ODM.git

# change into repo and install Python requirements
WORKDIR /app/ODM
RUN conda env create -f environment.yml


# activate conda env
SHELL ["/bin/bash", "-c"]
RUN echo "source activate ODM" > ~/.bashrc \
    && echo 'export LD_LIBRARY_PATH="/root/miniconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
ENV PATH /opt/conda/envs/ODM/bin:$PATH

CMD [ "/bin/bash" ]
