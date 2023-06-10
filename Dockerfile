# Use an official Python runtime as a parent image
FROM ubuntu:latest

# Set the working directory in the container to /app
WORKDIR /app

# Set environment varibles
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    python3 \
    python3-venv \
    python3-setuptools \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# install anaconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm -f Miniconda3-py39_23.3.1-0-Linux-x86_64.sh

RUN conda --version

# clone repo
RUN git clone https://github.com/mpsych/ODM.git

# change into repo and create conda env from yml file
WORKDIR /app/ODM
RUN conda env create -f environment.yml

# activate conda env
SHELL ["/bin/bash", "-c"]
RUN echo "source activate ODM" > ~/.bashrc
ENV PATH /opt/conda/envs/ODM/bin:$PATH

# finish by opening a bash shell for interactive use
CMD [ "/bin/bash" ]