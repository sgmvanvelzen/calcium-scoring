FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV CUDA_ROOT /usr/local/cuda/bin/
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
ENV MKL_THREADING_LAYER GNU

RUN ldconfig

# Install basic utilities
RUN apt-get update --fix-missing && \
  apt-get install -y --no-install-recommends wget bzip2 ca-certificates git && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

#/opt/conda/bin/conda install python=3.6 && \


# Install dependencies (python packages)
RUN conda install conda=4.5.11=py36_0 python=3.6.6  mkl-service=1.1.2  jupyter=1.0.0
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Setup user and directory for our software
RUN useradd -m -d /home/user -s /bin/bash user && (echo user ; echo user) | passwd user
USER user

# Put juypter notebook and related files into user directory
COPY --chown=user *.ipynb /home/user/
COPY --chown=user static_figures/ /home/user/static_figures/
COPY --chown=user src/ /home/user/src/
COPY --chown=user theanorc.txt /home/user/.theanorc

# Define entry point
WORKDIR /home/user
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
