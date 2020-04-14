FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         openssl \
         libssl-dev \
         libjpeg-dev \
         zlib1g-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*
RUN  curl -L -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ./Miniconda3-latest-Linux-x86_64.sh && \
     ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
     rm ./Miniconda3-latest-Linux-x86_64.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing ipykernel && \
     /opt/conda/bin/conda install -y -c pytorch matplotlib pandas scikit-learn nodejs dask pydot seaborn && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
# This must be done before pip so that requirements.txt is available

RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && git fetch && git fetch --tags && git checkout v0.3.0 \
 && python setup.py install
RUN apt-get update && apt-get install -y --no-install-recommends \
         software-properties-common \
         graphviz && \
     rm -rf /var/lib/apt/lists/*
RUN conda install python-graphviz
#RUN curl -O https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
#RUN tar -xzf pycuda-2019.1.2.tar.gz && rm pycuda-2019.1.2.tar.gz
#RUN cd pycuda-2019.1.2 && python configure.py --cuda-root=/usr/local/cuda && make install

RUN pip install jupyterlab

WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN mkdir notebooks
RUN chmod -R a+w notebooks
# RUN git clone https://github.com/pytorch/cogsmachine /workspace/examples
CMD jupyter-lab --ip=0.0.0.0 --port=8888 --allow-root
