FROM nvidia/cuda:10.2-devel-ubuntu18.04 

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y wget git vim nano bzip2 ca-certificates \
	libglib2.0-0 libxext6 libsm6 libxrender1 \
    	git mercurial subversion && apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN pip install --upgrade pip && \
    pip install --no-cache-dir nibabel pydicom pillow med2image matplotlib PyQt5 cython && \
    conda update -n base -c defaults conda


RUN git clone https://github.com/OpenPIV/openpiv-python.git && \
    cd openpiv-python && \
    git checkout tags/0.21.8b &&\
    python setup.py install && \
    cd .. && rm -rf openpiv-python

RUN pip install opencv-python 

