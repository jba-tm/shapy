# Dockerfile
# pull official base image
FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment \

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version


RUN mkdir /shapy
# set work directory
WORKDIR /shapy

# create and activate virtual environment

RUN python3.8 -m venv .venv/shapy


RUN /shapy/.venv/shapy/bin/python3 -m pip install pip setuptools wheel --upgrade

# ml data
COPY ./data /shapy/data


# copy and install pip requirements
COPY ./requirements.txt /shapy/requirements.txt

RUN /shapy/.venv/shapy/bin/pip3 install -r /shapy/requirements.txt

COPY ./attributes /shapy/attributes
ENV PYTHONPATH "${PYTHONPATH}:/shapy/attributes"
RUN cd /shapy/attributes
RUN /shapy/.venv/shapy/bin/python setup.py install
RUN cd /shapy

COPY ./mesh-mesh-intersection /shapy/mesh-mesh-intersection
RUN cd /shapy/mesh-mesh-intersection
ENV CUDA_SAMPLES_INC "/shapy/mesh-mesh-intersection/include"
RUN /shapy/.venv/shapy/bin/pip3 install -r /shapy/mesh-mesh-intersection/requirements.txt
RUN /shapy/.venv/shapy/bin/python setup.py install
RUN cd /shapy

# copy project files
COPY ./samples /shapy/samples
COPY ./regressor /shapy/regressor
COPY ./scripts /shapy/scripts
COPY ./static /shapy/static
COPY ./app /shapy/app


RUN chmod +x ./scripts/start.sh
RUN chmod +x ./scripts/start-reload.sh

CMD ["./scripts/start.sh"]