FROM mcr.microsoft.com/mirror/nvcr/nvidia/pytorch:22.04-py3
RUN apt update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt install -y software-properties-common build-essential
RUN apt install -y cmake curl vim libgl1 git
RUN apt install -y default-jre
RUN apt clean
WORKDIR /opt/verbenvs

# User must mount this data point
ENV VERBENVS_DATA=/opt/verbenvs/data
ENV ALFWORLD_DATA=$VERBENVS_DATA/alfworld/data
VOLUME $VERBENVS_DATA

COPY verbenvs verbenvs
COPY setup.py README.md .
RUN pip install .[touchdown,scienceworld,alfworld]

# Hack to update some libraries
RUN sed -i 's/from collections import defaultdict, Mapping/from collections.abc import Mapping\nfrom collections import defaultdict/' /opt/conda/lib/python3.8/site-packages/tatsu/grammars.py

COPY tests tests

RUN pip install lightning[extra]
RUN pip install tqdm hydra-core jsonargparse ujson
RUN pip install transformers
RUN pip install opencv-python==4.5.5.64

COPY verbenvs verbenvs

WORKDIR /opt/work
