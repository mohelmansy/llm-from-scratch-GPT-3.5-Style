
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
WORKDIR /repo
COPY requirements.txt /repo/requirements.txt
RUN pip install -r requirements.txt
COPY . /repo
