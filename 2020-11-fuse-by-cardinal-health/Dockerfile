# OS/ARCH: linux/amd64
ARG ROOT_CONTAINER=ubuntu:focal-20200925@sha256:2e70e9c81838224b5311970dbf7ed16802fbfe19e7a70b3cbfa3d7522aa285b4
FROM $ROOT_CONTAINER

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Make directory for project
RUN mkdir -p /cardinal

# Creating the environment
RUN apt-get update 
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade pip setuptools

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Install requirements first
COPY ./requirements.txt /cardinal/requirements.txt
RUN python3 -m pip install -r /cardinal/requirements.txt

# Jupyter + DataScience

WORKDIR /cardinal
EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "jupyter notebook --notebook-dir=/cardinal --NotebookApp.token='' --NotebookApp.password='' --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
