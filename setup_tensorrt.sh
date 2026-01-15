#!/bin/bash

os="ubuntu2404"
tag="10.14.1-cuda-13.0"
wget https://developer.download.nvidia.com/compute/tensorrt/10.14.1/local_installers/nv-tensorrt-local-repo-ubuntu2404-10.14.1-cuda-13.0_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt
