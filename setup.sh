#!/bin/bash
conda create -n segbase python=3.8
conda activate segbase

# with gpu
#conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch

pip install -r requirements.txt

