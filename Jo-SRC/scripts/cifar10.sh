#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=1
python cifar10.py --config config/cifar10.cfg --gpu ${GPU}
