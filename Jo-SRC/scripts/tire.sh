#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=1
python tire.py --config config/tire.cfg --gpu ${GPU}
