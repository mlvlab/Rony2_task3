#!/bin/bash
# Download weights models

python pretrained_models/download_weights.py && \
	python detector/yolo/data/download_weights.py && \
	python detector/tracker/data/download_weights.py
