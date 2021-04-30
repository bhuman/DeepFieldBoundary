#!/bin/bash

if [ "$1" = "Update" ]; then
  docker build -t fieldboundary Docker
elif [ "$1" = "Launch" ]; then
  docker run -v ${2:-~}:/workspace --net=host --gpus all -u $(id -u):$(id -g) -it --shm-size='256m' --rm fieldboundary
fi
