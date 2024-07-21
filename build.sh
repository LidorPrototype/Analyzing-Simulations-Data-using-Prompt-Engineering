#!/bin/bash

# create docker volume for model data
docker volume create model-data
# build docker image
docker build -t analyzing-simulations:latest .