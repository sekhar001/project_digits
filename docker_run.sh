#!/bin/bash

echo "Printing the model directory before runnig docker"

ls -lh models

docker build -t digits:v1 -f docker/DockerFile .

docker run -it -v /mnt/c/Users/Niladri/mlops/project_digits/models:/digits/models digits:v1

echo "Printing the model directory post running docker"

ls -lh models