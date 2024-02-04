#!/usr/bin/env bash

docker build -f utils/bazel/docker/Dockerfile \
             -t torch-mlir:dev \
             --build-arg GROUP=$(id -gn) \
             --build-arg GID=$(id -g) \
             --build-arg USER=$(id -un) \
             --build-arg UID=$(id -u) \
             .

docker run -it \
           -v "$(pwd)":"/opt/src/torch-mlir" \
           -v "${HOME}/.cache/bazel":"${HOME}/.cache/bazel" \
           torch-mlir:dev
