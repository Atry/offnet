#!/usr/bin/env bash
nvidia-docker run \
    --volume /etc/passwd:/etc/passwd:ro \
    --volume /etc/shadow:/etc/shadow:ro \
    --volume "$HOME:$HOME" \
    --volume "$PWD:/mnt/project-root" \
    --workdir /mnt/project-root \
    --tty --interactive \
    --init \
    popatry/pipenv-cuda:cuda-9.0-runtime-ubuntu16.04-pip3-pipenv-socat \
    su "$(whoami)"
