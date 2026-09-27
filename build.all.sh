#!/bin/bash
set -euo pipefail

rm -rf dist/*
./build.sh

rm -rf pynvr/frontend_dist/*
./build.ui.sh

./build.docker.sh
