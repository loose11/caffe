#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/uni_augsburg_swim/solver.prototxt -gpu 2
