#!/bin/bash

DIRECTORY=/home/hdd/remix-factory
MODEL=$DIRECTORY/neurons

./remix-factory --create --model $MODEL --thread 4
tar cf $DIRECTORY/initial-model.lzma --lzma $MODEL
./remix-factory --train --model $MODEL --thread 4 --dataset corpus/ --batchSize 10 --learningRate 1
tar cf $DIRECTORY/first-training-pass-model.lzma --lzma $MODEL
