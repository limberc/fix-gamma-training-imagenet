#!/usr/bin/env bash
mkdir train
ILSVRC2012_img_train.tar train/
cd train/
tar -xvf ILSVRC2012_img_train.tar
rm ILSVRC2012_ img_train.tar