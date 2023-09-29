#!/bin/bash
#export LD_PRELOAD="$CUDADIR/lib64/libcudart.so $CUDADIR/lib64/libcublas.so"
. ./scripts/values.sh

unset OMP_NUM_THREADS
python Launcher.py $xmin $xmax $ymin $maxiter prof own methods all sizes 256 512 1024 onlytimes                                                                                                                                                                                                                                                                                                                                                                                                              #| column -t -s ';'
