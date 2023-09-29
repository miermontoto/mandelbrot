#!/bin/bash
. ./scripts/values.sh

params="${xmin} ${xmax} ${ymin} ${maxiter} onlytimes py own sizes 256 512"

# Secuencial
export OMP_NUM_THREADS=1
python Launcher.py $params

echo

# Paralelo
unset OMP_NUM_THREADS
python Launcher.py $params noheader -py prof methods normal collapse tasks schedule_guided
