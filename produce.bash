#!/bin/bash

n="16"		         #mpi size
L="16"                   #size of lattice
range="1000"		 #the number of points bet initial and fin values
jacks="5"                #the number of data to use jackknife method
therm_run="200000"
NUPDATES="1000000"
initvalue="1.95"		 #initial value
delta="1.95"		 #Delta value
gap="0.1"			 #it determine final value by initvalue + gap
first_gpu="0"
last_gpu="7"

time mpirun --oversubscribe -np $n ./produce_mpi $L $range $jacks $therm_run $NUPDATES $initvalue $gap &
#time mpirun --oversubscribe -np $n ./produce_mpi $L $delta $NUPDATES &
#time ./produce_gpu $L $first_gpu $last_gpu $range $jacks $therm_run $NUPDATES $initvalue $gap &
