#!/bin/bash

n="16"			#mpi size
L="72"                   #size of lattice
starting_M="1"        #M is the spin flip trials for each thread
jacks="1"                #the number of data to use jackknife method
recursive_update="0"     #if '0', trivial update is activated
rand_config="1"          #if '0', the initial spin configuration is unifomly distributed, else, initial spin configuration is random.
continuing="0"           #init weight from extrapolated weight
starting_gpu="0"
last_gpu="0"

#time nohup mpirun -np $n ./weight_produce $L $starting_M $jacks $continuing >> result00.txt &
#time nohup ./multigpu_weight $L $starting_gpu $last_gpu $jacks $starting_M $recursive_update $continuing >> result_weight_gpu.txt &
time ./multigpu_weight $L $starting_gpu $last_gpu $jacks $starting_M $recursive_update $continuing 
