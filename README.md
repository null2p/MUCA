# MUCA
multicanonical method to the 2d Blume-Capel model


How to Compile
-----

Estimating Multicanonical Weight
```sh
CPU (C++)  : make muca_mag
GPU (cuda) : make multigpu_weight
```

Producing Order Parameters
```sh
CPU (C++)  : make produce_mpi
GPU (cuda) : make produce_gpu
```

How to run
------
Modify Weight Bash file

```sh
#!/bin/bash

L="72"                   #size of lattice
starting_M="1"           #M is the spin flip trials for each thread
jacks="1"                #the number of data to use jackknife method
recursive_update="0"     #if '0', trivial update is activated
rand_config="1"          #if '0', the initial spin configuration is unifomly distributed, else, initial spin configuration is random.
continuing="0"           #init weight from extrapolated weight

#CPU
n="16"                   #mpi size
#time nohup mpirun -np $n ./weight_produce $L $starting_M $jacks $continuing >> result00.txt &


#GPU
starting_gpu="0"
last_gpu="0"
time ./multigpu_weight $L $starting_gpu $last_gpu $jacks $starting_M $recursive_update $continuing
```

Estimating Multicanonical Weight
```
bash weight.bash
```
Producing Order Parameters
```
bash produce.bash
```

