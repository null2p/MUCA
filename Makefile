
MPICC=mpic++
EASY_FLAGS = -I /home/null/sprng5/include/ -L /home/null/sprng5/lib/ -lsprng -O3 -std=c++11
TIGHT_FLAGS = -O2 -pedantic -Wall -Wextra -std=c++11 -I /home/null/sprng5/include -L /home/null/sprng5/lib -lsprng
123_FLAGS = -O2 -std=c++11 -I ./Random123/include/
M_FLAGS = -O2 -std=c++11 -I ./Random123/include/
PRODUCE_FLAGS = -O2 -std=c++11 -I ./Random123/include/

MPI_FLAGS = -I /usr/include/openmpi-x86_64/ -L /usr/lib64/openmpi/lib -lmpi

ifeq ($(wildcard /opt/cuda/bin/nvcc),)
  NVCC=nvcc
else
  NVCC=/opt/cuda/bin/nvcc
endif
GPU_ARCHS=-arch=sm_35 -rdc=true -I./Random123/include/ -lineinfo
GPU_FLAGS=-std=c++11 -Xcompiler -Wall,-Wno-unused-function,-O3

all: easy tight 123 produce gpu
tight: muca_test_done
M: muca_mag
pr2 : produce2 
pr_mpi: produce_mpi
pr_gpu: produce_gpu
wolff : produce_wolff
device : DeviceQuery
gpu2 : muca_delta2
multi : multigpu_weight
kld : kld
chi : chi_max
chi_cpu : chi_max_cpu

muca_mag: muca_mag.cpp
	$(MPICC) muca_mag.cpp $(M_FLAGS) -o $@

produce2: produce2.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v produce2.cu -o $@

produce_mpi: produce.cpp
	$(MPICC) produce.cpp $(PRODUCE_FLAGS) -o $@

produce_gpu: produce2_multigpu.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v produce2_multigpu.cu -o $@

produce_wolff: wolff.cpp
	$(MPICC) $(PRODUCE_FLAGS) wolff.cpp -o $@

DeviceQuery: DeviceQuery.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v DeviceQuery.cu -o $@

muca_delta2: muca_delta2.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v muca_delta2.cu -o $@

multigpu_weight: muca_delta_multigpu.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v muca_delta_multigpu.cu -o $@

kld: kld.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v kld.cu -o $@

chi_max: chi_max.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v chi_max.cu -o $@

chi_max_cpu: chi_max.cpp
	$(MPICC) chi_max.cpp $(PRODUCE_FLAGS) -o $@



clean:
	rm -f muca_mag
	rm -f produce2
	rm -f produce_mpi
	rm -f DeviceQuery
	rm -f muca_delta2
	rm -f kld
	rm -f chi_max
	rm -f chi_cpu
