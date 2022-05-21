#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <getopt.h>
#include <algorithm>
#include <errno.h>

#include "Random123/philox.h"
#include "Random123/examples/uniform.hpp"

#define SEED 100

typedef r123::Philox4x32_R<7> RNG;
typedef unsigned long long my_uint64;
double my_uint64_max = pow(2.0,64)-1;

// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256
#define WORKER (blockIdx.x * blockDim.x + threadIdx.x)

// launch bounds depend on compute capability
#if __CUDA_ARCH__ >= 300
    #define MY_KERNEL_MIN_BLOCKS   2048/WORKERS_PER_BLOCK
#elif __CUDA_ARCH__ >= 200
    #define MY_KERNEL_MIN_BLOCKS   1536/WORKERS_PER_BLOCK
#else
    #define MY_KERNEL_MIN_BLOCKS   0
#endif

// random access to textures in global memory is faster due to caching
// this texture holds the logarithmic weights for MUCA
//texture<double, 1, cudaReadModeElementType> t_log_weight;

__constant__ unsigned d_L;
__constant__ unsigned d_N;
__constant__ unsigned d_NUM_WORKERS;

using namespace std;

__device__ __forceinline__ int inter_energy(unsigned i, int8_t* lattice);
__device__ int total_energy(int8_t* lattice);



void weight_shift(double beta,double delta,unsigned N,double* h_log_weight);
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
mcs(int8_t* lattice,my_uint64* d_hist,unsigned* index,unsigned* naver,double* d_log_weight, double beta, double delta,int* d_energy, int* d_Delta_E,int* d_Mag, unsigned iter, unsigned seed, my_uint64 therm_run, my_uint64 nupdates_run);

void observables(double beta, double delta,unsigned N, my_uint64* h_hist,double* boltz, double* h_log_weight, double* M_jacks,double* sus_jacks, unsigned k);

void jakknife (double* M_jacks, double* avg_M, double* error_M, int JACKS,unsigned range);


// initial calculation of total d_energy per worker 
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
computeEnergy(int8_t *d_lattice,int * d_energy)
{
  d_energy[WORKER] = total_energy(d_lattice);
}

__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
computeDeltaE(int8_t *d_lattice, int* d_Delta_E)
{
    int sum = 0;
    for(int i=0; i< d_N; i++) sum = sum + d_lattice[i*d_NUM_WORKERS+WORKER]*d_lattice[i*d_NUM_WORKERS+WORKER];
    d_Delta_E[WORKER] = sum;
}

__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
computeM(int8_t *d_lattice, int* d_Mag)
{
    int sum = 0;
    for(int i=0; i< d_N; i++) sum = sum + d_lattice[i*d_NUM_WORKERS+WORKER];
    d_Mag[WORKER] = sum;
}

int main (int argc, char* argv[])
{ //first argv is lattice size L. Second is gpu machine number.  Third argv is the number of delta in the range. Fourth argv is # of jackknife samples. 
  //for(int GPU = 1; GPU<=1; GPU ++){//to line #:260
  
  unsigned NUM_WORKERS = 0, NofGPU = 1+atoi(argv[3])-atoi(argv[2]),starting_gpu=atoi(argv[2]), last_gpu = atoi(argv[3]);

  //int REQUESTED_GPU = GPU;
  int REQUESTED_GPU = stoi(argv[2]);
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  // prefer cache over shared memory(PreferShared : 48KB, PreferEqual : 32KB, PreferL1 : 16KB, PreferNone : no prefer)
  for(unsigned i = 0; i<NofGPU; i++){
      cudaSetDevice(i+starting_gpu);
      cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  }
  // figure out optimal execution configuration
  // based on GPU architecture and generation
  int currentDevice;
  cudaGetDevice(&currentDevice);
  int maxresidentthreads, totalmultiprocessors;
  cudaDeviceGetAttribute(&maxresidentthreads, cudaDevAttrMaxThreadsPerMultiProcessor, currentDevice);
  cudaDeviceGetAttribute(&totalmultiprocessors, cudaDevAttrMultiProcessorCount, currentDevice);
  int optimum_number_of_workers = maxresidentthreads*totalmultiprocessors;
  if (NUM_WORKERS == 0) {
    NUM_WORKERS = optimum_number_of_workers;
  }

  unsigned L,N;
  L=stoi(argv[1]);
  N=L*L;
  int8_t** h_lattice;
  h_lattice = (int8_t**) calloc(NofGPU,sizeof(int8_t*));
  for(unsigned i=0;i<NofGPU;i++) h_lattice[i] = (int8_t*) calloc(N*NUM_WORKERS,sizeof(int8_t));

  RNG rng;
  for(unsigned gpu =0;gpu<NofGPU;gpu++){
      for(unsigned WORK = 0;WORK < NUM_WORKERS;WORK++){
	  RNG::key_type k = {{WORK*10, 0xdecafbad}};
	  RNG::ctr_type c = {{gpu*1000, SEED, 0, 0xBADCAFE3}};
	  RNG::ctr_type r;
	  for (unsigned i = 0; i < N; i++) {
	      if(i%4 == 0) {
		  ++c[2];
		  r = rng(c, k);
	      }
	      h_lattice[gpu][i*NUM_WORKERS+WORK] = (r123::u01fixedpt<float>(r.v[i%4]) < 1/3.0)+(r123::u01fixedpt<float>(r.v[i%4]) < 2/3.0)-1;
	  }//N*WORK+i (MAX_X*y +x) is same with i*NUM_WORKERS+WORK(x*MAX_Y + y).
      }
  }

  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
    cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }

  unsigned range = stoi(argv[4]);

  //delta, beta 
  double delta = atof(argv[8]), beta = 1.0/0.5;
  double delta_i = delta, gap = atof(argv[9]);

  my_uint64* h_hist, *each_hist[NofGPU];
  //my_uint64 *each_hist[NofGPU];
  h_hist = (my_uint64 *) calloc((N+1)*(N+2)/2,sizeof(my_uint64));
  for(unsigned GPU = 0; GPU<NofGPU;GPU++) each_hist[GPU] = (my_uint64 *) calloc((N+1)*(N+2)/2,sizeof(my_uint64));

  unsigned *naver[NofGPU],* index[NofGPU],*d_naver[NofGPU],* d_index[NofGPU];

  for(unsigned i=0;i<NofGPU;i++){
      naver[i] = (unsigned*) calloc(N*4,sizeof(unsigned));
      d_naver[i] = (unsigned*) calloc(N*4,sizeof(unsigned));
      index[i] = (unsigned*) calloc(N,sizeof(unsigned));
      d_index[i] = (unsigned*) calloc(N,sizeof(unsigned));
  }

  for(unsigned i = 0 ; i<N ; i++){
      int right = i + 1;
      int left = static_cast<int>(i) - 1;
      int up = i + L;
      int down = static_cast<int>(i) - L;

      if(right%L == 0) right = right - L;
      if(i%L == 0) left = left + L;
      if(up > (int)N-1) up = up - N;
      if(down < 0) down = down + N;

      for(unsigned j = 0 ; j<NofGPU ; j++){
	  naver[j][4*i] = right;
	  naver[j][4*i+1] = left; 
	  naver[j][4*i+2] = up;
	  naver[j][4*i+3] = down;
	  index[j][i] = (0==((i/(N/2))%2))*(2*i)%N + (1==((i/(N/2))%2))*(2*i+1)%N;
      }
  }
  int *Delta_set;
  Delta_set = (int *) calloc(N+1,sizeof(int));
  double *avg_M, *avg_sus, *error_M, *error_sus;
  double *h_log_weight[NofGPU],*boltz,*log_weight;
  avg_M = (double *) calloc(range,sizeof(double));
  avg_sus = (double *) calloc(range,sizeof(double));
  error_M = (double *) calloc(range,sizeof(double));
  error_sus = (double *) calloc(range,sizeof(double));
  log_weight = (double *) calloc(N+1,sizeof(double));
  boltz = (double *) calloc(N+1,sizeof(double));

  for(unsigned i=0;i<NofGPU;i++){
      h_log_weight[i] = (double *) calloc(N+1,sizeof(double));
      //h_pre_exp[i] = (float *) calloc(3*17*(N+1),sizeof(float));
  }

  //vector<int> weight_hist(N+1,0.0f);
  //my_uint64 weight_histsum = 0;
 
 //cudaBindTexture(NULL,t_log_weight,d_log_weight,(N+1)*sizeof(double));
  ifstream inFile;
  std::stringstream inname;
  //beta
  inname<<"./fss/L"<<argv[1]<<"/weight_mpi_T050c16_M200_JACK0_DKL1e-4.txt";
  inFile.open(inname.str().c_str());
  if(inFile.fail()){
    cout << "!error!" << endl; 
    //return 1; // no point continuing if the file didn't open...
  }
  int num=0;
  
  while(!inFile.eof()){
      inFile >> Delta_set[num] >> log_weight[num];
      //cout<<"num : "<<num<<"\tDelta : "<<Delta_set[num]<<"\tweight : "<<log_weight[num]<<endl;
      num++; // go to the next number
  }
  inFile.close();
  for(unsigned j=0;j<N+1;j++){ 
      for(unsigned i=0;i<NofGPU;i++){    
          h_log_weight[i][j] = log_weight[j];
      }
  }
/*
  int d_set_temp, weight_hist_temp;
  double h_w_temp;
  while(true){
    if (!(inFile >> d_set_temp))  { break; }
    if (!(inFile >> h_w_temp))    { break; }
    if (!(inFile >> weight_hist_temp)) { break; }
    cout<<"num : "<<num<<"\tDelta : "<<d_set_temp<<"\tweight : "<<h_w_temp<<"\thist : "<<weight_hist_temp<<endl;
    Delta_set.at(num) = d_set_temp;
    h_log_weight.at(num) = h_w_temp;
    weight_hist.at(num) = weight_hist_temp;
   num++; // go to the next number
 }
	  inFile.close();
*/



  //double tmp = h_log_weight[0],tmp2 = h_log_weight[N];
  //for(unsigned i=0;i<N+1;i++)  weight_histsum = weight_histsum + weight_hist[i];
  int8_t* d_lattice[NofGPU];
  my_uint64* d_hist[NofGPU];
  double* d_log_weight[NofGPU];
  int* d_energy[NofGPU], * d_Delta_E[NofGPU], * d_Mag[NofGPU]; 
  for(unsigned i = 0; i<NofGPU;i++){
      cudaSetDevice(i+starting_gpu);
      cudaMalloc((void**)&d_lattice[i], NUM_WORKERS * N * sizeof(int8_t));
      cudaMalloc((void**)&d_naver[i], (N*4)*sizeof(unsigned));
      cudaMalloc((void**)&d_index[i], (N)*sizeof(unsigned));
      cudaMalloc((void**)&d_hist[i], ((N+1)*(N+2)/2)*sizeof(my_uint64));
      cudaMalloc((void**)&d_log_weight[i], (N+1)*sizeof(double));
      cudaMalloc((void**)&d_energy[i],NUM_WORKERS*sizeof(int));
      cudaMalloc((void**)&d_Delta_E[i],NUM_WORKERS*sizeof(int));
      cudaMalloc((void**)&d_Mag[i], NUM_WORKERS*sizeof(int));

      cudaMemcpy(d_log_weight[i],h_log_weight[i],(N+1)*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_naver[i],naver[i],(N*4)*sizeof(unsigned), cudaMemcpyHostToDevice);
      cudaMemcpy(d_index[i],index[i],N*sizeof(unsigned), cudaMemcpyHostToDevice);
      cudaMemcpy(d_lattice[i],h_lattice[i],NUM_WORKERS*N*sizeof(int8_t),cudaMemcpyHostToDevice);

      //copy constants to GPU
      cudaMemcpyToSymbol(d_N, &N, sizeof(unsigned));
      cudaMemcpyToSymbol(d_L, &L, sizeof(unsigned));
      cudaMemcpyToSymbol(d_NUM_WORKERS, &NUM_WORKERS, sizeof(unsigned));

      computeEnergy<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[i],d_energy[i]);
      computeDeltaE<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[i],d_Delta_E[i]);
      computeM<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[i],d_Mag[i]);
  }
  
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    cout << "Error : "<< cudaGetErrorString(err)<< " in " << __FILE__ << __LINE__<< endl;
    exit(err);
  }

  my_uint64 NUPDATES,therm_run;

  //tmp_run too many?? L = 16 , 1e+05 -> 1e+06
  unsigned JACKS = stoi(argv[5]);
  therm_run = N*atoi(argv[6]);
  //therm_run = weight_histsum;
  //cout<<" sum of hist : "<<weight_histsum<<endl;
  NUPDATES = N*atoi(argv[7]);
  cout<<"\n TOTAL UPDATES : "<<NUPDATES*NUM_WORKERS<<" updates by each core : "<<NUPDATES<<" Therm run : "<<therm_run<<endl;

  for(unsigned l=0; l<range; l++){
      cout<<"L"<<argv[1]<<" Progress...  "<<100*l/range<<"%"<<"\r"<<flush;
      delta = delta + gap/(double)range;
      weight_shift(beta,delta,N,log_weight);
      for(unsigned j=0;j<N+1;j++)  boltz[j] = exp(-beta*delta*j-log_weight[j]);
   
      for(unsigned GPU = 0; GPU<NofGPU;GPU++){
          cudaSetDevice(GPU+starting_gpu);
          mcs<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[GPU],d_hist[GPU],d_index[GPU],d_naver[GPU],d_log_weight[GPU],beta,delta,d_energy[GPU],d_Delta_E[GPU],d_Mag[GPU],0,SEED+1000*GPU+l,therm_run,0); // only therm run.
      }

      double *M_jacks,*sus_jacks;
      M_jacks = (double *) calloc(NofGPU,sizeof(double));
      sus_jacks = (double *) calloc(NofGPU,sizeof(double));


      for(unsigned GPU = 0; GPU<NofGPU;GPU++){
          cudaSetDevice(GPU+starting_gpu);
          mcs<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[GPU],d_hist[GPU],d_index[GPU],d_naver[GPU],d_log_weight[GPU],beta,delta,d_energy[GPU],d_Delta_E[GPU],d_Mag[GPU],l,SEED+2000*GPU,0,NUPDATES); 
      }
      for(unsigned GPU=0;GPU<NofGPU;GPU++){
          cudaSetDevice(GPU+starting_gpu);
          cudaMemcpy(each_hist[GPU],d_hist[GPU],((N+1)*(N+2)/2)*sizeof(my_uint64),cudaMemcpyDeviceToHost);
      }
      //if(l==0) for(unsigned cc=0;cc<((N+1)*(N+2)/2);cc++)  cout<<"hist(GPU0):"<<each_hist[0][cc]<<", hist(GPU1):"<<each_hist[1][cc]<<endl;
      for(unsigned GPU=0;GPU<NofGPU;GPU++) observables(beta,delta,N,each_hist[GPU],boltz,log_weight,M_jacks,sus_jacks,GPU);

      jakknife(M_jacks,avg_M,error_M,NofGPU,l);
      jakknife(sus_jacks,avg_sus,error_sus,NofGPU,l);
      free(M_jacks);
      free(sus_jacks);
/*
      double *M_jacks,*sus_jacks;
      M_jacks = (double *) calloc(JACKS,sizeof(double));
      sus_jacks = (double *) calloc(JACKS,sizeof(double));

      for(unsigned jj=0 ; jj<JACKS ; jj++){
          for(unsigned GPU = 0; GPU<NofGPU;GPU++){
              cudaSetDevice(starting_gpu+GPU);
              mcs<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[GPU],d_hist[GPU],d_index[GPU],d_naver[GPU],d_log_weight[GPU],beta,delta,d_energy[GPU],d_Delta_E[GPU],d_Mag[GPU],jj,SEED+2000*GPU+l,0,NUPDATES); 
          }
          for(unsigned GPU=0;GPU<NofGPU;GPU++){
              cudaSetDevice(starting_gpu+GPU);
              cudaMemcpy(each_hist[GPU],d_hist[GPU],((N+1)*(N+2)/2)*sizeof(my_uint64),cudaMemcpyDeviceToHost);
          }
          for(unsigned GPU=0;GPU<NofGPU;GPU++){
              for(unsigned i=0;i<(N+1)*(N+1)/2;i++) h_hist[i] = h_hist[i] + each_hist[GPU][i];
          }
          observables(beta,delta,N,h_hist,boltz,log_weight,M_jacks,sus_jacks,jj);
      }
      jakknife(M_jacks,avg_M,error_M,JACKS,l);
      jakknife(sus_jacks,avg_sus,error_sus,JACKS,l);
      free(M_jacks);
      free(sus_jacks);
*/
  }

  ofstream producefile;
  std::stringstream pdname;
  //beta
  pdname<<"./fss/L"<<argv[1]<<"/observables_gpu"<<NofGPU<<"_T050_JACKgpu_THERM"<<argv[6]<<"N_NUP"<<argv[7]<<"N_delta"<<argv[8]<<"to"<<argv[9]<<"noreset.txt";
  producefile.open(pdname.str().c_str());
  for (size_t i = 0; i < range; i++){
      producefile << delta_i + gap*(i+1)/(double)range << " " << std::setprecision(10) << avg_M[i] << " " <<std::setprecision(10)<< error_M[i]<<" "<<std::setprecision(10)<< avg_sus[i]<<" "<<std::setprecision(10)<< error_sus[i] << std::endl;
  }
  producefile.close();

  
 
  return 0;
}

__device__ __forceinline__ int inter_energy(unsigned i, int8_t* lattice)
{
  int right = i+1;
  int left = static_cast<int>(i) - 1;
  int up = i + d_L;
  int down = static_cast<int>(i) - d_L;

  if(right%d_L == 0) right = right - d_L;
  if(i%d_L == 0) left = left + d_L;
  if(up > d_N-1) up = up - d_N;
  if(down < 0) down = down + d_N;

  return -lattice[i*d_NUM_WORKERS+WORKER]*(lattice[right*d_NUM_WORKERS+WORKER] + lattice[left*d_NUM_WORKERS+WORKER] + lattice[up*d_NUM_WORKERS+WORKER] + lattice[down*d_NUM_WORKERS+WORKER]);
  //return -lattice[i*d_NUM_WORKERS+WORKER+device*d_NUM_WORKERS*d_N]*(lattice[right*d_NUM_WORKERS+WORKER+device*d_NUM_WORKERS*d_N] + lattice[left*d_NUM_WORKERS+WORKER+device*d_NUM_WORKERS*d_N] + lattice[up*d_NUM_WORKERS+WORKER+device*d_NUM_WORKERS*d_N] + lattice[down*d_NUM_WORKERS+WORKER+device*d_NUM_WORKERS*d_N]);
}

__device__ __forceinline__ int total_energy(int8_t* lattice)
{
  int sum =0;
  for (unsigned i=0;i<d_N;i++){
    sum = sum + inter_energy(i,lattice);
  }
  //divide by 2, to ignore double count
  return (sum >> 1);
}

void weight_shift(double beta,double delta,unsigned N,double* h_log_weight)
{
    //double tmp = h_log_weight[0],tmp2 = h_log_weight[N];
    vector<double> inside_of_exp(N+1,0.0);
    for(unsigned i=0;i<N+1;i++)  inside_of_exp.at(i) = -h_log_weight[i]-beta*delta*i;

    double max = *max_element(inside_of_exp.begin(),inside_of_exp.end()); 
  
    for(unsigned i=0;i<N+1;i++)  h_log_weight[i] = h_log_weight[i] + max;
}
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK,MY_KERNEL_MIN_BLOCKS)
mcs(int8_t* lattice, my_uint64* d_hist,unsigned* index,unsigned* naver, double* d_log_weight, double beta, double delta, int* d_energy, int* d_Delta_E,int* d_Mag, unsigned iter,unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run)
{
    RNG rng;
    RNG::key_type k1 = {{WORKER, 0xdecafbad}};
    //RNG::key_type k3 = {{0xC001cafe, 0xdecafead}};
    RNG::key_type k3 = {{WORKER+40000, 0xdecafead}};
    RNG::ctr_type c = {{0, seed, iter, 0xBADC0DED}};
    RNG::ctr_type r1, r3;


    //reset histogram using WORKER not using memcpy. usually num of worker is larger than num of hist. so.. usually i = 0
    for(size_t i=0; i<((d_N+1)*(d_N+2)/(2*d_NUM_WORKERS)) +1; i++){
      if(i*d_NUM_WORKERS+WORKER<(d_N+2)*(d_N+1)/2+1) d_hist[i*d_NUM_WORKERS + WORKER] = 0;
    }
    __syncthreads();//before hist added, all elements of hist should be zero.

    int energy, E_Delta, magnetization;
    energy = d_energy[WORKER];
    E_Delta = d_Delta_E[WORKER];
    magnetization = d_Mag[WORKER];

    // thermalization
    for(my_uint64 i=0; i<therm_run; i++){
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1);  r3 = rng(c,k3);
      }
      unsigned idx = index[i%d_N];
      int new_s = lattice[idx*d_NUM_WORKERS+WORKER] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = lattice[idx*d_NUM_WORKERS+WORKER];
      new_s = new_s -3*(new_s==2) +3*(new_s==-2); //-2 to +1 and +2 to -1

      lattice[idx*d_NUM_WORKERS+WORKER] = new_s*(old_s==0)+old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx*d_NUM_WORKERS+WORKER])*(lattice[naver[4*idx]*d_NUM_WORKERS+WORKER] + lattice[naver[4*idx+1]*d_NUM_WORKERS+WORKER] + lattice[naver[4*idx+2]*d_NUM_WORKERS+WORKER] + lattice[naver[4*idx+3]*d_NUM_WORKERS+WORKER]), dDel = new_s*new_s - old_s*old_s, dM = new_s-old_s;

      //beta here
      double diff =  -dE*2 + d_log_weight[E_Delta+dDel] -d_log_weight[E_Delta];
      if(diff>=0){
        lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else lattice[idx*d_NUM_WORKERS+WORKER] = old_s;
    }//FOR end


    //estimating distribution with current Weight
    for(my_uint64 i=0;i<nupdates_run;i++){
      //randomly pick spin first -> check whether flip the spin or not.
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r3 = rng(c,k3);
      }
      
      unsigned idx = index[i%d_N];

      int new_s = lattice[idx*d_NUM_WORKERS+WORKER]+2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = lattice[idx*d_NUM_WORKERS+WORKER];
      new_s = new_s -3*(new_s==2) +3*(new_s==-2); //-2 to +1 and +2 to -1

      lattice[idx*d_NUM_WORKERS+WORKER] = new_s*(old_s==0) + old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx*d_NUM_WORKERS+WORKER])*(lattice[naver[4*idx]*d_NUM_WORKERS+WORKER]+ lattice[naver[4*idx+1]*d_NUM_WORKERS+WORKER] + lattice[naver[4*idx+2]*d_NUM_WORKERS+WORKER] + lattice[naver[4*idx+3]*d_NUM_WORKERS+WORKER]), dDel = new_s*new_s - old_s*old_s, dM = new_s-old_s;

      //beta here
      double diff =  -dE*2 + d_log_weight[E_Delta+dDel] -d_log_weight[E_Delta];
      if(diff>=0){
        lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else lattice[idx*d_NUM_WORKERS+WORKER] = old_s;

      atomicAdd(d_hist + (magnetization + E_Delta*E_Delta + 2*E_Delta)/2,1);
      //if(i%(d_N*40)) atomicAdd(d_hist + (magnetization + E_Delta*E_Delta + 2*E_Delta)/2,1);
    }
    
    d_energy[WORKER] = energy;
    d_Delta_E[WORKER] = E_Delta;
    d_Mag[WORKER] = magnetization;
}

void observables(double beta, double delta,unsigned N, my_uint64* h_hist,double* boltz, double* h_log_weight, double* M_jacks,double* sus_jacks, unsigned k)
{
  long double sum = 0, M1 = 0, M2 = 0;
  unsigned i;
  int  m; 
  for(i = 0; i <= N ; i++){  
    for(m = -i; m <= 0 ; m = m+2){
        sum = sum + h_hist[(i*i+2*i+m)/2]*boltz[i];
        M1 = M1 + -m*h_hist[(i*i+2*i+m)/2]*boltz[i];
        M2 = M2 + m*m*h_hist[(i*i+2*i+m)/2]*boltz[i];
    }
    for(; m <= (int)i ; m = m+2){
        sum = sum + h_hist[(i*i+2*i+m)/2]*boltz[i];
        M1 = M1 + m*h_hist[(i*i+2*i+m)/2]*boltz[i];
        M2 = M2 + m*m*h_hist[(i*i+2*i+m)/2]*boltz[i];
    }
  }

  M_jacks[k] = M1/sum;
  sus_jacks[k] = (M2/sum-(M1/sum)*(M1/sum))*beta/(long double)N;
}

void jakknife (double* M_jacks, double* avg_M, double* error_M, int JACKS,unsigned range)
{
 double avg=0;
 for(int i=0; i<JACKS ; i++) avg += M_jacks[i];
 double M_i[JACKS];
 for(int i=0; i<JACKS ; i++) M_i[i] = (avg - M_jacks[i])/(double)(JACKS-1);
 avg = avg/(double)JACKS;

 double dev=0;
 for(int i=0; i<JACKS ; i++) dev += (avg - M_i[i])*(avg-M_i[i]);
 dev = (JACKS-1)*dev/(double)(JACKS);
 dev = sqrt(dev);

 avg_M[range] = avg;
 error_M[range] = dev;
}

