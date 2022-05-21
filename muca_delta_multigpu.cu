//multiGPU. trivial + recursive weight update. checkerboard spin update
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <getopt.h>
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
// 2048 is max threads per multiprocessor
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
void hist_range(my_uint64* hist,int N, int& start, int& end);
double kull_leibler(my_uint64* hist,int N);
void jakknife (double* M_jacks, double* avg_M, double* error_M, unsigned JACKS);

__global__ void
__launch_bounds__(WORKERS_PER_BLOCK,MY_KERNEL_MIN_BLOCKS)
mcs2(int8_t* d_lattice, unsigned* d_hist, float* d_weight, unsigned* index, unsigned* naver, int* d_energy,int* d_Delta_E, unsigned iter, unsigned seed, unsigned therm_run,unsigned nupdates_run);


__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
mcs(int8_t* d_lattice, unsigned* d_hist, float* d_weight, unsigned* index, unsigned* naver, int* d_energy,int* d_Delta_E, unsigned iter, unsigned seed, unsigned therm_run,unsigned nupdates_run);

void weight_recursion(unsigned N,my_uint64* hist, double* p_acc, int hist_start, int hist_end, double* log_weight);

// initial calculation of total d_energy per worker 
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
computeEnergies(int8_t *d_lattice, int* d_energy)
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



int main(int argc, char* argv[]) // first argv is the number of lattice size. second is Requested GPU.
{ 
  unsigned NUM_WORKERS = 0, NofGPU = 1+atoi(argv[3])-atoi(argv[2]), starting_gpu=atoi(argv[2]),last_gpu = atoi(argv[3]),jacks = atoi(argv[4]), initNUPDATES = atoi(argv[5]),NUPDATES,Nwidth_step[jacks];
  bool recursion = (bool)atoi(argv[6]), extrapolation = (bool)atoi(argv[7]),converged = 0, reach = 0;
  my_uint64 total_run[jacks];
  // prefer cache over shared memory(PreferShared : 48KB, PreferEqual : 32KB, PreferL1 : 16KB, PreferNone : no prefer)
  for(unsigned i =0;i<NofGPU;i++){
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
  NUM_WORKERS = optimum_number_of_workers;

  //constant should be out of main func??
  unsigned L,N;
  L=atoi(argv[1]);
  N=L*L;
  //vector<int8_t> h_lattice(N*NUM_WORKERS,0.0);
  int8_t** h_lattice;
  h_lattice = (int8_t**) calloc(NofGPU,sizeof(int8_t*));
  for(unsigned i=0;i<NofGPU;i++) h_lattice[i] = (int8_t*) calloc(N*NUM_WORKERS,sizeof(int8_t));


  my_uint64 *h_hist;
  unsigned **each_hist;
  h_hist = (my_uint64 *) calloc(N+1,sizeof(my_uint64));
  each_hist = (unsigned **) calloc(NofGPU,sizeof(unsigned*));
  for(unsigned i=0;i<NofGPU;i++) each_hist[i] = (unsigned*) calloc(N+1,sizeof(unsigned));

  unsigned *naver[NofGPU], *index[NofGPU],*d_naver[NofGPU], *d_index[NofGPU];

  for(unsigned i=0;i<NofGPU;i++){
      naver[i] = (unsigned*) calloc(N*4,sizeof(unsigned));
      d_naver[i] = (unsigned*) calloc(N*4,sizeof(unsigned));
      index[i] = (unsigned*) calloc(N,sizeof(unsigned));
      d_index[i] = (unsigned*) calloc(N,sizeof(unsigned));
  }

  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
      cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }
 
  // copy constants to GPU
  //initializing random spins on the lattice
  RNG rng;
  for(unsigned j =0;j<NofGPU;j++){
      for(unsigned WORK = 0;WORK < NUM_WORKERS;WORK++){
          RNG::key_type k = {{WORK, 0xdecafbad}};
          RNG::ctr_type c = {{j, SEED, 0xBADCAB1E, 0xBADC0DED}};
          RNG::ctr_type r;
          for (unsigned i = 0; i < N; i++) {
              if(i%4 == 0) {
                 ++c[0];
                  r = rng(c, k);
              }
              h_lattice[j][i*NUM_WORKERS+WORK] = (r123::u01fixedpt<float>(r.v[i%4]) < 1/3.0)+(r123::u01fixedpt<float>(r.v[i%4]) < 2/3.0)-1;  
          }//N*WORK+i (MAX_X*y +x) is same with i*NUM_WORKERS+WORK(x*MAX_Y + y).
      }
      //cout<<"spin[0] : "<<(int)h_lattice[j][0]<<endl;
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

  int* d_energy[NofGPU],*DeltaE[NofGPU];
  int* d_Delta_E[NofGPU];
  int8_t* d_lattice[NofGPU];
  float* d_log_weight[NofGPU];
  for(unsigned i=0;i<NofGPU;i++){
    //d_log_weight[i] = (double*) calloc(N+1,sizeof(double));
      DeltaE[i] = (int*)calloc(NUM_WORKERS,sizeof(int));
  }
  unsigned* d_hist[NofGPU];

  //int8_t * latt;
  //latt = (int8_t*) calloc(N*NUM_WORKERS,sizeof(int8_t));
  for(unsigned i =0;i<NofGPU;i++){
      cudaSetDevice(i+starting_gpu);
      cudaMalloc((void**)&d_hist[i], (N+1)*sizeof(unsigned));
      cudaMalloc((void**)&d_Delta_E[i], NUM_WORKERS*sizeof(int));
      cudaMalloc((void**)&d_energy[i],NUM_WORKERS*sizeof(int));  
      cudaMalloc((void**)&d_lattice[i], NUM_WORKERS * N * sizeof(int8_t));
      cudaMalloc((void**)&d_log_weight[i], (N+1)*sizeof(float));
      cudaMalloc((void**)&d_naver[i], 4 * N * sizeof(int));
      cudaMalloc((void**)&d_index[i], N * sizeof(int));

      cudaMemcpyToSymbol(d_N, &N, sizeof(unsigned));
      cudaMemcpyToSymbol(d_L, &L, sizeof(unsigned));
      cudaMemcpyToSymbol(d_NUM_WORKERS, &NUM_WORKERS, sizeof(unsigned));    

      cudaMemcpy(d_naver[i],naver[i],N*4*sizeof(unsigned),cudaMemcpyHostToDevice);
      cudaMemcpy(d_index[i],index[i],N*sizeof(unsigned),cudaMemcpyHostToDevice);
  }


  //initializing d_energy, weight, histogram
  //is it okay to use N, L in cuda function not the d_N and d_L? maybe for the access speed?

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    cout << "Error : "<< cudaGetErrorString(err)<< " in " << __FILE__ << __LINE__<< endl;
    exit(err);
  }

  float *h_log_weight[NofGPU];
  double *p_acc,*log_weight;
  for(unsigned i = 0 ; i<NofGPU ; i++) h_log_weight[i] = (float *) calloc(N+1,sizeof(float));
  //p_acc = (double *) calloc(N+1,sizeof(double));
  log_weight = (double *) calloc(N+1,sizeof(double));

  if(extrapolation){ 
      ifstream initweightfile;
      stringstream initwtfilename;
      initwtfilename<<"./fss/L"<<argv[1]<<"/initweight_L128_T050.txt";
      initweightfile.open(initwtfilename.str().c_str());
      for(unsigned i =0;i<=N;i++){  
          initweightfile>>log_weight[i];
          for(unsigned j = 0; j<NofGPU;j++){
              h_log_weight[j][i] = (float)log_weight[i];
          }
      }
      initweightfile.close(); 
  }

//load saved data...
/*
  ifstream inFile;
  std::stringstream inname;
  //beta here
  inname<<"/home/null/bc_muca/fss/L"<<argv[1]<<"/iter_luxe_T050.txt";
  double Delta[N+1];
  inFile.open(inname.str().c_str());
  if(inFile.fail()){
      cout << "error" << endl;
      return 1; // no point extrapolation if the file didn't open...
  }
  //double beta = 1/0.609;
  string s;
  getline(inFile,s);
  int num=0;
  for(int i =0 ; i< N+1 ; i++){ 
      inFile >> Delta[i] >> h_log_weight[i] >> h_hist[i];
      if(h_log_weight[i] ==0) num++;
  }
  inFile.close();
  
  double tmp_end = h_log_weight[N]+2*N*beta;
  //init arrays to zero.
  for(int i =0; i<N+1;i++){
     if(i<num) h_log_weight[i] = -2*i*beta;
     else h_log_weight[i] =h_log_weight[i] - tmp_end;
  }
  for(unsigned i =0; i<N+1;i++){
      //for(unsigned j =0; j<NofGPU;j++)  h_log_weight[i] = -2*i/0.609;
      p_acc[i] = 0;
  }
*/

  //T050
/*  double f_x = 0.176*N,m_x=num,s_x=0.888*N,f_y=6535.349,m_y=-2*(double)num/1.2-h_log_weight[num],s_y=1043.27;

  int i =0;
  for(i=0;i<f_x;i++){ 
    if(i==num) break;
    h_log_weight[i] = h_log_weight[i] - i*f_y/f_x;
  }
  for(;i<m_x;i++) {
    if(i==num) break;
    h_log_weight[i] = h_log_weight[i] + (i-f_x)*(f_y-m_y)/(m_x-f_x)-f_y;
  }
  for(;i<s_x;i++) {
    if(i==num) break;
    h_log_weight[i] = h_log_weight[i] + (i-m_x)*(m_y-s_y)/(s_x-m_x)-m_y;
  }
  for(;i<=N;i++) {
    if(i==num) break;
    h_log_weight[i] = h_log_weight[i] + (double)(i-s_x)*s_y/(double)(N-s_x)-s_y;
  }
*/
/*
  //T0609
  double f_x = 0.176*N,s_x=0.888*N,f_y=947.3867,s_y=585.2373;


  int i =0;
  for(i=0;i<f_x;i++){ 
    h_log_weight[i] = h_log_weight[i] - i*f_y/f_x;
  }
  for(;i<s_x;i++) {
    h_log_weight[i] = h_log_weight[i] + (i-f_x)*(f_y-s_y)/(s_x-f_x)-f_y;
  }
  for(;i<=N;i++) {
    h_log_weight[i] = h_log_weight[i] + (double)(i-s_x)*s_y/(double)(N-s_x)-s_y;
  }
*/

//for(int j =0;j<N+1;j++)cout<<h_log_weight[j]<<endl;
//return 0;


  //unsigned width = N-1;
  unsigned width = 10;
  unsigned therm_coeff=1;
  unsigned therm_run=therm_coeff*N;
  //double nupdates_run = N;
  int hist_start, hist_end;
  float kernel_time,mem_time;
  double *total_kernel,*total_mem,*Nwidth_time;
  total_kernel = (double *) calloc(jacks,sizeof(double));
  total_mem = (double *) calloc(jacks,sizeof(double));
  Nwidth_time = (double *) calloc(jacks,sizeof(double));
  cudaEvent_t i_ker,f_ker,i_mem,f_mem;

  ofstream iterfile,shortiter;
  std::stringstream itrname,shortitrname;

  for (unsigned mm = 0; mm<jacks;mm++){
      //beta
      //itrname<<"/home/null/muca/fss/L"<<argv[1]<<"/iter_mpi_T120c"<<NUM_WORKERS<<"M"<<NUPDATES<<".txt";
      //if(extrapolation)  shortitrname<<"./fss/L"<<argv[1]<<"/shortiter_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_Jack"<<mm<<"_extrapolated"<<extrapolation<<".txt";
      //else shortitrname<<"./fss/L"<<argv[1]<<"/shortiter_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_Jack"<<mm<<"_iterupdate1.01.txt";
      //shortiter.open(shortitrname.str().c_str());
      //shortiter<<"#iter width D_KL each_time accumulated_time NUPDATES acc_NUP"<<endl;
      total_run[mm] = 0;
      converged = 0, reach = 0;
      NUPDATES = initNUPDATES;
      RNG rng;
      for(unsigned j =0;j<NofGPU;j++){
          for(unsigned WORK = 0;WORK < NUM_WORKERS;WORK++){
              RNG::key_type k = {{10*WORK + mm*2, 0xdecafbad}};
              RNG::ctr_type c = {{j*2, SEED, 0xBADCAB1E, 0xBADC0DED}};
              RNG::ctr_type r;
              for (unsigned i = 0; i < N; i++) {
                  if(i%4 == 0) {
                     ++c[0];
                      r = rng(c, k);
                  }
                  h_lattice[j][i*NUM_WORKERS+WORK] = (r123::u01fixedpt<float>(r.v[i%4]) < 1/3.0)+(r123::u01fixedpt<float>(r.v[i%4]) < 2/3.0)-1;  
              }//N*WORK+i (MAX_X*y +x) is same with i*NUM_WORKERS+WORK(x*MAX_Y + y).
          }
      //cout<<"spin[0] : "<<(int)h_lattice[j][0]<<endl;
      }
      for(unsigned i =0;i<NofGPU;i++){
          cudaSetDevice(i+starting_gpu);
          cudaMemcpy(d_lattice[i],h_lattice[i],NUM_WORKERS*N*sizeof(int8_t),cudaMemcpyHostToDevice);
          computeEnergies<<<NUM_WORKERS/WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice[i], d_energy[i]);
          computeDeltaE<<<NUM_WORKERS/WORKERS_PER_BLOCK,WORKERS_PER_BLOCK>>>(d_lattice[i], d_Delta_E[i]);

          cudaMemcpy(h_lattice[i],d_lattice[i],NUM_WORKERS*N*sizeof(int8_t),cudaMemcpyDeviceToHost);
          cudaMemcpy(DeltaE[i],d_Delta_E[i],NUM_WORKERS*sizeof(int8_t),cudaMemcpyDeviceToHost);
          cout<<"E_DELTA : "<<DeltaE[i][0]<<endl; 
          //for(int j = 0;j<N;j++){cout<<(int)h_lattice[j*NUM_WORKERS]<<" "; if((j+1)%L==0) cout<<endl;}
          cout<<"---------------------------------------- "<<endl;
      }
      if(extrapolation)  shortitrname<<"./fss/L"<<argv[1]<<"/shortiter_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_therm"<<therm_coeff<<"N_Jack"<<mm<<"_extrapolated"<<extrapolation<<".txt";
      else shortitrname<<"./fss/L"<<argv[1]<<"/shortiter_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_therm"<<therm_coeff<<"N_Jack"<<mm<<"_iterupdate1.01_rand.txt";
      shortiter.open(shortitrname.str().c_str());
      shortiter<<"#iter width D_KL mcs_time memcpy_time accumulated_time NUPDATES acc_NUP"<<endl;
      //shortiter.close();

      //main loop for weight convergence
      for(size_t k=0;  ; k++){
          //width is the length of the histogram.
          //if(width<N) nupdates_run = 20*5.7*width*width*sqrt(width)/(NUM_WORKERS*NofGPU); //optimal # of sweeps per iteration.
          //else nupdates_run = nupdates_run*1.01; //This is where width = L^2 + 1 (whole d_energy covered)
          //therm_run = 30*width;
          if(width == N){
              NUPDATES = NUPDATES + 1 ;
              NUPDATES = NUPDATES*1.01;
          }
          total_run[mm] = total_run[mm] + NUPDATES*NUM_WORKERS*NofGPU; 
          for(unsigned l=0;l<NofGPU;l++){
              cudaSetDevice(l+starting_gpu);
	      cudaMemcpy(d_log_weight[l], h_log_weight[l], (N+1)*sizeof(float), cudaMemcpyHostToDevice);
	  }
          cudaEventCreate(&i_ker);
          cudaEventCreate(&f_ker);
          cudaEventRecord(i_ker,0);
          for(unsigned l=0;l<NofGPU;l++){
              cudaSetDevice(l+starting_gpu);
              mcs<<<NUM_WORKERS/WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice[l],d_hist[l],d_log_weight[l],d_index[l],d_naver[l],d_energy[l],d_Delta_E[l],k,SEED+1000*l,therm_run,NUPDATES);
          }
          cudaEventRecord(f_ker,0);
          cudaEventSynchronize(f_ker);
          cudaEventElapsedTime(&kernel_time,i_ker,f_ker);
          cudaEventDestroy(i_ker);
          cudaEventDestroy(f_ker);

          cudaEventCreate(&i_mem);
          cudaEventCreate(&f_mem);
          cudaEventRecord(i_mem,0);
          for(unsigned l=0;l<NofGPU;l++){
		  cudaSetDevice(l+starting_gpu);
		  cudaMemcpy(each_hist[l],d_hist[l],(N+1)*sizeof(unsigned), cudaMemcpyDeviceToHost);
          }
	  cudaEventRecord(f_mem,0);
          cudaEventSynchronize(f_mem);
          cudaEventElapsedTime(&mem_time,i_mem,f_mem);
          cudaEventDestroy(i_mem);
          cudaEventDestroy(f_mem);

          total_kernel[mm] += (double)kernel_time;
          total_mem[mm] += (double)mem_time;

          cudaError_t err2 = cudaGetLastError();
          if (err2 != cudaSuccess) {
              cout << "Error: " << cudaGetErrorString(err2) << " in " << __FILE__ << __LINE__ << endl;
              exit(err2);
          }
          for(unsigned i=0;i<N+1;i++) h_hist[i]=0;
          for(unsigned i=0;i<N+1;i++){ 
              for(unsigned j=0;j<NofGPU;j++){ 
                  h_hist[i] += each_hist[j][i];
                  //if(each_hist[j][i]!=0) cout<<each_hist[j][i]<<endl;
                  //cout<<each_hist[0][i]<<endl;
              }
          }

          hist_range(h_hist,N,hist_start,hist_end);
          unsigned width_temp = hist_end - hist_start;
          if(width_temp>width) width = width_temp; //update width with histogram width.
          // end if the weight is converged(D_KL:Kullback-Leiber divergence).
          double D_KL = kull_leibler(h_hist,N);
          //shortiter<<"#iter width D_KL mcs_time memcpy_time accumulated_time NUPDATES acc_NUP"<<endl;
          //shortiter.open(shortitrname.str().c_str(),std::ios_base::app);
          //cudaDeviceSynchronize();
          shortiter<<k<<" "<<width_temp<<" "<<D_KL<<" "<<kernel_time*1e-3<<" "<<mem_time*1e-3<<" "<<total_kernel[mm]*1e-3<<" "<<NUPDATES<<" "<<total_run[mm]<<endl;
          //shortiter.close();
          if((width==N)&&(reach==0)){
                  cout<<"Width reached N... Total mcs time : "<<total_kernel[mm]*1e-3<<endl;
                  Nwidth_time[mm] = total_kernel[mm]*1e-3;
                  Nwidth_step[mm] = total_run[mm];
                  reach = 1;
          }
          if((width==N)&&(D_KL<1e-4)) converged = 1;


          /*
          ofstream iterfile,timefile;
          std::stringstream itrname,timename;
          //beta
          itrname<<"/home/null/bc_muca/fss/L"<<argv[1]<<"/iter_multi_T120.txt";
          timename<<"/home/null/bc_muca/fss/L"<<argv[1]<<"/time_T120_GPU"<<argv[2]<<".txt";
          iterfile.open(itrname.str().c_str());
          iterfile<<"#NITER = " << k << "width = " << width << "; nupdates = "<< nupdates_run<<"dk= "<< D_KL<< endl;
          for(unsigned i = 0; i<= N;i++)  iterfile << static_cast<int>(i) << " " << setprecision(10) << log_weight[i] << " " << h_hist[i] << endl;
         iterfile.close();
          timefile.open(timename.str().c_str());
          timefile << setprecision(10) << kernel_time/1000 << " " << setprecision(10) <<mem_time/1000<< endl; 
          timefile.close();
          */
          if(converged == 1) break; 
          //UPDATE WEIGHT if it uses extrapolated weight, it does not update weight until it reaches width == N
          if(~(extrapolation&(width!=N))){
              for(unsigned i = 0; i<=N; i++) if(h_hist[i]>1) log_weight[i] = log_weight[i] - log((double)(h_hist[i]));
          //weight_recursion(N,h_hist,p_acc,hist_start,hist_end,log_weight);
              double tmp = log_weight[0];
              for(unsigned i = 0; i<=N;i++) {
                  log_weight[i] = log_weight[i] - tmp;
                  for(unsigned j = 0; j<NofGPU;j++){
                      h_log_weight[j][i] = (float)log_weight[i];
                  }
	      }
          }
      }//end of the main loop for weight convergence
  
      shortiter.close();
      if(converged) {
          total_kernel[mm] = total_kernel[mm]*1e-3;
          cout <<"D_KL is below 1e-4 !! "<<endl;
          cout<< "Total run : "<<total_run[mm]<<""<<"\tN_width time : "<<Nwidth_time[mm]<<"\tMCS time : "<<total_kernel[mm]<<"\tMEM time : "<<total_mem[mm]<<"\tRatio of time : "<<Nwidth_time[mm]/total_kernel[mm]<<endl;

          ofstream weightfile;
          std::stringstream weightname;
          if(extrapolation)  weightname<<"./fss/L"<<argv[1]<<"/weight_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_therm"<<therm_coeff<<"N_JACK"<<mm<<"_DKL1e-4_fromextrapolatedinit"<<extrapolation<<".txt";
          else  weightname<<"./fss/L"<<argv[1]<<"/weight_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_therm"<<therm_coeff<<"N_JACK"<<mm<<"_DKL1e-4_iterupdate1.01_rand.txt";
          weightfile.open(weightname.str().c_str());
          for (unsigned i = 0; i <= N; i++){
              weightfile << i <<" "<< std::setprecision(10)<<log_weight[i]<<endl;
          }
          weightfile.close();
      }
      else if ((WORKER==0)&&(converged!=1)) cout<<"Noooo...not converged";
  }

  double total_run_d[jacks],Nwidth_step_d[jacks],total_time_mean = 0,total_memtime_mean = 0,Nwidth_time_mean = 0, total_time_std = 0,total_memtime_std = 0, Nwidth_time_std = 0,total_run_mean=0,total_run_std=0,Nwidth_step_mean=0,Nwidth_step_std=0;
  for(unsigned i=0;i<jacks;i++) {
      total_run_d[i] = (double)total_run[i];
      Nwidth_step_d[i] = (double)Nwidth_step[i];
  }
  jakknife (total_kernel, &total_time_mean, &total_time_std, jacks);
  jakknife (total_mem, &total_memtime_mean, &total_memtime_std, jacks);
  jakknife (Nwidth_time, &Nwidth_time_mean, &Nwidth_time_std, jacks);
  jakknife (total_run_d, &total_run_mean, &total_run_std, jacks);
  jakknife (Nwidth_step_d, &Nwidth_step_mean, &Nwidth_step_std, jacks);

  if(converged){
  //wrting result of converged weight, time consumed and total steps.
      ofstream producefile;
      std::stringstream pdname;
      //beta
      if(extrapolation)  pdname<<"./fss/L"<<argv[1]<<"/converge_result_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"M"<<initNUPDATES<<"_therm"<<therm_coeff<<"N_extrapolatedinit"<<extrapolation<<".txt";
      else  pdname<<"./fss/L"<<argv[1]<<"/converge_result_gpu"<<NofGPU<<"_T050c"<<NUM_WORKERS<<"M"<<initNUPDATES<<"_therm"<<therm_coeff<<"N_iterupdate1.01_rand.txt";
      producefile.open(pdname.str().c_str());
      producefile <<"#total_time_mean"<< " " <<"total_time_std"<<" " <<"total_memtime_mean"<<" " <<"total_memtime_std"<<" "<<"Nwidth_time_mean"<<" "<<"Nwidth_time_std"<<" "<<"total_run_mean"<<" "<<"total_run_std"<<" "<<"Nwidth_step_mean"<<" "<<"Nwidth_step_std"<<std::endl;
      producefile <<total_time_mean<< " " <<total_time_std<<" "<<" " <<total_memtime_mean<<" " <<total_memtime_std<<" "<<Nwidth_time_mean<<" "<<Nwidth_time_std<<" "<<total_run_mean<<" "<<total_run_std<<" "<<Nwidth_step_mean<<" "<<Nwidth_step_std<<std::endl;
      producefile <<"#total_mcstime"<<" "<<"total_memtime"<<" "<<"Nwidth_time"<<" "<<"total_run"<<" "<<"Nwidth_step"<<std::endl;
      for (unsigned i = 0; i < jacks; i++){
          producefile <<total_kernel[i]<<" "<<total_mem[i]<<" "<<Nwidth_time[i]<<" "<<total_run[i]<<" "<<Nwidth_step[i]<<std::endl;
      }
     producefile.close();
  }



/*
  //producing distribution with converged weight.
  if(converged) cout <<"produce!"<<endl;
  else if (converged!=1) cout<<"not converged"<<endl;
  
  ofstream producefile;
  std::stringstream pdname;
  //beta
  pdname<<"/home/null/bc_muca/fss/L"<<argv[1]<<"/E_Delta_multi_T120.txt";
  producefile.open(pdname.str().c_str());
  for (unsigned i = 0; i < N + 1; i++){
    producefile << static_cast<int>(i) << " " << std::setprecision(10) << log_weight[i] << " " << h_hist[i] << std::endl;
  }
  producefile.close();

  cout<<"\ntotal kernel time : "<<total_kernel/1000<<"\ntotal mem time : "<<total_mem/1000<<endl;
*/
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
}

__device__ int total_energy(int8_t* lattice)
{
  int sum =0;
  for (unsigned i=0;i<d_N;i++){
    sum = sum + inter_energy(i,lattice);
  }
  //divide by 2, to ignore double count
  return (sum >> 1);
}

void hist_range(my_uint64* hist,int N, int& start, int& end)
{
  int i;
//cout<<"\nhist range started";
  start = N+1;
  end = 0;
  for(i=0; i<N+1;i++){
    if(hist[i]>1){
      if(i<start) start = i;
      break;
    }
  }
  for(i=N+1;i-->0;){
    if(hist[i]>1){
      if(i>end) end=i;
      break;
    }
  }
  if(start > end){ 
    start =0;
    end = 0;
  }
//cout<<"\n hist start : "<<start<<" end : "<<end;
}

double kull_leibler(my_uint64* hist, int N)
{
  double num_bin = 0, num_event = 0, kld = 0,P,Q;
  for(int i=0;i<N+1;i++){
    if(hist[i]>0){
      num_event = num_event + hist[i];
      num_bin = num_bin + 1.0;
    }
  }
  //printf("event:%lf, bin:%lf\n",num_event,num_bin);

  for(int i=0; i<N+1;i++){
    if(hist[i]>0){
      P = hist[i]/num_event;
      Q = 1.0/num_bin;
      kld = kld + P*log(P/Q);
    }
  }

  return kld;
}

void jakknife (double* M_jacks, double* avg_M, double* error_M, unsigned JACKS)
{
    double avg=0;
    for(unsigned i=0; i<JACKS ; i++) avg += M_jacks[i];
    double M_i[JACKS];
    for(unsigned i=0; i<JACKS ; i++) M_i[i] = (avg - M_jacks[i])/(double)(JACKS-1);
    avg = avg/(double)JACKS;

    double dev=0;
    for(unsigned i=0; i<JACKS ; i++) dev += (avg - M_i[i])*(avg-M_i[i]);
    dev = (JACKS-1)*dev/(double)(JACKS);
    dev = sqrt(dev);

    *avg_M = avg;
    *error_M = dev;
}

__global__ void
__launch_bounds__(WORKERS_PER_BLOCK,MY_KERNEL_MIN_BLOCKS)
mcs2(int8_t* d_lattice, unsigned* d_hist, float* d_weight,unsigned* index, unsigned* naver, int* d_energy, int* d_Delta_E,  unsigned iter,unsigned seed, unsigned therm_run,unsigned nupdates_run)
{
    RNG rng;
    RNG::key_type k1 = {{2*WORKER, 0xdecafbad}};
    RNG::key_type k2 = {{4*WORKER+1000, 0xc0decafe}};
    RNG::key_type k3 = {{2*WORKER+50000, 0x11cafead}};
    RNG::ctr_type c = {{0, seed, iter, 0xBADC0DED}};
    RNG::ctr_type r1, r2, r3;
    //reset histogram using WORKER not using memcpy. usually num of worker is larger than num of hist. so.. usually i = 0

    for(size_t i=0; i<((d_N+1)/d_NUM_WORKERS) +1; i++){
        if(i*d_NUM_WORKERS+WORKER<d_N+1) d_hist[i*d_NUM_WORKERS + WORKER] = 0;
    }
    __syncthreads();//before adding hist, all elements of hist should be zero.

    int energy,Delta_E;
    energy = d_energy[WORKER];
    Delta_E = d_Delta_E[WORKER];

    // thermalization
    for(my_uint64 i=0; i<therm_run; i++){
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r2=rng(c,k2); r3 = rng(c,k3);
      }

      unsigned idx = (r123::u01fixedpt<float>(r2.v[i%4]))*d_N;
      int new_s = d_lattice[idx*d_NUM_WORKERS+WORKER] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = d_lattice[idx*d_NUM_WORKERS+WORKER];
      new_s = new_s -3*(new_s==2) +3*(new_s==-2); //-2 to +1 and +2 to -1

      ///unsigned new_idx = idx*d_NUM_WORKERS+WORKER;
      d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s*(old_s==0) + old_s;

      //int dE = old_s*(new_s - old_s)*inter_energy(idx,d_lattice), dDel = new_s*new_s - old_s*old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-d_lattice[idx*d_NUM_WORKERS+WORKER])*(d_lattice[naver[4*idx]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+1]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+2]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+3]*d_NUM_WORKERS+WORKER]), dDel = new_s*new_s - old_s*old_s;
      //beta
      //double diff =  exp(-dE/1.4+d_weight[Delta_E+dDel]-d_weight[Delta_E]);
//beta
      float diff =  -dE*2+d_weight[Delta_E+dDel]-d_weight[Delta_E];
      if(diff>=0){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else d_lattice[idx*d_NUM_WORKERS+WORKER] = old_s;
 
    }//FOR end


    //estimating distribution with current Weight
    for(my_uint64 i=0;i<nupdates_run;i++){
      //randomly pick spin first -> check whether flip the spin or not.
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r2=rng(c,k2); r3 = rng(c,k3);
      }

      unsigned idx = (r123::u01fixedpt<float>(r2.v[i%4]))*d_N;
      int new_s = d_lattice[idx*d_NUM_WORKERS+WORKER] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = d_lattice[idx*d_NUM_WORKERS+WORKER];
      new_s = new_s -3*(new_s==2) +3*(new_s==-2); //-2 to +1 and +2 to -1

      d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s*(old_s==0) + old_s;
      //int dE = old_s*(new_s - old_s)*inter_energy(idx,d_lattice), dDel = new_s*new_s - old_s*old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-d_lattice[idx*d_NUM_WORKERS+WORKER])*(d_lattice[naver[4*idx]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+1]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+2]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+3]*d_NUM_WORKERS+WORKER]), dDel = new_s*new_s - old_s*old_s;
      //beta
      //double diff =  exp(-dE/1.4+d_weight[Delta_E+dDel]-d_weight[Delta_E]);
//beta
      float diff =  -dE*2+d_weight[Delta_E+dDel]-d_weight[Delta_E];
      if(diff>=0){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else d_lattice[idx*d_NUM_WORKERS+WORKER] = old_s;
 
      atomicAdd(d_hist + Delta_E,1);
    }

  d_energy[WORKER] = energy;
  d_Delta_E[WORKER] = Delta_E;



}
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK,MY_KERNEL_MIN_BLOCKS)
mcs(int8_t* d_lattice, unsigned* d_hist, float* d_weight,unsigned* index, unsigned* naver, int* d_energy, int* d_Delta_E,  unsigned iter,unsigned seed, unsigned therm_run,unsigned nupdates_run)
{
    RNG rng;
    RNG::key_type k1 = {{2*WORKER, 0xdecafbad}};
    RNG::key_type k3 = {{2*WORKER+50000, 0x11cafead}};
    RNG::ctr_type c = {{0, seed, iter, 0xBADC0DED}};
    RNG::ctr_type r1, r3;
    //reset histogram using WORKER not using memcpy. usually num of worker is larger than num of hist. so.. usually i = 0

    for(size_t i=0; i<((d_N+1)/d_NUM_WORKERS) +1; i++){
      if(i*d_NUM_WORKERS+WORKER<d_N+1) d_hist[i*d_NUM_WORKERS + WORKER] = 0;
    }
    __syncthreads();//before adding hist, all elements of hist should be zero.

    int energy,Delta_E;
    energy = d_energy[WORKER];
    Delta_E = d_Delta_E[WORKER];

    // thermalization
    for(my_uint64 i=0; i<therm_run; i++){
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r3 = rng(c,k3);
      }

      unsigned idx = index[i%d_N];
      int new_s = d_lattice[idx*d_NUM_WORKERS+WORKER] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = d_lattice[idx*d_NUM_WORKERS+WORKER];
      new_s = new_s -3*(new_s==2) +3*(new_s==-2); //-2 to +1 and +2 to -1

      ///unsigned new_idx = idx*d_NUM_WORKERS+WORKER;
      d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s*(old_s==0) + old_s;

      //int dE = old_s*(new_s - old_s)*inter_energy(idx,d_lattice), dDel = new_s*new_s - old_s*old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-d_lattice[idx*d_NUM_WORKERS+WORKER])*(d_lattice[naver[4*idx]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+1]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+2]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+3]*d_NUM_WORKERS+WORKER]), dDel = new_s*new_s - old_s*old_s;
      //beta
      //double diff =  exp(-dE/1.4+d_weight[Delta_E+dDel]-d_weight[Delta_E]);
//beta
      float diff =  -dE*2+d_weight[Delta_E+dDel]-d_weight[Delta_E];
      if(diff>=0){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else d_lattice[idx*d_NUM_WORKERS+WORKER] = old_s;
 
    }//FOR end


    //estimating distribution with current Weight
    for(my_uint64 i=0;i<nupdates_run;i++){
      //randomly pick spin first -> check whether flip the spin or not.
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r3 = rng(c,k3);
      }

      unsigned idx = index[i%d_N];
      int new_s = d_lattice[idx*d_NUM_WORKERS+WORKER] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = d_lattice[idx*d_NUM_WORKERS+WORKER];
      new_s = new_s -3*(new_s==2) +3*(new_s==-2); //-2 to +1 and +2 to -1

      d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s*(old_s==0) + old_s;
      //int dE = old_s*(new_s - old_s)*inter_energy(idx,d_lattice), dDel = new_s*new_s - old_s*old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-d_lattice[idx*d_NUM_WORKERS+WORKER])*(d_lattice[naver[4*idx]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+1]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+2]*d_NUM_WORKERS+WORKER] + d_lattice[naver[4*idx+3]*d_NUM_WORKERS+WORKER]), dDel = new_s*new_s - old_s*old_s;
      //beta
      //double diff =  exp(-dE/1.4+d_weight[Delta_E+dDel]-d_weight[Delta_E]);
//beta
      float diff =  -dE*2+d_weight[Delta_E+dDel]-d_weight[Delta_E];
      if(diff>=0){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        d_lattice[idx*d_NUM_WORKERS+WORKER] = new_s;
        energy = energy + dE;
        Delta_E = Delta_E +dDel;
      }
      else d_lattice[idx*d_NUM_WORKERS+WORKER] = old_s;
 
      atomicAdd(d_hist + Delta_E,1);
    }

  d_energy[WORKER] = energy;
  d_Delta_E[WORKER] = Delta_E;

}



void weight_recursion(unsigned N,my_uint64* hist, double* p_acc, int hist_start, int hist_end, double* log_weight)
{
/*
    double sum = 0, tmp = 0;
    tmp = log_weight[hist_start];
    
    for(int i = hist_start ; i<hist_end ; i++){
        double p = hist[i]*hist[i+1]/(double)(hist[i]+hist[i+1]);
        double k = p/(p+p_acc[i]);
        double delta_S = -tmp + log_weight[i+1] + k*log(hist[i]/(double)hist[i+1]);
        tmp = log_weight[i+1];
        log_weight[i+1] = log_weight[i] + delta_S;
        p_acc[i] = p_acc[i] + p;
    }//slower
*/
    double log_weight_bf[N+1];
    for(unsigned i = 0 ; i<=N ; i++) log_weight_bf[i] = log_weight[i];
    for(int i = hist_start ; i<hist_end ; i++){
        if((hist[i]!=0)&(hist[i+1]!=0)){
          double p = hist[i]*hist[i+1]/(double)(hist[i]+hist[i+1]);
          double k = p/(p+p_acc[i]);
          double delta_S = -log_weight_bf[i] + log_weight_bf[i+1] + k*log(hist[i]/(double)hist[i+1]);
          log_weight[i+1] = log_weight[i] + delta_S;
          p_acc[i] = p_acc[i] + p;
        }
    }
}
