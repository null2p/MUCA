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
#include <mpi.h>

#include "Random123/philox.h"
#include "Random123/examples/uniform.hpp"

#define SEED 100

unsigned L,N;
typedef r123::Philox4x32_R<7> RNG;
typedef unsigned long long my_uint64;
double my_uint64_max = pow(2.0,64)-1;

using namespace std;

int inter_energy(unsigned i, int* lattice);
int total_energy(int* lattice);
void hist_range(my_uint64* hist, int& start, int& end);
double kull_leiber(my_uint64* hist);
void mcs(int* lattice, my_uint64* hist, double* log_weight,int* index, int* naver, int* energy,int* E_delta, unsigned WORKER, unsigned iter, unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run);
void weight_recursion(my_uint64* hist,double* p_acc,int hist_start,int hist_end,double* log_weight);
void jakknife (double* M_jacks, double* avg_M, double* error_M, int JACKS);

int main(int argc, char* argv[])
{ 
  MPI::Init();
  unsigned WORKER = static_cast<unsigned>(MPI::COMM_WORLD.Get_rank());
  unsigned NUM_WORKERS = static_cast<unsigned>(MPI::COMM_WORLD.Get_size());
  
  L=atoi(argv[1]);
  N=L*L;
  int lattice[N]; 
  int jacks = atoi(argv[3]);
  my_uint64 NUPDATES=atoi(argv[2]),therm_run,Nwidth_step[jacks], total_run[jacks];
  unsigned initNUPDATES=NUPDATES;
  bool converged,continuing=atoi(argv[4]);
  double mcs_time[jacks], Nwidth_time[jacks];
  float iterupdate = 1.05;

  
  if(WORKER==0) cout <<"#############L = "<<L<<", Jacks = "<<jacks<<", initNUPDATES = "<<NUPDATES<<", DKL < 1e-4, NUPDATES *= "<<iterupdate<<", Cores = "<<NUM_WORKERS<<", extrapolated init weight = "<<continuing<<endl;
  for(int i=0;i<jacks;i++){
      Nwidth_step[i]=0;
      total_run[i]=0;
      mcs_time[i]=0;
      Nwidth_time[i]=0;
  }
  for(int mm=0;mm<jacks;mm++){
      NUPDATES = initNUPDATES;
      //initializing random spins on the lattice
      RNG rng;
      RNG::key_type k = {{10*WORKER+mm*10000, 0xdecafbad}};
      RNG::ctr_type c = {{0, SEED, 0xBADCAB1E, 0xBADC0DED}};
      RNG::ctr_type r;
      for (size_t i = 0; i < N; i++) {
          if(i%4 == 0) {
              ++c[0];
              r = rng(c, k);
          }
          lattice[i] = (r123::u01fixedpt<float>(r.v[i%4]) < 1/3.0)+(r123::u01fixedpt<float>(r.v[i%4]) < 2/3.0)-1;
      } 

      //initializing energy, weight, histogram
      int energy = total_energy(lattice);
      int E_delta =0;
      for(unsigned i=0; i<N;i++) E_delta = E_delta + lattice[i]*lattice[i];
      double *log_weight,*p_acc;
      my_uint64 *hist, *mpi_hist;

      log_weight = (double *) calloc(N+1,sizeof(double));
      p_acc = (double *) calloc(N+1,sizeof(double));
      hist = (my_uint64 *) calloc(N+1,sizeof(my_uint64));
      mpi_hist = (my_uint64 *) calloc(N+1,sizeof(my_uint64));

      int naver[N*4], index[N];

      for(unsigned i = 0; i < N ; i++){
          int right = i + 1;
          int left = static_cast<int>(i) - 1;
          int up = i + L;
          int down = static_cast<int>(i) - L;

          if(right%L == 0) right = right - L;
          if(i%L == 0) left = left + L;
          if(up > N-1) up = up - N;
          if(down < 0) down = down + N;

          naver[4*i] = right;
          naver[4*i+1] = left;
          naver[4*i+2] = up;
          naver[4*i+3] = down;

          index[i] = (0==((i/(N/2))%2))*(2*i)%N + (1==((i/(N/2))%2))*(2*i+1)%N;
          //for(int j =0; j<4 ; j++) cout<<"naver : "<<naver[i][j]<<endl;
      }

      unsigned width = 10;
      int hist_start, hist_end;
      if((WORKER==0)&continuing){
          ifstream initweightfile, latfile;
          stringstream initwtfilename,latfilename;
          initwtfilename<<"./weight/extrapolated_initweight_L"<<argv[1]<<"_T050_DKL1e-4.txt";
          initweightfile.open(initwtfilename.str().c_str());
          for(unsigned i =0;i<=N;i++)  initweightfile>>log_weight[i];

          latfile.close();
	  initweightfile.close();
      }

      ofstream iterfile,shortiter;
      std::stringstream itrname,shortitrname;
      //beta
      //itrname<<"/home/null/muca/fss/L"<<argv[1]<<"/iter_mpi_T120c"<<NUM_WORKERS<<"M"<<NUPDATES<<".txt";
      if(continuing)  shortitrname<<"/home/null/muca/fss/L"<<argv[1]<<"/shortiter_mpi_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_Jack"<<mm<<"_extrapolated"<<continuing<<".txt";
      else shortitrname<<"/home/null/muca/fss/L"<<argv[1]<<"/shortiter_mpi_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_Jack"<<mm<<"_iterupdate1.01.txt";
      shortiter.open(shortitrname.str().c_str());
      shortiter<<"#iter width D_KL each_time accumulated_time NUPDATES acc_NUP"<<endl;
      clock_t time_i, time_f;
      bool reach = 0;
      converged = 0;
      //main loop for weight convergence
      for(size_t k=0; ;k++){
          NUPDATES = NUPDATES*1.05;
          MPI::COMM_WORLD.Barrier();
          MPI::COMM_WORLD.Bcast(log_weight,N+1,MPI::DOUBLE,0); 
          //width is the length of the histogram.
          //if(width<N) nupdates_run = 10*width*width*sqrt(width)/NUM_WORKERS; //optimal # of sweeps per iteration. 
   
          total_run[mm] =total_run[mm] + NUPDATES*NUM_WORKERS;
          //therm_run = 30*width;
          time_i = clock();
          mcs(lattice,hist,log_weight,index,naver,&energy,&E_delta,WORKER,k,mm,0,NUPDATES);
          time_f = clock();
          mcs_time[mm] = mcs_time[mm] + (time_f - time_i);
          MPI::COMM_WORLD.Reduce(hist,mpi_hist,N+1,MPI::UNSIGNED_LONG_LONG,MPI::SUM,0);

          if(WORKER ==0){
              hist_range(mpi_hist,hist_start,hist_end);
              unsigned width_temp = hist_end - hist_start;
              if(width_temp>width) width = width_temp; //update width with histogram width.
              // end if the weight is converged(KLD:Kullback-Leiber divergence).
              double KLD = kull_leiber(mpi_hist);
              shortiter<<	k <<" "<<width_temp<<" "<<KLD<<" "<<1e-6*(time_f-time_i)<<" "<<mcs_time[mm]*1e-6<<NUPDATES<<" "<<total_run[mm]<<endl;
              if((width==N)&&(reach==0)){
                  cout<<"Width reached N... Total mcs time : "<<mcs_time[mm]/1e+6<<endl;
                  Nwidth_time[mm] = mcs_time[mm]/1e+6;
                  Nwidth_step[mm] = total_run[mm];
                  reach = 1;
              }
              if((width==N)&&(KLD<1e-4)) converged = 1; 
              //if(width==N) converged = 1; 
          }
          MPI::COMM_WORLD.Barrier();
          MPI::COMM_WORLD.Bcast(&converged,1,MPI::INT,0);
          MPI::COMM_WORLD.Bcast(&width,1,MPI::INT,0); 
          //UPDATE WEIGHT
          if(WORKER==0){
              if(continuing){
                  if(reach==1){
                      for(int i = 0; i<= N;++i) if(mpi_hist[i]>0) log_weight[i] = log_weight[i] - log((double)(mpi_hist[i]));

                      double temp = log_weight[0];
                      for(int i = 0; i<= N;++i) if(mpi_hist[i]>0) log_weight[i] = log_weight[i] - temp;
                  }
              }
              else{
                  for(int i = 0; i<= N;++i) if(mpi_hist[i]>0) log_weight[i] = log_weight[i] - log((double)(mpi_hist[i]));
             
                  double temp = log_weight[0];
                  for(int i = 0; i<= N;++i) if(mpi_hist[i]>0) log_weight[i] = log_weight[i] - temp;
              }
              //weight_recursion(mpi_hist,p_acc,hist_start,hist_end,log_weight);
          }
          if(converged==1) break;
      }//end of the main loop for weight convergence

      if((WORKER==0)&&converged) {
          mcs_time[mm] = mcs_time[mm]/1e+6;
          cout <<"D_KL is below 1e-4 !! "<<endl;
          cout<< "Total run : "<<total_run[mm]<<""<<"\tN_width time : "<<Nwidth_time[mm]<<"\tMCS time : "<<mcs_time[mm]<<"\tRatio of time : "<<Nwidth_time[mm]/mcs_time[mm]<<endl;

          ofstream weightfile;
          std::stringstream weightname;
          if(continuing)  weightname<<"/home/null/muca/fss/L"<<argv[1]<<"/weight_mpi_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_JACK"<<mm<<"_DKL1e-4_fromextrapolatedinit"<<continuing<<".txt";
          else  weightname<<"/home/null/muca/fss/L"<<argv[1]<<"/weight_mpi_T050c"<<NUM_WORKERS<<"_M"<<initNUPDATES<<"_JACK"<<mm<<"_DKL1e-4_iterupdate1.01.txt";
          weightfile.open(weightname.str().c_str());
          for (int i = 0; i <= N; i++){
              weightfile << i <<" "<< std::setprecision(10)<<log_weight[i]<<endl;
          }
          weightfile.close();
      }
      else if ((WORKER==0)&&(converged!=1)) cout<<"Noooo...not converged";

  }

  double total_run_d[jacks],Nwidth_step_d[jacks],total_time_mean = 0,Nwidth_time_mean = 0, total_time_std = 0, Nwidth_time_std = 0,total_run_mean=0,total_run_std=0,Nwidth_step_mean=0,Nwidth_step_std=0;
  for(int i=0;i<jacks;i++) {
      total_run_d[i] = (double)total_run[i];
      Nwidth_step_d[i] = (double)Nwidth_step[i];
  }
  jakknife (mcs_time, &total_time_mean, &total_time_std, jacks);
  jakknife (Nwidth_time, &Nwidth_time_mean, &Nwidth_time_std, jacks);
  jakknife (total_run_d, &total_run_mean, &total_run_std, jacks);
  jakknife (Nwidth_step_d, &Nwidth_step_mean, &Nwidth_step_std, jacks);

  if((WORKER==0)&converged){ 
  //wrting result of converged weight, time consumed and total steps.
      ofstream producefile;
      std::stringstream pdname;
      //beta
      if(continuing)  pdname<<"/home/null/Desktop/muca/fss/L"<<argv[1]<<"/converge_result_mpi_T050c"<<NUM_WORKERS<<"M"<<initNUPDATES<<"extrapolatedinit"<<continuing<<".txt";
      else  pdname<<"/home/null/muca/fss/L"<<argv[1]<<"/converge_result_mpi_T050c"<<NUM_WORKERS<<"M"<<initNUPDATES<<"iterupdate1.01"<<".txt";
      producefile.open(pdname.str().c_str());
      producefile <<"#total_time_mean"<< " " <<"total_time_std"<<" "<<"Nwidth_time_mean"<<" "<<"Nwidth_time_std"<<" "<<"total_run_mean"<<" "<<"total_run_std"<<" "<<"Nwidth_step_mean"<<" "<<"Nwidth_step_std"<<std::endl;
      producefile <<total_time_mean<< " " <<total_time_std<<" "<<Nwidth_time_mean<<" "<<Nwidth_time_std<<" "<<total_run_mean<<" "<<total_run_std<<" "<<Nwidth_step_mean<<" "<<Nwidth_step_std<<std::endl;
      producefile <<"#total_time"<<" "<<"Nwidth_time"<<" "<<"total_run"<<" "<<"Nwidth_step"<<std::endl;
      for (int i = 0; i < jacks; i++){
          producefile <<mcs_time[i]<<" "<<Nwidth_time[i]<<" "<<total_run[i]<<" "<<Nwidth_step[i]<<std::endl;
      }
     producefile.close();
  }

  MPI::COMM_WORLD.Barrier();
  MPI::Finalize();
  return 0;
}

int inter_energy(unsigned i, int* lattice)
{
  int right = i+1;
  int left = static_cast<int>(i) - 1;
  int up = i + L;
  int down = static_cast<int>(i) - L;

  if(right%L == 0) right = right - L;
  if(i%L == 0) left = left + L;
  if(up > N-1) up = up - N;
  if(down < 0) down = down + N;

  return -lattice[i]*(lattice[right] + lattice[left] + lattice[up] + lattice[down]);
}

int total_energy(int* lattice)
{
  int sum =0;
  for (unsigned i=0;i<N;i++){
    sum = sum + inter_energy(i,lattice);
  }
  //divide by 2, to ignore double count
  return (sum >> 1);
}

void hist_range(my_uint64* hist, int& start, int& end)
{
  int i;
//cout<<"\nhist range started";
  start = N+1;
  end = 0;
  for(i=0; i<N+1;i++){
    if(hist[i]>1){
      if(i<start) start =i;
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

double kull_leiber(my_uint64* hist)
{
  double num_bin = 0, num_event = 0, kld = 0,P,Q;
  for(unsigned i=0;i<N+1;i++){
    if(hist[i]>0){
      num_event = num_event + hist[i];
      num_bin = num_bin + 1.0;
    }
  }
  //printf("event:%lf, bin:%lf\n",num_event,num_bin);

  for(unsigned i=0; i<N+1;i++){
    if(hist[i]>0){
      P = hist[i]/num_event;
      Q = 1.0/num_bin;
      kld = kld + P*log(P/Q);
    }
  }

  return kld;
}

void mcs(int* lattice, my_uint64* hist, double* log_weight,int* index, int* naver, int* energy_tmp, int* E_delta_tmp, unsigned WORKER, unsigned iter,unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run)
{
    RNG rng;
    RNG::key_type k1 = {{10*WORKER+1000*seed, 0xdecafbad}};
    RNG::key_type k2 = {{100*WORKER+4000*seed, 0x20202041}};
    RNG::key_type k3 = {{1000*WORKER+10000*seed, 0x11cafead}};
    RNG::ctr_type c = {{0, 0xbadbad11, 2*iter, 0xBADC0DED}};
    RNG::ctr_type r1,r2,r3;
    //reset histogram
    for(unsigned i=0;i<N+1;i++) hist[i] = 0;
    int energy = *energy_tmp;
    int E_Delta = *E_delta_tmp;

  // thermalization
    for(my_uint64 i=0; i<therm_run; i++){
        if(i%4 == 0) {
            ++c[0];
            r1 = rng(c, k1); r2 = rng(c,k2); r3 = rng(c,k3);
        }

        //unsigned idx = index[i%N];
        unsigned idx = (unsigned)(r123::u01fixedpt<float>(r2.v[i%4])*N);
        int new_s = lattice[idx] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
        int old_s = lattice[idx];
        new_s = new_s -3*(new_s == 2) +3*(new_s == -2); //-2 to +1 and +2 to -1
        lattice[idx] = new_s*(old_s==0)+old_s;
        int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx])*(lattice[naver[4*idx]] + lattice[naver[4*idx+1]] + lattice[naver[4*idx+2]] + lattice[naver[4*idx+3]]), dDel = new_s*new_s - old_s*old_s;
        //double diff =  exp(-dE/1.4+log_weight[E_Delta+dDel]-log_weight[E_Delta]);
//beta -dE*beta
        double diff =  -dE*2+log_weight[E_Delta+dDel]-log_weight[E_Delta];
        if(diff >= 0){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
        }
        else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
        }
        else lattice[idx] = old_s;
    } 



    //estimating distribution with current Weight
    for(my_uint64 i=0;i<nupdates_run;i++){
        if(i%4 == 0) {
            ++c[0];
            r1 = rng(c, k1); r2 = rng(c,k2); r3 = rng(c,k3);
         }

        //unsigned idx = index[i%N];
        unsigned idx = (unsigned)(r123::u01fixedpt<float>(r2.v[i%4])*N);
//cout<< "idx : "<<idx<<endl;
        int new_s = lattice[idx] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
        int old_s = lattice[idx];
        new_s = new_s -3*(new_s == 2) +3*(new_s == -2); //-2 to +1 and +2 to -1
 //cout<<" E_J = "<<E_J<<" = "<<inter_energy(idx,lattice)<<endl;
        lattice[idx] = new_s*(old_s==0)+old_s;
        int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx])*(lattice[naver[4*idx]] + lattice[naver[4*idx+1]] + lattice[naver[4*idx+2]] + lattice[naver[4*idx+3]]), dDel = new_s*new_s - old_s*old_s;
  //cout<<"dE = "<<dE<<" = "<<dE2<<endl;
        //double diff =  exp(-dE/1.4+log_weight[E_Delta+dDel]-log_weight[E_Delta]);
//beta -beta*dE
        double diff =  -dE*2+log_weight[E_Delta+dDel]-log_weight[E_Delta];
        if(diff >= 0){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
        }
        else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
        }  
        else lattice[idx] = old_s;
        hist[E_Delta] = hist[E_Delta]+1;
    }

  *energy_tmp = energy;
  *E_delta_tmp = E_Delta;
}


void weight_recursion(my_uint64* hist,double* p_acc,int hist_start,int hist_end,double* log_weight)
{
    double sum = 0, log_weight_bf[N+1];

    for(int i = 0 ; i<=N ; i++) log_weight_bf[i] = log_weight[i];

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

void jakknife (double* M_jacks, double* avg_M, double* error_M, int JACKS)
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
