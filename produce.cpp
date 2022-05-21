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

typedef r123::Philox4x32_R<7> RNG;
typedef unsigned long long my_uint64;
double my_uint64_max = pow(2.0,64)-1;

unsigned L,N;

using namespace std;

int inter_energy(unsigned i, int* lattice);
int total_energy(int* lattice);
void weight_shift(double beta,double delta,double* log_weight);
//void mcs(int* therm_mag, int* lattice,double beta, double delta,double& M1, double& M2,double& sum, my_uint64* hist,int* index, int* naver,double* boltz, double* log_weight,int* E_Delta_tmp, int* energy_tmp, unsigned WORKER, int* magnetization_tmp, unsigned iter, unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run);
void mcs(int* lattice,double beta, double delta, unsigned long* hist, int* index, int* naver,double* boltz, double* log_weight,int* E_Delta_tmp, int* energy_tmp, unsigned WORKER,int* magnetization_tmp, unsigned iter,unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run);
void observables(double beta, double delta,unsigned N, unsigned long* h_hist,double* boltz, double* log_weight, double* M_jacks,double* sus_jacks, unsigned k);
void jakknife(double* M_jacks, double* avg_M, double* error_M, int JACKS,unsigned range);

/*
inline unsigned bin(int M, unsigned N)
{
  return (M+N)>>1;
}*/


int main(int argc, char* argv[])
{ // first argv is lattice size L. Second argv is the number of delta in the range of 1.5 ~ 2.5 Third argv is # of jackknife samples.
  
  MPI::Init();
  unsigned WORKER = static_cast<unsigned>(MPI::COMM_WORLD.Get_rank()); 
  unsigned NUM_WORKERS = static_cast<unsigned>(MPI::COMM_WORLD.Get_size());


  L=atoi(argv[1]);
  N=L*L;
  int* lattice;
  lattice = (int *) calloc(N,sizeof(int));
  unsigned range = atoi(argv[2]);
 
  int* Delta_set;
  Delta_set = (int *) calloc(N+1,sizeof(int));
  double* avg_M, *error_M, *avg_sus, *error_sus, *log_weight,*boltz;  
  avg_M = (double *) calloc(range,sizeof(double));
  error_M = (double *) calloc(range,sizeof(double));
  avg_sus = (double *) calloc(range,sizeof(double));
  error_sus = (double *) calloc(range,sizeof(double));
  log_weight = (double *) calloc(N+1,sizeof(double));
  boltz = (double *) calloc(N+1,sizeof(double));

  int histsize = ((N+1)*(N+2))/2;
  unsigned long * hist, *mpi_hist;
  hist = (unsigned long *) calloc(histsize,sizeof(unsigned long));
  mpi_hist = (unsigned long *) calloc(histsize,sizeof(unsigned long));
 
  int *naver,* index;
  naver = (int *) calloc(N*4,sizeof(int));
  index = (int *) calloc(N,sizeof(int));
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
  }
 

  ifstream inFile;
  std::stringstream inname;
  //beta here
  //inname<<"/home/null/Desktop/muca/fss/L"<<argv[1]<<"/weight_mpi_T050c16_M200_JACK2_DKL1e-4.txt";
  //inname<<"/home/null/bc_muca/fss/L"<<argv[1]<<"/weight_mpi_T050c16_M200_JACK0_DKL1e-4_iterupdate1.01.txt";
  //inname<<"/home/null/Desktop/muca/fss/L"<<argv[1]<<"/weight_mpi_T050c16_M200_JACK2_DKL1e-4_fromextrapolatedinit1.txt";
  inname<<"./fss/L"<<argv[1]<<"/weight_mpi_T050c16_M200_JACK0_DKL1e-4.txt";
  if(WORKER==0){
    inFile.open(inname.str().c_str());
    if(inFile.fail()){
      cout << "error" << endl; 
      return 1; // no point continuing if the file didn't open...
    }
    int num=0;
    while(!inFile.eof()){ 
      inFile >> Delta_set[num] >> log_weight[num] ;
      num++; // go to the next number
    }
    inFile.close(); 
    //for(unsigned i=0;i<=num;i++) log_weight.at(i) = log_weight.at(i) - log_weight.at(bin(0,N));
  }

  MPI::COMM_WORLD.Barrier();
  MPI::COMM_WORLD.Bcast(log_weight,N+1,MPI::DOUBLE,0);
  MPI::COMM_WORLD.Bcast(Delta_set,N+1,MPI::INT,0);

  RNG rng;
  RNG::key_type k = {{WORKER*10, 0xdecafbad}};
  RNG::ctr_type c = {{0, SEED, 0xBADCAB1E, 0xBADC0DED}};
  RNG::ctr_type r;
  for (size_t i = 0; i < N; i++) {
      if(i%4 == 0) {
          ++c[0];
          r = rng(c, k);
      }
      lattice[i] = (r123::u01fixedpt<float>(r.v[i%4]) < 1/3.0) + (r123::u01fixedpt<float>(r.v[i%4]) < 2/3.0)-1;
  }
  
  int energy = total_energy(lattice);
  int magnetization = 0;
  int Delta_E = 0;
  my_uint64 hist_sum = 0;
  for(unsigned i =0; i<N;i++){
     magnetization += lattice[i];
     Delta_E += lattice[i]*lattice[i];
     hist_sum += hist[i];
  }
  hist_sum +=hist[N];

  double M1,M2,sum,M,sus,mpi_M1,mpi_M2,mpi_sum; //estimation of observable M.
  //beta here also nee to change delta by the beta.
  //double delta = 1.9875 , beta = 2;
  //double delta_i = delta, gap = 0.002;
  double delta = atof(argv[6]) , beta = 2;
  double delta_i = delta, gap = atof(argv[7]);
  //end point will be delta + gap, each step will be divided by the range... gap/range.  

  //cout <<"worker = "<<WORKER<<"\tinit_magnet = "<< magnetization <<"\n";
  
  my_uint64 NUPDATES,therm_run;

  size_t JACKS = atoi(argv[3]);
  therm_run = N*atoi(argv[4]);
  NUPDATES = N*atoi(argv[5]);
  if(WORKER==0) cout<<"\n TOTAL UPDATES : "<<NUM_WORKERS*NUPDATES<<"\n THERM RUN : "<<therm_run<<endl;

  //int* therm_mag;
  //therm_mag = (int *) calloc((int)therm_run,sizeof(int));

  for(unsigned l=0; l<range; l++){

      RNG rng;
      RNG::key_type k = {{WORKER*2000+l*2, 0xdecafbad}};
      RNG::ctr_type c = {{0, SEED, 0xBADCAB1E, 0xBADC0DED}};
      RNG::ctr_type r;
      for (size_t i = 0; i < N; i++) {
          if(i%4 == 0) {
              ++c[0];
              r = rng(c, k);
          }
          lattice[i] = (r123::u01fixedpt<float>(r.v[i%4]) < 1/3.0) + (r123::u01fixedpt<float>(r.v[i%4]) < 2/3.0)-1;
      }
  
      int energy = total_energy(lattice);
      int magnetization = 0;
      int Delta_E = 0;
      my_uint64 hist_sum = 0;
      for(unsigned i =0; i<N;i++){
          magnetization += lattice[i];
          Delta_E += lattice[i]*lattice[i];
          hist_sum += hist[i];
      }
      hist_sum +=hist[N];
      
      if(WORKER==0) cout<<"L"<<argv[1]<<" Progress(CPU)...  "<<100*l/range<<"%"<<"\r"<<flush;
      delta = delta + gap/(double)range;
      weight_shift(beta,delta,log_weight);
      for(int i =0;i<N+1;i++) boltz[i] = expl(-beta*delta*i-log_weight[i]);

      //mcs(therm_mag,lattice,beta,delta,M1,M2,sum,hist,index,naver,boltz,log_weight,&Delta_E ,&energy,WORKER,&magnetization,0,SEED+1000+l,therm_run,0);
      mcs(lattice,beta,delta,hist,index,naver,boltz,log_weight,&Delta_E,&energy,WORKER,&magnetization,l,SEED+1000,therm_run,0);
      double* M_jacks, *sus_jacks;
      M_jacks = (double *) calloc(JACKS,sizeof(double));
      sus_jacks = (double *) calloc(JACKS,sizeof(double));
      
      for(unsigned k=0 ; k<JACKS ; k++){
          /*
          //thermfile is made for saving all observables by time steps. It is used for drawing figure how much thermaization we need.
          if(WORKER==0){
              ofstream thermfile;
              std::stringstream thermname;
              //beta here
              thermname<<"/home/null/bc_muca/fss/L"<<argv[1]<<"/thermmag_Ed"<<delta<<"_Jack"<<k<<"_T050.txt";
              thermfile.open(thermname.str().c_str());
              thermfile << "#Magnetization, _T050 _Ed"<< delta <<" _Jack"<<k<< std::endl;
              for (int i = 0; i < (int)therm_run; i++){
                  thermfile << therm_mag[i] << std::endl;
              }//not beta but need to change first delta
              thermfile.close();
          }
          */
          //M=0; M1=0; M2=0; mpi_M1=0;sum=0;mpi_M2=0;
          //mcs(therm_mag,lattice,beta, delta,M1,M2,sum,hist,index,naver,boltz,log_weight,&Delta_E,&energy,WORKER,&magnetization,k,SEED+2000+l,0,NUPDATES);
          for(int i = 0;i<histsize;i++)  mpi_hist[i] = 0;

          mcs(lattice,beta,delta,hist,index,naver,boltz,log_weight,&Delta_E,&energy,WORKER,&magnetization,l,k,0,NUPDATES);
          MPI::COMM_WORLD.Reduce(hist,mpi_hist,histsize,MPI::UNSIGNED_LONG,MPI::SUM,0);
          observables(beta, delta,N, mpi_hist,boltz, log_weight, M_jacks,sus_jacks, k);

/*
      double RealM = M1/sum;
      double RealChi = M2/sum - RealM*RealM;
      MPI::COMM_WORLD.Reduce(&RealM,&mpi_M1,1,MPI::DOUBLE,MPI::SUM,0);
      MPI::COMM_WORLD.Reduce(&RealChi,&mpi_M2,1,MPI::DOUBLE,MPI::SUM,0); 
      if(WORKER==0){
        M_jacks[k] = mpi_M1/(double)NUM_WORKERS;
        sus_jacks[k] = beta*mpi_M2/(double)(N*NUM_WORKERS);
      }
      MPI::COMM_WORLD.Reduce(&M1,&mpi_M1,1,MPI::DOUBLE,MPI::SUM,0);
      MPI::COMM_WORLD.Reduce(&M2,&mpi_M2,1,MPI::DOUBLE,MPI::SUM,0); 
      MPI::COMM_WORLD.Reduce(&sum,&mpi_sum,1,MPI::DOUBLE,MPI::SUM,0);
      if(WORKER==0){
        M_jacks[k] = mpi_M1/mpi_sum;
        sus_jacks[k] = beta*(mpi_M2/mpi_sum-(mpi_M1/mpi_sum)*(mpi_M1/mpi_sum))/(double)N;
      }

*/
    }
    if(WORKER==0) {
      jakknife(M_jacks,avg_M,error_M,JACKS,l);
      jakknife(sus_jacks,avg_sus,error_sus,JACKS,l);
    }
  }

  if(WORKER==0){
    ofstream producefile;
    std::stringstream pdname;
    //beta here
    //pdname<<"./fss/L"<<argv[1]<<"/observables_mpi_T050_JACK"<<argv[3]<<"_THERM"<<argv[4]<<"_NUP"<<argv[5]<<"_delta"<<argv[6]<<"to"<<argv[7]<<"_"<<argv[2]<<"pointsnoreset.txt";
    pdname<<"./fss/L"<<argv[1]<<"/observables_mpi"<<NUM_WORKERS<<"_T050_JACK"<<argv[3]<<"_THERM"<<argv[4]<<"N_NUP"<<argv[5]<<"N_delta"<<argv[6]<<"to"<<argv[7]<<"_"<<argv[2]<<"pointsallreset.txt";
    //pdname<<"./fss/L"<<argv[1]<<"/observables_mpi_T050_JACK"<<argv[3]<<"_THERM"<<argv[4]<<"N_NUP"<<argv[5]<<"N_delta"<<argv[6]<<"to"<<argv[7]<<"_"<<argv[2]<<"pointsnoreset_fromextrapolatedweight.txt";
    //pdname<<"./fss/L"<<argv[1]<<"/observables_mpi_T050_JACK"<<argv[3]<<"_THERM"<<argv[4]<<"_NUP"<<argv[5]<<"_delta"<<argv[6]<<"to"<<argv[7]<<"fromextrapolatedweight.txt";
    producefile.open(pdname.str().c_str());
    producefile << "#Delta M_avg M_std Sus_avg Sus_std "<< std::endl;
    for (size_t i = 0; i < range; i++){
      producefile << delta_i + gap*(i+1)/(double)range << " " << std::setprecision(10) << avg_M[i] << " " <<std::setprecision(10)<< error_M[i]<<" "<<std::setprecision(10)<< avg_sus[i]<<" "<<std::setprecision(10)<< error_sus[i] << std::endl;
    }//not beta but need to change first delta
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

//shifting weights not to diverge the exponential valeus
void weight_shift(double beta,double delta,double* log_weight)
{
    double tmp = log_weight[0],tmp2 = log_weight[N];
    vector<double> inside_of_exp(N+1,0.0);
    for(unsigned i=0;i<N+1;i++)  {
        inside_of_exp.at(i) = -log_weight[i]-beta*delta*i;
    }
    double max = *max_element(inside_of_exp.begin(),inside_of_exp.end());

    for(unsigned i=0;i<N+1;i++){
        log_weight[i] = log_weight[i] + max;
    }
}

/*
void mcs(int* lattice,double beta, double delta,double& M1,double& M2, double& sum, my_uint64* hist, int* index, int* naver,double* boltz, double* log_weight,int* E_Delta_tmp, int* energy_tmp, unsigned WORKER,int* magnetization_tmp, unsigned iter,unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run)
{
    RNG rng;
    RNG::key_type k1 = {{WORKER, 0xdecafbad}};
    RNG::key_type k3 = {{WORKER+1000, 0x11cafead}};
    RNG::ctr_type c = {{0, seed, iter, 0xBADC0DED}};
    RNG::ctr_type r1, r3;


//for(int i = 0; i<64 ; i++) cout<< "idx : "<<index[i%N]<<endl;

    for(size_t i=0; i<N+1; i++) hist[i] = 0;
    int energy = *energy_tmp;
    int E_Delta = *E_Delta_tmp;
    int magnetization = *magnetization_tmp;


    // thermalization
    for(my_uint64 i=0; i<therm_run; i++){
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r3 = rng(c,k3);
      }

      unsigned idx = index[i%N];
      int new_s = lattice[idx] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = lattice[idx];

      new_s = new_s -3*(new_s == 2) +3*(new_s == -2); //-2 to +1 and +2 to -1
      //cout<<" E_J = "<<E_J<<" = "<<inter_energy(idx,lattice)<<endl;

      lattice[idx] = new_s*(old_s==0)+old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx])*(lattice[naver[4*idx]] + lattice[naver[4*idx+1]] + lattice[naver[4*idx+2]] + lattice[naver[4*idx+3]]), dDel = new_s*new_s - old_s*old_s, dM = new_s-old_s;
      //cout<<"dE = "<<dE<<" = "<<dE2<<endl;
      //double diff =  exp(-dE/1.4+log_weight[E_Delta+dDel]-log_weight[E_Delta]);
   
      //beta here
      double diff =  -dE*2 + log_weight[E_Delta+dDel] -log_weight[E_Delta];
      if(diff>=0){
        lattice[idx] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        lattice[idx] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else lattice[idx] = old_s;
    }


    double MxExp = 0;

    //estimating distribution with current Weight
    for(my_uint64 i=0;i<nupdates_run;i++){
      if(i%4 == 0) {
        ++c[0];
        r1 = rng(c, k1); r3 = rng(c,k3);
      }

      unsigned idx = index[i%N];
//cout<< "idx : "<<idx<<endl;
      int new_s = lattice[idx] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
      int old_s = lattice[idx];
      new_s = new_s -3*(new_s == 2) +3*(new_s == -2); //-2 to +1 and +2 to -1
      //cout<<" E_J = "<<E_J<<" = "<<inter_energy(idx,lattice)<<endl;

      lattice[idx] = new_s*(old_s==0)+old_s;
      int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx])*(lattice[naver[4*idx]] + lattice[naver[4*idx+1]] + lattice[naver[4*idx+2]] + lattice[naver[4*idx+3]]), dDel = new_s*new_s - old_s*old_s, dM = new_s-old_s;
      //cout<<"dE = "<<dE<<" = "<<dE2<<endl;
      //double diff =  exp(-dE/1.4+log_weight[E_Delta+dDel]-log_weight[E_Delta]);
      //beta here
      double diff =  -dE*2 + log_weight[E_Delta+dDel]-log_weight[E_Delta];
      if(diff>=0){
        lattice[idx] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else if(log(r123::u01fixedpt<float>(r1.v[i%4])) < diff){
        lattice[idx] = new_s;
        energy = energy + dE;
        E_Delta = E_Delta +dDel;
        magnetization = magnetization + dM;
      }
      else lattice[idx] = old_s;

      MxExp = magnetization*boltz[E_Delta];

      if(magnetization<0) M1 = M1 - MxExp;
      else M1 = M1 + MxExp;

      M2 = M2 + magnetization*MxExp;
      sum = sum + boltz[E_Delta];
    }

//cout<<"M1: "<<M1<<" M2: "<<M2<<" sum: "<<sum<<endl;

   *energy_tmp = energy;
   *E_Delta_tmp = E_Delta;
   *magnetization_tmp = magnetization; 
}
*/



void mcs(int* lattice,double beta, double delta, unsigned long* hist, int* index, int* naver,double* boltz, double* log_weight,int* E_Delta_tmp, int* energy_tmp, unsigned WORKER,int* magnetization_tmp, unsigned iter,unsigned seed, my_uint64 therm_run,my_uint64 nupdates_run)
{
    RNG rng;
    RNG::key_type k1 = {{10*WORKER, 0xdecafbad}};
    RNG::key_type k3 = {{10*WORKER+1000, 0x11cafead}};
    RNG::ctr_type c = {{0, 10000*seed, 2*iter, 0xBADC0DED}};
    RNG::ctr_type r1, r3;

    for(size_t i=0; i<N+1; i++) hist[i] = 0;
    int energy = *energy_tmp;
    int E_Delta = *E_Delta_tmp;
    int magnetization = *magnetization_tmp;


    // thermalization
    for(my_uint64 i=0; i<therm_run; i++){
        if(i%4 == 0) {
            ++c[0];
            r1 = rng(c, k1); r3 = rng(c,k3);
        }

        unsigned idx = index[i%N];
        int new_s = lattice[idx] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
        int old_s = lattice[idx];

        new_s = new_s -3*(new_s == 2) +3*(new_s == -2); //-2 to +1 and +2 to -1
        //cout<<" E_J = "<<E_J<<" = "<<inter_energy(idx,lattice)<<endl;

        lattice[idx] = new_s*(old_s==0)+old_s;
        int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx])*(lattice[naver[4*idx]] + lattice[naver[4*idx+1]] + lattice[naver[4*idx+2]] + lattice[naver[4*idx+3]]), dDel = new_s*new_s - old_s*old_s, dM = new_s-old_s;
        //cout<<"dE = "<<dE<<" = "<<dE2<<endl;
        //double diff =  exp(-dE/1.4+log_weight[E_Delta+dDel]-log_weight[E_Delta]);
   
        //beta here
        double diff =  -dE*2 + log_weight[E_Delta+dDel] -log_weight[E_Delta];
        if(diff>=0){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
            magnetization = magnetization + dM;
        }
        else if(log(r123::u01fixedpt<double>(r1.v[i%4])) < diff){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
            magnetization = magnetization + dM;
        }
        else lattice[idx] = old_s;
    }


    //estimating distribution with current Weight
    for(my_uint64 i=0;i<nupdates_run;i++){
        if(i%4 == 0) {
            ++c[0];
            r1 = rng(c, k1); r3 = rng(c,k3);
        }

        hist[(magnetization+E_Delta*E_Delta +2*E_Delta)/2] = hist[(magnetization+E_Delta*E_Delta +2*E_Delta)/2] + 1; 
        unsigned idx = index[i%N];
        //cout<< "idx : "<<idx<<endl;
        int new_s = lattice[idx] + 2*(r123::u01fixedpt<float>(r3.v[i%4])<0.5)-1;
        int old_s = lattice[idx];
        new_s = new_s -3*(new_s == 2) +3*(new_s == -2); //-2 to +1 and +2 to -1
        //cout<<" E_J = "<<E_J<<" = "<<inter_energy(idx,lattice)<<endl;

        lattice[idx] = new_s*(old_s==0)+old_s;
        int dE = ((old_s==0) + old_s*(new_s - old_s))*(-lattice[idx])*(lattice[naver[4*idx]] + lattice[naver[4*idx+1]] + lattice[naver[4*idx+2]] + lattice[naver[4*idx+3]]), dDel = new_s*new_s - old_s*old_s, dM = new_s-old_s;
        //cout<<"dE = "<<dE<<" = "<<dE2<<endl;
        //double diff =  exp(-dE/1.4+log_weight[E_Delta+dDel]-log_weight[E_Delta]);
        //beta here
        double diff =  -dE*2 + log_weight[E_Delta+dDel]-log_weight[E_Delta];
        if(diff>=0){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
            magnetization = magnetization + dM;
        }
        else if(log(r123::u01fixedpt<double>(r1.v[i%4])) < diff){
            lattice[idx] = new_s;
            energy = energy + dE;
            E_Delta = E_Delta +dDel;
            magnetization = magnetization + dM;
        }
        else lattice[idx] = old_s;

    }
    *energy_tmp = energy;
    *E_Delta_tmp = E_Delta;
    *magnetization_tmp = magnetization; 
}


void observables(double beta, double delta,unsigned N, unsigned long* h_hist,double* boltz, double* log_weight, double* M_jacks, double* sus_jacks, unsigned k)
{

  long double sum = 0, M1 = 0, M2 = 0;
  int i, m;
  for(i = 0; i <= N ; i++){
      for(m = -i; m <= 0 ; m = m+2){
          sum = sum + h_hist[(i*i+2*i+m)/2]*boltz[i];
          M1 = M1 + -m*h_hist[(i*i+2*i+m)/2]*boltz[i];
          M2 = M2 + m*m*h_hist[(i*i+2*i+m)/2]*boltz[i];
      }
      for(; m <= i ; m = m+2){
          sum = sum + h_hist[(i*i+2*i+m)/2]*boltz[i];
          M1 = M1 + m*h_hist[(i*i+2*i+m)/2]*boltz[i];
          M2 = M2 + m*m*h_hist[(i*i+2*i+m)/2]*boltz[i];
      }
  }

  M_jacks[k] = M1/sum;
  sus_jacks[k] = (M2/sum-(M1/sum)*(M1/sum))*beta/(double)N;

}

void jakknife (double* M_jacks, double* avg_M, double* error_M, int JACKS,unsigned range)
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

 avg_M[range] = avg;
 error_M[range] = dev; 
}
