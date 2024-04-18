#include<iostream>
#include <stdlib.h>
#include <malloc.h>
#include "omp.h"
#include "./solver/origin.h"
#include <fstream>

#include "testrun_gen.h"
using namespace std;
#define ALIGNMENT 4096


#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif


int main(){
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_INIT;
#endif

    cout<<"\n\nstart     ##############################"<<endl;
    int Asize = (uSx*uNx+uNw-1)*(uSy*uNy+uNh-1)*uNc;
    int Bsize = uNf*uNc*uNw*uNh;
    int Csize = uNx*uNy*uNf;
    int iter = 1;
    // cout<<"uNc="<<uNc<<"  uNw="<<uNw<<"  uNh="<<uNh<<"  uNf="<<uNf<<"  uNx="<<uNx<<"  uNy="<<uNy<<"  uSx="<<uSx<<"  uSy="<<uSy;
    // cout<<"\nAsize="<<Asize<<"  Bsize="<<Bsize<<"  Csize="<<Csize<<"    sizeof(float)="<<sizeof(float);
    // float* A = (float*)malloc(Asize * sizeof(float));
    // float* postB = (float*)malloc(Bsize* sizeof(float));
    // float* B = (float*)malloc(Bsize* sizeof(float));
    // float* C = (float*)malloc(Csize * sizeof(float));
    // float* C2 = (float*)malloc(Csize * sizeof(float));


    float* A = (float*)memalign(4096, (iter+1)*Asize * sizeof(float));
    float* postB = (float*)memalign(4096, (iter+1)*Bsize* sizeof(float));
    float* B = (float*)memalign(4096, (iter+1)*Bsize* sizeof(float));
    float* C = (float*)memalign(4096, (iter+1)*Csize * sizeof(float));
    float* C2 = (float*)memalign(4096, Csize * sizeof(float));
    // cout<<"\nA="<<(iter+1)*Asize * sizeof(float)<<" postB="<<(iter+1)*Bsize * sizeof(float)<<" B="<<(iter+1)*Bsize * sizeof(float)<<" C="<<(iter+1)*Csize * sizeof(float)<<endl;
    
    // rand()%10 optimize --->  https://zhidao.baidu.com/question/152763941.html
    for(int i = 0; i < (iter+1)*Asize; i++) A[i] = rand()%10;
    for(int i = 0; i < (iter+1)*Bsize; i++) B[i] = rand()%10;
    for(int i = 0; i < (iter+1)*Csize; i++) C[i] = 0;
    for(int i = 0; i < Csize; i++) C2[i] = C[i];


#ifdef LIKWID_PERFMON
    LIKWID_MARKER_THREADINIT;
#endif
    
#ifdef LIKWID_PERFMO
    LIKWID_MARKER_START("Compute");
#endif
    // cout<<"\nT00001    ##############################"<<endl;
    // cout<<"testrun(A, postB, C, B)=( "<<8*Asize<<","<<8*Bsize<<","<<8*Csize<<","<<8*Bsize<<" )"<<"     8*size*8bit"<<endl;
    testrun(A, postB, C, B);

#ifdef LIKWID_PERFMON
	LIKWID_MARKER_STOP("Compute");
#endif
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
  return 5;
#endif
   
    float sum_time;
    double opamount;
    float ressss;
    float tttmp[8];
    int flushsz=100000000;
    

    cout<< "Problem: bfxycwh = 1," + to_string(uNf) + ", "+to_string(uNx) + ", "+ to_string(uNy) + ", "+ to_string(uNc) + ", "+ to_string(uNw) + ", "+ to_string(uNh) + ", "<<endl;;
    // cout<<"\n";
    vector<float> gflops;
    vector<float> times;
    //int totiter = 50; //run 50 times
    int totiter = 100;
    for (int it = 0; it < totiter; it++)
    {
        float *dirty = (float *)malloc(flushsz * sizeof(float));
        #pragma omp parallel for
        for (int dirt = 0; dirt < flushsz; dirt++){ 
            dirty[dirt] += dirt%100;
            tttmp[dirt%8] += dirty[dirt];
        }
        for(int ii =0; ii<8;ii++){ressss+= tttmp[ii];}
        // cout<<"\nflush="<<ressss<<"  "<<"start";

        double tstart = omp_get_wtime();
        //cout<<"A+(1)*Asize="<<A+(1)*Asize<<"  postB+(1)*Bsize="<<postB+(1)*Bsize<<"  C+(1)*Csize="<<C+(1)*Csize<<"  B+(1)*Bsize)="<<B+(1)*Bsize<<endl;
        // cout<<"\n\nT00002     ###############################"<<endl;
        // cout<<"testrun(A+(1)*Asize, postB+(1)*Bsize, C+(1)*Csize, B+(1)*Bsize)=( "<<8*Asize+(1)*Asize<<","<<8*Bsize+(1)*Bsize<<","<<8*Csize+(1)*Csize<<","<<8*Bsize+(1)*Bsize<<" )"<<"     9*size*8bit"<<endl;
        testrun(A+(1)*Asize, postB+(1)*Bsize, C+(1)*Csize, B+(1)*Bsize);

        double tend = omp_get_wtime();
        free (dirty);
//    cout<<"generated run "+ std::to_string(iter)+ " times, ";
        opamount = 2.0*uNf*uNc*uNw*uNh*uNx*uNy*iter;
        double gflop = opamount/(tend-tstart)/1000/1000/1000;
        cout<<"opamount="<<opamount<<"  time="<<tend-tstart<<"  gflop="<<gflop<<endl;
        gflops.push_back(gflop);
        times.push_back(tend-tstart);
    }

    double average = 0;
    //cout<<"\n\nprint flops :"<<endl;
    for (auto i : gflops){
    //    cout<<i<<"  ";
        average += i;
        //#####################################
        // ofstream f;    //定义输出流对象
        // f.open("./files_temporary/gflops.py",ios::out|ios::app);  //打开文件
        // f << i <<",";  //向文件中写入数据
        // f.close(); 
        //##################################################################
    }
    for (auto i : times){
        sum_time += i;
    }
    cout<<"\n\ntotiter = "<<totiter;
    cout<<"   opamount = "<<opamount;
    cout<<"   avg time = "<<sum_time/totiter;
    cout<<"   avg flops = "<<average/totiter<<endl;

    //////////////////////////////////////////////////////////////////////////////////
    ofstream f;    //定义输出流对象
    f.open("./files_temporary/data_out_cpp.txt",ios::out|ios::app);  //打开文件
    f << average/totiter <<" ";  //向文件中写入数据
    f.close();    //关闭文件
    ////////////////////////////////////////////////////////////////////////////////
    //cout<<"time = "<<(tend-tstart)<<", gflop = "<<gflop<<endl;
    return 0;
}