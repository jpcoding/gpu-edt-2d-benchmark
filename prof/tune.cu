#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
extern "C" void pba2DInitialization(int,int);
extern "C" void pba2DDeinitialization();
extern "C" void pba2DVoronoiDiagram(short*,short*,int,int,int);
void pba2DInitializeInput(short*);
void pba2DCompute(int,int,int);
using clk=std::chrono::high_resolution_clock;
static double msd(clk::time_point t){return std::chrono::duration<double,std::milli>(clk::now()-t).count();}
int main(int argc,char**argv){
  int S=argc>1?atoi(argv[1]):4096; size_t N=(size_t)S*S;
  std::vector<short> in(2*N),out(2*N); srand(1);
  for(size_t i=0;i<N;i++){int x=i%S,y=i/S; if(rand()%100==0){in[2*i]=(short)x;in[2*i+1]=(short)y;}else{in[2*i]=-32768;in[2*i+1]=-32768;}}
  auto distmax=[&](std::vector<short>&o,std::vector<double>&ref)->double{double m=0;for(size_t i=0;i<N;i++){int x=i%S,y=i/S;double d=hypot((double)o[2*i]-x,(double)o[2*i+1]-y);m=std::max(m,fabs(d-ref[i]));}return m;};
  int d1=S/64; // default m1=m2
  // reference with default bands
  pba2DInitialization(S,d1); pba2DVoronoiDiagram(in.data(),out.data(),d1,d1,2);
  std::vector<double> ref(N); for(size_t i=0;i<N;i++){int x=i%S,y=i/S; ref[i]=hypot((double)out[2*i]-x,(double)out[2*i+1]-y);}
  double bd=1e30; for(int k=0;k<10;k++){pba2DInitializeInput(in.data());cudaDeviceSynchronize();auto t=clk::now();pba2DCompute(d1,d1,2);cudaDeviceSynchronize();double e=msd(t);if(e<bd)bd=e;}
  pba2DDeinitialization();
  printf("S=%d  default m1=m2=%d m3=2 : %.4f ms (%.2f Gpix/s)\n",S,d1,bd,N/1e9/(bd/1e3));
  int m1s[]={2,4,8,16,32,64}, m3s[]={1,2,4,8,16};
  double bestT=1e30; int bm1=0,bm2=0,bm3=0;
  for(int m1:m1s) for(int m2:m1s) for(int m3:m3s){
    if(m1>S/64) continue;                         // m1 <= S/64
    pba2DInitialization(S,m1);
    pba2DVoronoiDiagram(in.data(),out.data(),m1,m2,m3);
    double err=distmax(out,ref);
    if(err>0.5){ pba2DDeinitialization(); continue; }   // wrong result -> reject
    double best=1e30;
    for(int k=0;k<10;k++){pba2DInitializeInput(in.data());cudaDeviceSynchronize();auto t=clk::now();pba2DCompute(m1,m2,m3);cudaDeviceSynchronize();double e=msd(t);if(e<best)best=e;}
    pba2DDeinitialization();
    if(best<bestT){bestT=best;bm1=m1;bm2=m2;bm3=m3;}
  }
  printf("        BEST(valid) m1=%d m2=%d m3=%d : %.4f ms (%.2f Gpix/s)  speedup %.2fx\n",bm1,bm2,bm3,bestT,N/1e9/(bestT/1e3),bd/bestT);
  return 0;
}
