// Minimal NUS PBA+ 2D driver for profiling with nsys / ncu.
//   ./nus_prof [size=4096] [iters=20] [m3=16]
// Runs the academic PBA+ kernels on a device-resident ~1% binary image. Input is
// restored before each Compute (Compute is destructive) but only the kernels are
// of interest; filter the profiler to the gpukernsum table. Mirrors npp_prof.cu so
// the two can be compared kernel-for-kernel (NPP internally runs these same kernels).
#include "pba2D.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv){
  int S     = argc>1 ? atoi(argv[1]) : 4096;
  int iters = argc>2 ? atoi(argv[2]) : 20;
  int m3    = argc>3 ? atoi(argv[3]) : 16;          // phase-3 band (tuned default)
  int tex=64; while(tex<S) tex<<=1;                 // PBA texture size = next pow2 (>=64)
  int m1=tex/64, m2=64;                             // tuned bands (must divide tex)
  size_t T=(size_t)tex*tex;
  std::vector<short> in(2*T, MARKER);
  srand(1);
  for(int y=0;y<S;y++) for(int x=0;x<S;x++) if(rand()%100==0){
    size_t i=(size_t)y*tex+x; in[2*i]=(short)x; in[2*i+1]=(short)y; }
  pba2DInitialization(tex, m1);
  pba2DInitializeInput(in.data()); pba2DCompute(m1,m2,m3); cudaDeviceSynchronize();   // warmup
  for(int i=0;i<iters;i++){ pba2DInitializeInput(in.data()); pba2DCompute(m1,m2,m3); }
  cudaDeviceSynchronize();
  pba2DDeinitialization();
  printf("NUS PBA+ %dx%d (tex=%d) x%d iters, bands m1=%d m2=%d m3=%d\n", S,S,tex,iters,m1,m2,m3);
  return 0;
}
