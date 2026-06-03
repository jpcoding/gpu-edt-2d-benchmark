// Concurrency / batching benchmark: does running independent EDTs on separate CUDA
// streams recover the idle SM capacity that a single latency-bound transform leaves?
//
// Single PBA/FH transforms are latency-bound and use only a fraction of the SMs (ncu:
// ~6-32%). Aggregate throughput should therefore rise when many *independent* transforms
// overlap. We use the FH baseline because it is reentrant (all buffers are per-call); the
// NUS/NPP PBA engine keeps global device buffers, so batching it would first require making
// it stream-safe (per-instance buffers) — that refactor is the actual engineering work.
//
//   ./concurrency [M=8] [iters=10] [sizes...]   (default sizes 256 512 1024 2048)
#include <cuda_runtime.h>
#include "edt_fh.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(int argc,char**argv){
  int M     = argc>1 ? atoi(argv[1]) : 8;
  int iters = argc>2 ? atoi(argv[2]) : 10;
  std::vector<int> sizes; for(int i=3;i<argc;i++) sizes.push_back(atoi(argv[i]));
  if(sizes.empty()) sizes={256,512,1024,2048};

  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("# GPU: %s (%d SMs).  M=%d independent FH transforms, best of %d.\n", p.name,p.multiProcessorCount,M,iters);
  printf("# GiB/s = aggregate 4*M*pixels/1024^3/s (base-1024). speedup = serial_ms / concurrent_ms.\n");
  printf("%-6s %10s %10s %9s %12s %12s\n","size","serial_ms","concur_ms","speedup","serial_GiB/s","concur_GiB/s");

  for(int S : sizes){
    size_t N=(size_t)S*S;
    std::vector<char> bnd(N,0); srand(1);
    for(size_t i=0;i<N;i++) bnd[i]= (rand()%100==0)?1:0;

    std::vector<char*> db(M); std::vector<float*> A(M),B(M),dist(M); std::vector<int*> vb(M); std::vector<float*> zb(M);
    std::vector<cudaStream_t> st(M);
    for(int i=0;i<M;i++){
      cudaMalloc(&db[i],N);   cudaMemcpy(db[i],bnd.data(),N,cudaMemcpyHostToDevice);
      cudaMalloc(&A[i],N*4);  cudaMalloc(&B[i],N*4);  cudaMalloc(&dist[i],N*4);
      cudaMalloc(&vb[i],N*sizeof(int)); cudaMalloc(&zb[i],fh::fh_zb_floats(S,S)*4);
      cudaStreamCreate(&st[i]);
    }
    auto serial = [&]{ for(int i=0;i<M;i++) fh::edt_fh(db[i],dist[i],S,S,A[i],B[i],vb[i],zb[i],0); };
    auto concur = [&]{ for(int i=0;i<M;i++) fh::edt_fh(db[i],dist[i],S,S,A[i],B[i],vb[i],zb[i],st[i]); };
    serial(); cudaDeviceSynchronize(); concur(); cudaDeviceSynchronize();   // warmup

    double bs=1e30,bc=1e30;
    for(int k=0;k<iters;k++){ cudaDeviceSynchronize(); auto t=clk::now(); serial(); cudaDeviceSynchronize(); bs=std::min(bs,ms_since(t)); }
    for(int k=0;k<iters;k++){ cudaDeviceSynchronize(); auto t=clk::now(); concur(); cudaDeviceSynchronize(); bc=std::min(bc,ms_since(t)); }

    double agg=(double)M*N*4.0/(1024.0*1024.0*1024.0);
    printf("%-6d %10.4f %10.4f %9.2f %12.2f %12.2f\n", S, bs, bc, bs/bc, agg/(bs/1e3), agg/(bc/1e3));

    for(int i=0;i<M;i++){ cudaFree(db[i]);cudaFree(A[i]);cudaFree(B[i]);cudaFree(dist[i]);cudaFree(vb[i]);cudaFree(zb[i]);cudaStreamDestroy(st[i]); }
  }
  return 0;
}
