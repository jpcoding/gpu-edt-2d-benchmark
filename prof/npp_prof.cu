// Minimal NPP-only driver for profiling nppiDistanceTransformPBA with nsys / ncu.
//   ./npp_prof [size=4096] [iters=20]
// Runs the 2D PBA distance transform on a device-resident random binary image (no H2D/D2H
// in the loop), so a profiler sees only the NPP kernels.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char** argv){
  int S     = argc>1 ? atoi(argv[1]) : 4096;
  int iters = argc>2 ? atoi(argv[2]) : 20;
  size_t N = (size_t)S*S;
  std::vector<unsigned char> img(N); srand(1);
  for(size_t i=0;i<N;i++) img[i] = (rand()%100==0)?0:255;   // ~1% sites (value 0)
  Npp8u*  dS;  cudaMalloc(&dS,  N);
  Npp16u* dT;  cudaMalloc(&dT,  N*sizeof(Npp16u));
  Npp16s* dV;  cudaMalloc(&dV,  N*2*sizeof(Npp16s));
  cudaMemcpy(dS, img.data(), N, cudaMemcpyHostToDevice);
  NppiSize roi={S,S}; size_t bufSz=0; nppiDistanceTransformPBAGetBufferSize(roi,&bufSz);
  Npp8u* dBuf; cudaMalloc(&dBuf,bufSz);
  NppStreamContext ctx; memset(&ctx,0,sizeof ctx);
  cudaGetDevice(&ctx.nCudaDeviceId);
  cudaDeviceProp p; cudaGetDeviceProperties(&p,ctx.nCudaDeviceId);
  ctx.nMultiProcessorCount=p.multiProcessorCount; ctx.nMaxThreadsPerMultiProcessor=p.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock=p.maxThreadsPerBlock; ctx.nSharedMemPerBlock=p.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor=p.major; ctx.nCudaDevAttrComputeCapabilityMinor=p.minor;
  auto run=[&](){ return nppiDistanceTransformPBA_8u16u_C1R_Ctx(
      dS,S,0,0, dV,S*2*sizeof(Npp16s), NULL,0, NULL,0, dT,S*sizeof(Npp16u), roi, dBuf, ctx); };
  run(); cudaDeviceSynchronize();                    // warmup
  for(int i=0;i<iters;i++) run();
  cudaDeviceSynchronize();
  printf("NPP nppiDistanceTransformPBA %dx%d x%d iters, buf=%.1f MB\n", S,S,iters, bufSz/1048576.0);
  return 0;
}
