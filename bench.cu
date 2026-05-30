// 2D Euclidean Distance Transform / Voronoi — three GPU implementations, head to head.
//   (1) NVIDIA NPP   nppiDistanceTransformPBA   (2D-native, vendor library)
//   (2) NUS  PBA+    pba2D                       (2D-native, academic reference, MIT)
//   (3) "ours"       edt_3d_pba (this project)   (3D-native PBA, run as W x H x 1)
//
// Same random binary image fed to all three; compute-only timing (data device-resident,
// warmup + best-of-K); cross-verified that every implementation yields the same EDT.
//
// NOTE on "ours": it is a 3D PBA. For a 2D image it pads depth 1 -> 4 (a 4x cell tax the
// native-2D codes don't pay) and caps each in-plane dim at 1024 (10-bit packed coords).
// So it is included only for sizes <= 1024 and is expected to trail the native-2D codes.
#include <cuda_runtime.h>
#include <npp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include "edt_pba.hpp"   // ours (PBA_ prefixed; provides edt_3d_pba, pba_buffer_size)

// ---- NUS PBA+ 2D API (third_party/nus/pba2DHost.cu) ----
extern "C" void pba2DInitialization(int textureSize, int phase1Band);
extern "C" void pba2DDeinitialization();
extern "C" void pba2DVoronoiDiagram(short* input, short* output, int m1, int m2, int m3);
void pba2DInitializeInput(short* input);
void pba2DCompute(int m1, int m2, int m3);
#define NUS_MARKER (-32768)

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

int main(int argc, char** argv){
  std::vector<int> sizes = {256,512,1024,2048,4096};
  if (argc > 1){ sizes.clear(); for(int i=1;i<argc;i++) sizes.push_back(atoi(argv[i])); }
  const int K = 10;            // timed iterations (report best)
  const int site_pct = 1;      // ~1% of pixels are sites

  // NPP stream context
  NppStreamContext ctx; memset(&ctx,0,sizeof ctx);
  cudaGetDevice(&ctx.nCudaDeviceId);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop,ctx.nCudaDeviceId);
  ctx.nMultiProcessorCount=prop.multiProcessorCount; ctx.nMaxThreadsPerMultiProcessor=prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock=prop.maxThreadsPerBlock; ctx.nSharedMemPerBlock=prop.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor=prop.major; ctx.nCudaDevAttrComputeCapabilityMinor=prop.minor;

  printf("# GPU: %s\n", prop.name);
  printf("# 2D EDT benchmark — ~%d%% random sites, best of %d (compute-only, device-resident)\n#\n", site_pct, K);
  printf("%-6s %-12s %10s %12s %12s %10s\n","size","impl","time_ms","Mpix/s","Gpix/s","max_err");
  printf("%-6s %-12s %10s %12s %12s %10s\n","----","----","-------","------","------","-------");

  for (int S : sizes){
    size_t N = (size_t)S*S;
    // sites: 1 = site
    std::vector<unsigned char> site(N,0);
    srand(12345);
    size_t nsite=0; for(size_t i=0;i<N;i++){ if(rand()%100 < site_pct){ site[i]=1; nsite++; } }
    if(nsite==0){ site[0]=1; nsite=1; }

    // reference EDT (exact, double) computed by "ours" if available, else NUS
    std::vector<double> ref_dist;  // filled below

    cudaEvent_t ev0,ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    // ============================ OURS (<=1024) ============================
    bool ours_ok = (S <= 1024);
    if (ours_ok){
      std::vector<char> bnd(N); for(size_t i=0;i<N;i++) bnd[i]=site[i]?1:0;
      char* d_b; cudaMalloc(&d_b,N); cudaMemcpy(d_b,bnd.data(),N,cudaMemcpyHostToDevice);
      int* d_idx; cudaMalloc(&d_idx,N*sizeof(int));
      float* d_dist; cudaMalloc(&d_dist,N*sizeof(float));
      size_t pb=pba_buffer_size(S,S,1); int *b0,*b1; cudaMalloc(&b0,pb); cudaMalloc(&b1,pb);
      edt_3d_pba(d_b,d_idx,d_dist,S,S,1,b0,b1); cudaDeviceSynchronize(); // warmup
      double best=1e30;
      for(int k=0;k<K;k++){ cudaDeviceSynchronize(); auto t=clk::now();
        edt_3d_pba(d_b,d_idx,d_dist,S,S,1,b0,b1); cudaDeviceSynchronize();
        best=std::min(best,ms_since(t)); }
      std::vector<float> dist(N); cudaMemcpy(dist.data(),d_dist,N*4,cudaMemcpyDeviceToHost);
      ref_dist.assign(N,0); for(size_t i=0;i<N;i++) ref_dist[i]=dist[i];
      printf("%-6d %-12s %10.4f %12.1f %12.3f %10s\n", S,"ours",best, N/1e6/(best/1e3), N/1e9/(best/1e3), "ref");
      cudaFree(d_b); cudaFree(d_idx); cudaFree(d_dist); cudaFree(b0); cudaFree(b1);
    }

    // ============================ NUS PBA+ 2D ============================
    {
      int p1 = std::max(1, S/64), p2 = std::max(1, S/64), p3 = 2;   // p1<=S/64, must divide S
      std::vector<short> in(2*N), out(2*N);
      for(size_t i=0;i<N;i++){ int x=i%S, y=i/S;
        if(site[i]){ in[2*i]=(short)x; in[2*i+1]=(short)y; } else { in[2*i]=NUS_MARKER; in[2*i+1]=NUS_MARKER; } }
      pba2DInitialization(S, p1);
      pba2DVoronoiDiagram(in.data(), out.data(), p1,p2,p3);        // correctness pass
      // timed compute-only (re-init input each iter; H2D untimed)
      double best=1e30;
      for(int k=0;k<K;k++){ pba2DInitializeInput(in.data()); cudaDeviceSynchronize();
        auto t=clk::now(); pba2DCompute(p1,p2,p3); cudaDeviceSynchronize(); best=std::min(best,ms_since(t)); }
      // verify vs ref
      double maxerr=0;
      if(!ref_dist.empty()){ ref_dist.size(); }
      std::vector<double> nus_dist(N);
      for(size_t i=0;i<N;i++){ int x=i%S,y=i/S; int nx=out[2*i],ny=out[2*i+1];
        nus_dist[i]=sqrt(double(nx-x)*(nx-x)+double(ny-y)*(ny-y)); }
      if(ref_dist.empty()) ref_dist=nus_dist;   // NUS becomes ref when ours absent (S>1024)
      for(size_t i=0;i<N;i++) maxerr=std::max(maxerr, fabs(nus_dist[i]-ref_dist[i]));
      pba2DDeinitialization();
      printf("%-6d %-12s %10.4f %12.1f %12.3f %10.2f\n", S,"NUS-PBA+",best, N/1e6/(best/1e3), N/1e9/(best/1e3), maxerr);
    }

    // ============================ NVIDIA NPP ============================
    {
      std::vector<unsigned char> img(N); for(size_t i=0;i<N;i++) img[i]=site[i]?0:255; // sites in [0,0]
      Npp8u* dS; cudaMalloc(&dS,N); cudaMemcpy(dS,img.data(),N,cudaMemcpyHostToDevice);
      Npp16u* dT; cudaMalloc(&dT,N*sizeof(Npp16u));
      NppiSize roi={S,S}; size_t bufSz=0; nppiDistanceTransformPBAGetBufferSize(roi,&bufSz);
      Npp8u* dBuf; cudaMalloc(&dBuf,bufSz);
      auto run=[&](){ return nppiDistanceTransformPBA_8u16u_C1R_Ctx(dS,S,0,0,NULL,0,NULL,0,NULL,0,dT,S*sizeof(Npp16u),roi,dBuf,ctx); };
      run(); cudaDeviceSynchronize(); // warmup
      double best=1e30;
      for(int k=0;k<K;k++){ cudaDeviceSynchronize(); auto t=clk::now(); run(); cudaDeviceSynchronize(); best=std::min(best,ms_since(t)); }
      std::vector<unsigned short> tr(N); cudaMemcpy(tr.data(),dT,N*sizeof(Npp16u),cudaMemcpyDeviceToHost);
      double maxerr=0; for(size_t i=0;i<N;i++) maxerr=std::max(maxerr, fabs((double)tr[i]-ref_dist[i]));
      printf("%-6d %-12s %10.4f %12.1f %12.3f %10.2f\n", S,"NPP",best, N/1e6/(best/1e3), N/1e9/(best/1e3), maxerr);
      cudaFree(dS); cudaFree(dT); cudaFree(dBuf);
    }
    printf("#\n");
  }
  return 0;
}
