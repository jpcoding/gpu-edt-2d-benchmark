// 2D Euclidean Distance Transform / Voronoi — GPU implementations, head to head.
//   NPP              nppiDistanceTransformPBA   (NVIDIA vendor library; internally = PBA)
//   NUS-PBA+ m3=16   pba2D, tuned bands         (academic reference, MIT)
//   NUS-PBA+ m3=2    pba2D, default band        (shows the band-tuning lever)
//   ours-2D          edt_2d_pba (this project)  (native 2D EDT, our 3D-style device
//                                                interface: boundary -> index + distance)
//   ours-3Don2D      edt_3d_pba (this project)  (our 3D PBA run as W x H x 1; <=1024)
//   FH               edt_fh    (this project)   (Felzenszwalb-Huttenlocher separable EDT;
//                                                the "portable" SOTA alternative, exact)
//
// Same binary image fed to all; compute-only timing (data device-resident, warmup +
// best/median of K); every implementation cross-verified to yield the same EDT (max_err).
// Throughput: Gpix/s (pixel-count rate, base-1000) and GiB/s (4-byte float distance field
// out, base-1024).  Site density swept {1,10,50}% for synthetic; single edge map for NYX.
#include <cuda_runtime.h>
#include <npp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <fstream>
#include <algorithm>
#include "edt_pba.hpp"     // ours 3D  (edt_3d_pba, pba_buffer_size)
#include "edt_2d.hpp"      // ours native-2D (edt_2d_pba)
#include "edt_fh.hpp"      // Felzenszwalb-Huttenlocher baseline

#include "pba2D.h"        // NUS PBA+ 2D public API (init/compute/voronoi/bridges, MARKER)

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

struct Timing { double best, med; };
// setup() runs each iteration BEFORE the timer (e.g. restoring a destructive
// kernel's device input); only run() is timed. H2D transfers belong in setup.
template<class S, class F> static Timing time_it(S setup, F run, int K){
  std::vector<double> ts; ts.reserve(K);
  for(int k=0;k<K;k++){ setup(); cudaDeviceSynchronize(); auto t=clk::now(); run(); cudaDeviceSynchronize(); ts.push_back(ms_since(t)); }
  std::sort(ts.begin(),ts.end());
  return { ts.front(), ts[ts.size()/2] };
}
template<class F> static Timing time_it(F run, int K){ return time_it([]{}, run, K); }
static void row(int S,const char* name,Timing tm,size_t N,double maxerr,bool isref){
  double gpix = N/1e9/(tm.best/1e3);
  double gibs = (double)N*4.0/(1024.0*1024.0*1024.0)/(tm.best/1e3);
  if(isref) printf("%-6d %-15s %9.4f %9.4f %9.3f %9.3f %9s\n",S,name,tm.best,tm.med,gpix,gibs,"ref");
  else      printf("%-6d %-15s %9.4f %9.4f %9.3f %9.3f %9.2f\n",S,name,tm.best,tm.med,gpix,gibs,maxerr);
}

int main(int argc, char** argv){
  std::vector<int> sizes = {256,512,1024,2048,4096};
  std::vector<int> densities = {1,10,50};          // % sites (synthetic)
  bool nyx = false; const char* nyxfile=nullptr; int nyxdim=0;
  if (argc>3 && std::string(argv[1])=="nyx"){ nyx=true; nyxfile=argv[2]; nyxdim=atoi(argv[3]); sizes={nyxdim}; densities={0}; }
  else if (argc>1){ sizes.clear(); for(int i=1;i<argc;i++) sizes.push_back(atoi(argv[i])); }
  const int K = 11;     // timed iterations (best + median reported)

  NppStreamContext ctx; memset(&ctx,0,sizeof ctx);
  cudaGetDevice(&ctx.nCudaDeviceId);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop,ctx.nCudaDeviceId);
  ctx.nMultiProcessorCount=prop.multiProcessorCount; ctx.nMaxThreadsPerMultiProcessor=prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock=prop.maxThreadsPerBlock; ctx.nSharedMemPerBlock=prop.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor=prop.major; ctx.nCudaDevAttrComputeCapabilityMinor=prop.minor;

  printf("# GPU: %s   (CUDA %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  printf("# 2D exact EDT — compute-only, device-resident, best+median of %d\n", K);
  printf("# Gpix/s = pixels/1e9/s (base-1000 count) ; GiB/s = 4*pixels/1024^3/s (float dist field, base-1024)\n");

  for (int dens : densities){
   for (int S : sizes){
    size_t N=(size_t)S*S;
    std::vector<unsigned char> site(N,0); size_t nsite=0;
    if (nyx){
      std::vector<float> v(N); { std::ifstream f(nyxfile,std::ios::binary); f.read((char*)v.data(),N*4); }
      double mn=v[0],mx=v[0]; for(size_t i=0;i<N;i++){ if(v[i]<mn)mn=v[i]; if(v[i]>mx)mx=v[i]; }
      double eb=1e-2*(mx-mn), inv=1.0/(2*eb);
      std::vector<int> q(N); for(size_t i=0;i<N;i++) q[i]=(int)llround(v[i]*inv);
      for(int y=0;y<S;y++)for(int x=0;x<S;x++){ int c=q[(size_t)y*S+x]; bool b=false;
        if(x>0   && q[(size_t)y*S+x-1]!=c)b=true;  if(x<S-1 && q[(size_t)y*S+x+1]!=c)b=true;
        if(y>0   && q[(size_t)(y-1)*S+x]!=c)b=true; if(y<S-1 && q[(size_t)(y+1)*S+x]!=c)b=true;
        if(b){ site[(size_t)y*S+x]=1; nsite++; } }
    } else {
      srand(12345); for(size_t i=0;i<N;i++) if((rand()%100)<dens){ site[i]=1; nsite++; }
    }
    if(nsite==0){ site[0]=1; nsite=1; }

    if(nyx) printf("\n# === NYX slice %s  %dx%d  edge sites=%.1f%% ===\n", nyxfile,S,S,100.0*nsite/N);
    else if(S==sizes.front()) printf("\n# ===================== site density = %d%% =====================\n", dens);
    printf("%-6s %-15s %9s %9s %9s %9s %9s\n","size","impl","best_ms","med_ms","Gpix/s","GiB/s","max_err");

    std::vector<double> ref_dist;
    std::vector<char> bnd(N); for(size_t i=0;i<N;i++) bnd[i]=site[i]?1:0;

    int tex=edt_2d_texsize(S,S), eband=edt_2d_band(tex);
    pba2DInitialization(tex,eband);

    // -------- ours-3Don2D (<=1024) --------
    if (S<=1024){
      char* d_b; cudaMalloc(&d_b,N); cudaMemcpy(d_b,bnd.data(),N,cudaMemcpyHostToDevice);
      int* d_idx; cudaMalloc(&d_idx,N*sizeof(int)); float* d_dist; cudaMalloc(&d_dist,N*4);
      size_t pb=pba_buffer_size(S,S,1); int *b0,*b1; cudaMalloc(&b0,pb); cudaMalloc(&b1,pb);
      edt_3d_pba(d_b,d_idx,d_dist,S,S,1,b0,b1); cudaDeviceSynchronize();
      Timing tm=time_it([&]{ edt_3d_pba(d_b,d_idx,d_dist,S,S,1,b0,b1); },K);
      std::vector<float> d(N); cudaMemcpy(d.data(),d_dist,N*4,cudaMemcpyDeviceToHost);
      ref_dist.assign(N,0); for(size_t i=0;i<N;i++) ref_dist[i]=d[i];
      row(S,"ours-3Don2D",tm,N,0,true);
      cudaFree(d_b);cudaFree(d_idx);cudaFree(d_dist);cudaFree(b0);cudaFree(b1);
    }
    // -------- ours-2D --------
    {
      char* d_b; cudaMalloc(&d_b,N); cudaMemcpy(d_b,bnd.data(),N,cudaMemcpyHostToDevice);
      int* d_idx; cudaMalloc(&d_idx,N*sizeof(int)); float* d_dist; cudaMalloc(&d_dist,N*4);
      edt_2d_pba(d_b,d_idx,d_dist,S,S); cudaDeviceSynchronize();
      Timing tm=time_it([&]{ edt_2d_pba(d_b,d_idx,d_dist,S,S); },K);
      std::vector<float> d(N); cudaMemcpy(d.data(),d_dist,N*4,cudaMemcpyDeviceToHost);
      std::vector<double> md(N); for(size_t i=0;i<N;i++) md[i]=d[i];
      double me=0; if(ref_dist.empty()) ref_dist=md; else for(size_t i=0;i<N;i++) me=std::max(me,fabs(md[i]-ref_dist[i]));
      row(S,"ours-2D",tm,N,me,false);
      cudaFree(d_b);cudaFree(d_idx);cudaFree(d_dist);
    }
    // -------- NUS PBA+ (tuned m3=16, then default m3=2) --------
    {
      std::vector<short> in(2*N),out(2*N);
      for(size_t i=0;i<N;i++){ int x=i%S,y=i/S;
        if(site[i]){in[2*i]=(short)x;in[2*i+1]=(short)y;} else {in[2*i]=MARKER;in[2*i+1]=MARKER;} }
      int p1=eband,p2=edt_2d_m2(tex);
      for (int p3 : {edt_2d_m3(tex), 2}){
        pba2DVoronoiDiagram(in.data(),out.data(),p1,p2,p3);   // correctness pass
        // setup (H2D restore of destructive input) is untimed; only Compute is timed
        Timing tm=time_it([&]{ pba2DInitializeInput(in.data()); }, [&]{ pba2DCompute(p1,p2,p3); },K);
        std::vector<double> nd(N);
        for(size_t i=0;i<N;i++){ int x=i%S,y=i/S,nx=out[2*i],ny=out[2*i+1];
          nd[i]=sqrt(double(nx-x)*(nx-x)+double(ny-y)*(ny-y)); }
        if(ref_dist.empty()) ref_dist=nd;
        double me=0; for(size_t i=0;i<N;i++) me=std::max(me,fabs(nd[i]-ref_dist[i]));
        char nm[32]; snprintf(nm,sizeof nm,"NUS-PBA+(m3=%d)",p3);
        row(S,nm,tm,N,me,false);
      }
    }
    // -------- NVIDIA NPP --------
    {
      std::vector<unsigned char> img(N); for(size_t i=0;i<N;i++) img[i]=site[i]?0:255;
      Npp8u* dS; cudaMalloc(&dS,N); cudaMemcpy(dS,img.data(),N,cudaMemcpyHostToDevice);
      Npp16u* dT; cudaMalloc(&dT,N*sizeof(Npp16u));
      NppiSize roi={S,S}; size_t bufSz=0; nppiDistanceTransformPBAGetBufferSize(roi,&bufSz);
      Npp8u* dBuf; cudaMalloc(&dBuf,bufSz);
      auto run=[&]{ nppiDistanceTransformPBA_8u16u_C1R_Ctx(dS,S,0,0,NULL,0,NULL,0,NULL,0,dT,S*sizeof(Npp16u),roi,dBuf,ctx); };
      run(); cudaDeviceSynchronize();
      Timing tm=time_it(run,K);
      std::vector<unsigned short> tr(N); cudaMemcpy(tr.data(),dT,N*sizeof(Npp16u),cudaMemcpyDeviceToHost);
      double me=0; for(size_t i=0;i<N;i++) me=std::max(me,fabs((double)tr[i]-ref_dist[i]));
      row(S,"NPP",tm,N,me,false);
      cudaFree(dS);cudaFree(dT);cudaFree(dBuf);
    }
    // -------- Felzenszwalb-Huttenlocher (exact, portable alternative) --------
    {
      char* d_b; cudaMalloc(&d_b,N); cudaMemcpy(d_b,bnd.data(),N,cudaMemcpyHostToDevice);
      float *A,*B,*d_dist; int* vb; float* zb;
      cudaMalloc(&A,N*4); cudaMalloc(&B,N*4); cudaMalloc(&d_dist,N*4);
      cudaMalloc(&vb,N*sizeof(int)); cudaMalloc(&zb,fh::fh_zb_floats(S,S)*4);
      fh::edt_fh(d_b,d_dist,S,S,A,B,vb,zb); cudaDeviceSynchronize();
      Timing tm=time_it([&]{ fh::edt_fh(d_b,d_dist,S,S,A,B,vb,zb); },K);
      std::vector<float> d(N); cudaMemcpy(d.data(),d_dist,N*4,cudaMemcpyDeviceToHost);
      double me=0; for(size_t i=0;i<N;i++) me=std::max(me,fabs((double)d[i]-ref_dist[i]));
      row(S,"FH",tm,N,me,false);
      cudaFree(d_b);cudaFree(A);cudaFree(B);cudaFree(d_dist);cudaFree(vb);cudaFree(zb);
    }

    pba2DDeinitialization();
   }
  }
  return 0;
}
