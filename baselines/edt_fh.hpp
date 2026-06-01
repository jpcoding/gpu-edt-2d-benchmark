#ifndef EDT_FH_HPP
#define EDT_FH_HPP
// ============================================================================
// Felzenszwalb-Huttenlocher exact 2D Euclidean Distance Transform, GPU.
//
// The separable O(N) lower-envelope-of-parabolas algorithm
//   ("Distance Transforms of Sampled Functions", Felzenszwalb & Huttenlocher,
//    Theory of Computing 2012).
// This is the algorithm used by the "modern, portable" GPU EDT libraries
// (e.g. DistanceTransforms.jl, IEEE Access 2025) because it is simple and maps
// to any vendor. We implement it here as a *competently coalesced* CUDA baseline
// so PBA can be compared against the portable alternative on the same GPU.
//
// Pass structure (each pass is a 1D squared DT along a contiguous-stride axis,
// transposes in between to keep all global accesses coalesced):
//   init -> col-DT(y) -> transpose -> col-DT(x) -> transpose+sqrt -> distance
//
// Exact in integer arithmetic; computed in float, so squared distances above
// 2^24 (~size 4096) can carry sub-pixel float rounding (reported as max_err).
// Output: float Euclidean distance, same layout as the other implementations.
// MIT (this benchmark).
// ============================================================================
#include <cuda_runtime.h>

namespace fh {

#define FH_INF 1e20f
#define FH_TILE 32

__global__ void fh_init(const char* b, float* f, int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) f[i] = b[i] ? 0.f : FH_INF;
}

// 1D squared distance transform along the height axis (stride = width).
// Thread = contiguous column index in [0,width); processes a line of length
// height. Per-thread envelope scratch v[]/z[] is laid out [k*width + col] so
// that, for a fixed k, the whole warp's accesses are coalesced.
// src and dst are distinct buffers (the second sweep reads src at apex points,
// so in-place would be a hazard).
__global__ void fh_pass(const float* __restrict__ src, float* __restrict__ dst,
                        int width, int height, int* __restrict__ vb, float* __restrict__ zb){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if(col>=width) return;
  const int n = height;
  int k = 0;
  vb[(size_t)0*width+col] = 0;
  zb[(size_t)0*width+col] = -FH_INF;
  zb[(size_t)1*width+col] =  FH_INF;
  for(int q=1; q<n; q++){
    float fq = src[(size_t)q*width+col];
    float s;
    for(;;){
      int vk = vb[(size_t)k*width+col];
      float fvk = src[(size_t)vk*width+col];
      s = ((fq + (float)q*q) - (fvk + (float)vk*vk)) / (float)(2*(q-vk));
      if(s > zb[(size_t)k*width+col]) break;
      k--;
    }
    k++;
    vb[(size_t)k*width+col] = q;
    zb[(size_t)k*width+col] = s;
    zb[(size_t)(k+1)*width+col] = FH_INF;
  }
  k = 0;
  for(int q=0; q<n; q++){
    while(zb[(size_t)(k+1)*width+col] < (float)q) k++;
    int vk = vb[(size_t)k*width+col];
    float fvk = src[(size_t)vk*width+col];
    dst[(size_t)q*width+col] = (float)(q-vk)*(q-vk) + fvk;
  }
}

// Tiled, bank-conflict-free transpose. in: (height rows x width cols),
// out: (width rows x height cols). SQRT=true also takes the square root
// (used on the final pass to emit Euclidean distance).
template<bool SQRT>
__global__ void fh_transpose(const float* __restrict__ in, float* __restrict__ out,
                             int width, int height){
  __shared__ float t[FH_TILE][FH_TILE+1];
  int x = blockIdx.x*FH_TILE + threadIdx.x;   // col in [0,width)
  int y = blockIdx.y*FH_TILE + threadIdx.y;   // row in [0,height)
  if(x<width && y<height) t[threadIdx.y][threadIdx.x] = in[(size_t)y*width + x];
  __syncthreads();
  int ox = blockIdx.y*FH_TILE + threadIdx.x;  // out col in [0,height)
  int oy = blockIdx.x*FH_TILE + threadIdx.y;  // out row in [0,width)
  if(ox<height && oy<width){
    float v = t[threadIdx.x][threadIdx.y];
    out[(size_t)oy*height + ox] = SQRT ? sqrtf(v) : v;
  }
}

// Pre-allocated scratch so timing is compute-only:
//   A,B : float[W*H] each   vb : int[W*H]   zb : float[W*H + max(W,H)]
inline size_t fh_scratch_floats(int W,int H){ return (size_t)W*H; }      // A or B
inline size_t fh_zb_floats(int W,int H){ return (size_t)W*H + (W>H?W:H); }

inline void edt_fh(const char* d_b, float* d_dist, int W, int H,
                   float* A, float* B, int* vb, float* zb){
  int N = W*H;
  fh_init<<<(N+255)/256,256>>>(d_b, A, N);
  // pass 1: DT along y (width=W, height=H)        A -> B
  fh_pass<<<(W+127)/128,128>>>(A, B, W, H, vb, zb);
  // transpose B(W x H) -> A(now H-wide)
  dim3 thr(FH_TILE,FH_TILE);
  dim3 g1((W+FH_TILE-1)/FH_TILE,(H+FH_TILE-1)/FH_TILE);
  fh_transpose<false><<<g1,thr>>>(B, A, W, H);
  // pass 2: DT along x (width=H, height=W)         A -> B
  fh_pass<<<(H+127)/128,128>>>(A, B, H, W, vb, zb);
  // transpose + sqrt  B -> d_dist (W x H)
  dim3 g2((H+FH_TILE-1)/FH_TILE,(W+FH_TILE-1)/FH_TILE);
  fh_transpose<true><<<g2,thr>>>(B, d_dist, H, W);
}

} // namespace fh
#endif
