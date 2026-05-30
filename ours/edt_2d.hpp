#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
// Native 2D Euclidean Distance Transform with a device interface shaped like our 3D EDT
// (edt_3d_pba): a 1-byte boundary map (1 = site) goes in; a packed nearest-site index and a
// float distance come out — all device-resident. Drop-in for the 2D version of the pipeline.
//
// The core 2D Parallel Banding kernels are the (MIT) NUS PBA+ 2D code in third_party/nus,
// driven here through a thin device bridge — the same lineage as how our 3D EDT wraps the
// NUS 3D kernels. Coordinates are 16-bit (short2), so this has NO 1024 cap (up to 32767/axis)
// and uses a native 2-axis schedule (flood + one proximate/color), unlike the 3D-on-2D path.
//
// Output formats (match what the 3D path hands to fill_sign / compensation):
//   index[y*W+x]    = packed nearest-site index  = (ny << 16) | (nx & 0xFFFF)
//   distance[y*W+x] = exact Euclidean distance to the nearest site

#define EDT2D_MARKER (-32768)
extern "C" void    pba2DInitialization(int textureSize, int phase1Band);
extern "C" void    pba2DDeinitialization();
void               pba2DCompute(int m1, int m2, int m3);   // C++ linkage in pba2DHost.cu
extern "C" short2* pba2DInputDevice();
extern "C" short2* pba2DOutputDevice();

__global__ void edt2d_fill_input(const char* boundary, short2* tex, int W, int H, int size) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= size || y >= size) return;
  short2 v;
  if (x < W && y < H && boundary[(size_t)y*W + x] == (char)1) { v.x = (short)x; v.y = (short)y; }
  else { v.x = EDT2D_MARKER; v.y = EDT2D_MARKER; }
  tex[(size_t)y*size + x] = v;                       // square texture, padding = MARKER
}

__global__ void edt2d_extract(const short2* tex, int W, int H, int size,
                              unsigned int* index, float* distance) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  short2 s = tex[(size_t)y*size + x];
  float dx = (float)s.x - x, dy = (float)s.y - y;
  size_t o = (size_t)y*W + x;
  distance[o] = sqrtf(dx*dx + dy*dy);
  index[o] = ((unsigned int)(unsigned short)s.y << 16) | (unsigned int)(unsigned short)s.x;
}

inline int edt2d_next_pow2(int v) { int s = 64; while (s < v) s <<= 1; return s; }  // PBA wants pow2 square
inline int edt_2d_texsize(unsigned int W, unsigned int H) { return edt2d_next_pow2((int)(W > H ? W : H)); }
inline int edt_2d_band(int size) { int p = size/64; return p < 1 ? 1 : p; }   // phase1 (=margin alloc)
// Tuned phase-2/3 bands (5090 sweep, correctness-verified): phase3 block is (64, m3) so m3<=16;
// m3=16 (vs the common default 2) keeps kernelColor at full 1024-thread occupancy -> ~1.5x overall.
inline int edt_2d_m2(int size) { return size >= 64 ? 64 : size; }
inline int edt_2d_m3(int size) { return size >= 16 ? 16 : size; }

// Native-2D EDT. Same I/O contract as edt_3d_pba (boundary -> index + distance, device).
// The engine (square textures) must be initialized once by the caller:
//   int tex = edt_2d_texsize(W,H);  pba2DInitialization(tex, edt_2d_band(tex));   ... pba2DDeinitialization();
// This keeps it stateless and lets it share the engine with a direct NUS run in the same process.
inline void edt_2d_pba(char* d_boundary, int* index, float* distance, unsigned int W, unsigned int H) {
  int size = edt_2d_texsize(W,H);
  int p1 = edt_2d_band(size), p2 = edt_2d_m2(size), p3 = edt_2d_m3(size);
  dim3 b(16,16);
  { dim3 g((size+15)/16,(size+15)/16); edt2d_fill_input<<<g,b>>>(d_boundary, pba2DInputDevice(), W,H,size); }
  pba2DCompute(p1, p2, p3);
  { dim3 g((W+15)/16,(H+15)/16); edt2d_extract<<<g,b>>>(pba2DOutputDevice(), W,H,size,
                                                        (unsigned int*)index, distance); }
}
