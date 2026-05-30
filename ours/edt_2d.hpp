#pragma once
#include "edt_pba.hpp"
// Native-2D specialization of our 3D Parallel Banding code.
//
// The 3D entry (edt_3d_pba) treats a W x H image as a W x H x 1 volume, but its axis
// chooser rounds the depth up to a multiple of 4 -> it processes 4x the cells. Here we
// drive the exact same kernels with z_size = 1 (no depth padding), so a 2D image costs
// one z-plane instead of four. The two Maurer/Color passes are the real X and Y axes;
// the flood-Z pass is trivial (a 1-element column). Output is the exact 2D EDT.
//
// Coordinates are still packed in 10 bits => max 1024 per axis (same as the 3D code).

inline size_t pba2d_buffer_size(uint W, uint H) {
  int xy = pba_next_mult((W > H ? W : H), 32); if (xy < 32) xy = 32;
  return (size_t)xy * xy * 1 * sizeof(int);   // z_size = 1
}

// d_buf0/d_buf1: two device buffers each >= pba2d_buffer_size(W,H) bytes.
inline void edt_2d_pba(char* d_boundary, int* index, float* distance,
                       uint W, uint H, int* d_buf0, int* d_buf1) {
  int xy_size = pba_next_mult((W > H ? W : H), 32); if (xy_size < 32) xy_size = 32;
  const int z_size = 1, z_axis = 2;
  if (xy_size > 1024) { printf("[edt_2d_pba] ERROR: xy_size=%d > 1024 (10-bit coord limit)\n", xy_size); return; }

  int* d_buf[2] = {d_buf0, d_buf1};
  { dim3 block(8,8,8); dim3 grid((xy_size+7)/8,(xy_size+7)/8,1);
    pba_init_from_boundary<<<grid,block>>>(d_boundary, d_buf[0], W,H,1, xy_size, z_size, z_axis); }
  int cur = 0;
  { dim3 block(PBA_BLOCKX,PBA_BLOCKY);
    dim3 grid((xy_size+PBA_BLOCKX-1)/PBA_BLOCKX,(xy_size+PBA_BLOCKY-1)/PBA_BLOCKY);
    pba_kernelFloodZ<<<grid,block>>>(d_buf[cur], d_buf[1-cur], xy_size, z_size); cur = 1-cur; }
  for (int pass = 0; pass < 2; ++pass) {   // Maurer(Y/X) + Color, twice
    { dim3 block(PBA_BLOCKX,PBA_BLOCKY);
      dim3 grid((xy_size+PBA_BLOCKX-1)/PBA_BLOCKX,(z_size+PBA_BLOCKY-1)/PBA_BLOCKY);
      pba_kernelMaurerAxis<<<grid,block>>>(d_buf[cur], d_buf[1-cur], xy_size, z_size); }
    { dim3 block(PBA_BLOCKSIZE,2); dim3 grid(xy_size/PBA_BLOCKSIZE, z_size);
      pba_kernelColorAxis<<<grid,block>>>(d_buf[1-cur], d_buf[cur], xy_size, z_size); }
  }
  { dim3 block(64,4,2); dim3 grid((W+63)/64,(H+3)/4,1);
    pba_extract_result<<<grid,block>>>(d_buf[cur], (unsigned int*)index, distance, W,H,1, xy_size, z_size, z_axis); }
}
