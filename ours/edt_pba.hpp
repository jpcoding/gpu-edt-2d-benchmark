#ifndef EDT_PBA_HPP
#define EDT_PBA_HPP

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// PBA+ (Parallel Banding Algorithm Plus) for 3D Euclidean Distance Transform
//
// Ported from: https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus
// License: MIT (Cao Thanh Tung, Zheng Jiaqi, NUS 2019)
//
// Non-cubic support: PBA+ originally requires cubic power-of-2 volumes.
// This version uses separate (xy_size, z_size) dimensions:
//   - xy_size: shared by PBA X and Y axes (they transpose), mult of 32
//   - z_size:  PBA Z axis, must be multiple of 4
// Automatic axis remapping picks which original dim maps to PBA_z to
// minimize total padded voxels (xy_size² × z_size).
//
// Max dimension per axis: 1024 (10-bit coordinate encoding).
// ============================================================================

// --- PBA macros ---
#define PBA_MARKER      (-2147483648)
#define PBA_MAX_INT     2147483647
#define PBA_ENCODE(x, y, z, a, b)  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))
#define PBA_DECODE(value, x, y, z) \
    x = ((value) >> 20) & 0x3ff; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff
#define PBA_NOTSITE(value)  (((value) >> 31) & 1)
#define PBA_HASNEXT(value)  (((value) >> 30) & 1)
#define PBA_GET_X(value)    (((value) >> 20) & 0x3ff)
#define PBA_GET_Y(value)    (((value) >> 10) & 0x3ff)
#define PBA_GET_Z(value)    ((PBA_NOTSITE((value))) ? PBA_MAX_INT : ((value) & 0x3ff))

// Index into non-cubic PBA buffer: xy_size used for both X and Y strides
#define PBA_IDX(x, y, z, xy_sz) ((((z) * (xy_sz)) + (y)) * (xy_sz) + (x))

#define PBA_BLOCKX      32
#define PBA_BLOCKY      4
#define PBA_BLOCKSIZE   32
#define PBA_JFA_INVALID 0xFFFFFFFFu

// ============================================================================
// PBA+ Kernels — modified for non-cubic (xy_size, z_size)
// ============================================================================

// Phase 1: Flood along Z axis (sweep z_size elements per (x,y) column)
__global__ void pba_kernelFloodZ(int *input, int *output, int xy_size, int z_size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= xy_size || ty >= xy_size) return;

    int plane = xy_size * xy_size;
    int id = ty * xy_size + tx;  // PBA_IDX(tx, ty, 0, xy_size)
    int pixel1, pixel2;

    pixel1 = PBA_ENCODE(0,0,0,1,0);

    for (int i = 0; i < z_size; i++, id += plane) {
        pixel2 = input[id];
        if (!PBA_NOTSITE(pixel2))
            pixel1 = pixel2;
        output[id] = pixel1;
    }

    int dist1, dist2, nz;
    id -= plane + plane;

    for (int i = z_size - 2; i >= 0; i--, id -= plane) {
        nz = PBA_GET_Z(pixel1);
        dist1 = abs(nz - i);
        pixel2 = output[id];
        nz = PBA_GET_Z(pixel2);
        dist2 = abs(nz - i);
        if (dist2 < dist1)
            pixel1 = pixel2;
        output[id] = pixel1;
    }
}

#define PBA_LL long long
__device__ bool pba_dominate(PBA_LL x_1, PBA_LL y_1, PBA_LL z_1,
                             PBA_LL x_2, PBA_LL y_2, PBA_LL z_2,
                             PBA_LL x_3, PBA_LL y_3, PBA_LL z_3,
                             PBA_LL x_0, PBA_LL z_0)
{
    PBA_LL k_1 = y_2 - y_1, k_2 = y_3 - y_2;
    return (((y_1 + y_2) * k_1 + ((x_2 - x_1) * (x_1 + x_2 - (x_0 << 1)) + (z_2 - z_1) * (z_1 + z_2 - (z_0 << 1)))) * k_2 >
            ((y_2 + y_3) * k_2 + ((x_3 - x_2) * (x_2 + x_3 - (x_0 << 1)) + (z_3 - z_2) * (z_2 + z_3 - (z_0 << 1)))) * k_1);
}
#undef PBA_LL

// Phase 2: Maurer axis (sweep xy_size elements along Y for each (x,z))
__global__ void pba_kernelMaurerAxis(int *input, int *stack, int xy_size, int z_size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tz = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= xy_size || tz >= z_size) return;

    int id = PBA_IDX(tx, 0, tz, xy_size);
    int lasty = 0;
    int x1, y1, z1, x2, y2, z2, nx, ny, nz;
    int p = PBA_ENCODE(0,0,0,1,0), s1 = PBA_ENCODE(0,0,0,1,0), s2 = PBA_ENCODE(0,0,0,1,0);
    int flag = 0;

    for (int ty = 0; ty < xy_size; ++ty, id += xy_size) {
        p = input[id];
        if (!PBA_NOTSITE(p)) {
            while (PBA_HASNEXT(s2)) {
                PBA_DECODE(s1, x1, y1, z1);
                PBA_DECODE(s2, x2, y2, z2);
                PBA_DECODE(p, nx, ny, nz);
                if (!pba_dominate(x1, y2, z1, x2, lasty, z2, nx, ty, nz, tx, tz))
                    break;
                lasty = y2; s2 = s1; y2 = y1;
                if (PBA_HASNEXT(s2))
                    s1 = stack[PBA_IDX(tx, y2, tz, xy_size)];
            }
            PBA_DECODE(p, nx, ny, nz);
            s1 = s2;
            s2 = PBA_ENCODE(nx, lasty, nz, 0, flag);
            y2 = lasty;
            lasty = ty;
            stack[id] = s2;
            flag = 1;
        }
    }
    if (PBA_NOTSITE(p))
        stack[PBA_IDX(tx, xy_size - 1, tz, xy_size)] = PBA_ENCODE(0, lasty, 0, 1, flag);
}

// Phase 3: Color along Y axis (transposes X↔Y in output)
__global__ void pba_kernelColorAxis(int *input, int *output, int xy_size, int z_size)
{
    __shared__ int block[PBA_BLOCKSIZE][PBA_BLOCKSIZE + 1];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int tz = blockIdx.y;
    if (tx >= xy_size || tz >= z_size) return;

    int x1, y1, z1, x2, y2, z2;
    int last1 = PBA_ENCODE(0,0,0,1,0), last2 = PBA_ENCODE(0,0,0,1,0), lasty;
    long long dx, dy, dz, best, dist;

    lasty = xy_size - 1;

    last2 = input[PBA_IDX(tx, lasty, tz, xy_size)];
    PBA_DECODE(last2, x2, y2, z2);

    if (PBA_NOTSITE(last2)) {
        lasty = y2;
        if(PBA_HASNEXT(last2)) {
            last2 = input[PBA_IDX(tx, lasty, tz, xy_size)];
            PBA_DECODE(last2, x2, y2, z2);
        }
    }

    if (PBA_HASNEXT(last2)) {
        last1 = input[PBA_IDX(tx, y2, tz, xy_size)];
        PBA_DECODE(last1, x1, y1, z1);
    }

    int y_start, y_end, n_step = xy_size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = xy_size - step * blockDim.x - 1;
        y_end = xy_size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = x2 - tx; dy = lasty - ty; dz = z2 - tz;
            best = dx * dx + dy * dy + dz * dz;

            while (PBA_HASNEXT(last2)) {
                dx = x1 - tx; dy = y2 - ty; dz = z1 - tz;
                dist = dx * dx + dy * dy + dz * dz;
                if(dist > best) break;
                best = dist; lasty = y2; last2 = last1;
                PBA_DECODE(last2, x2, y2, z2);
                if (PBA_HASNEXT(last2)) {
                    last1 = input[PBA_IDX(tx, y2, tz, xy_size)];
                    PBA_DECODE(last1, x1, y1, z1);
                }
            }
            block[threadIdx.x][ty - y_end] = PBA_ENCODE(lasty, x2, z2, PBA_NOTSITE(last2), 0);
        }

        __syncthreads();

        if(!threadIdx.y) {
            // Transposed write: X↔Y
            int id = PBA_IDX(y_end + threadIdx.x, blockIdx.x * blockDim.x, tz, xy_size);
            for(int i = 0; i < blockDim.x; i++, id += xy_size) {
                output[id] = block[i][threadIdx.x];
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Init: boundary map → PBA encoding, with axis remapping
// z_axis: which original axis (0=width,1=height,2=depth) maps to PBA_z
// ============================================================================
__global__ void pba_init_from_boundary(
    const char* d_boundary, int* d_voronoi,
    uint orig_w, uint orig_h, uint orig_d,
    int xy_size, int z_size, int z_axis)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z * blockDim.z + threadIdx.z;
    if (px >= xy_size || py >= xy_size || pz >= z_size) return;

    int pid = PBA_IDX(px, py, pz, xy_size);
    // Map PBA coords → original coords
    int ox, oy, oz;
    if      (z_axis == 0) { ox = pz; oy = px; oz = py; }
    else if (z_axis == 1) { ox = px; oy = pz; oz = py; }
    else                  { ox = px; oy = py; oz = pz; }

    if (ox >= 0 && ox < (int)orig_w && oy >= 0 && oy < (int)orig_h && oz >= 0 && oz < (int)orig_d) {
        size_t oidx = (size_t)ox + (size_t)oy * orig_w + (size_t)oz * orig_w * orig_h;
        d_voronoi[pid] = (d_boundary[oidx] == (char)1)
            ? PBA_ENCODE(px, py, pz, 0, 0)
            : PBA_MARKER;
    } else {
        d_voronoi[pid] = PBA_MARKER;
    }
}

// ============================================================================
// Extract: PBA Voronoi → packed uint32 index + float distance
// Decodes PBA coords back to original coords via axis remapping.
// Output packed format: (orig_z<<20)|(orig_y<<10)|orig_x (JFA-compatible)
// ============================================================================
__global__ void pba_extract_result(
    const int* d_voronoi, unsigned int* d_packed_index, float* d_distance,
    uint orig_w, uint orig_h, uint orig_d,
    int xy_size, int z_size, int z_axis)
{
    uint ox = blockIdx.x * blockDim.x + threadIdx.x;
    uint oy = blockIdx.y * blockDim.y + threadIdx.y;
    uint oz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ox >= orig_w || oy >= orig_h || oz >= orig_d) return;

    size_t oidx = (size_t)ox + (size_t)oy * orig_w + (size_t)oz * orig_w * orig_h;

    // Map original coords → PBA coords
    int px, py, pz;
    if      (z_axis == 0) { px = (int)oy; py = (int)oz; pz = (int)ox; }
    else if (z_axis == 1) { px = (int)ox; py = (int)oz; pz = (int)oy; }
    else                  { px = (int)ox; py = (int)oy; pz = (int)oz; }

    int pid = PBA_IDX(px, py, pz, xy_size);
    int voronoi = d_voronoi[pid];

    if (PBA_NOTSITE(voronoi) || voronoi == PBA_MARKER) {
        d_packed_index[oidx] = PBA_JFA_INVALID;
        if (d_distance) d_distance[oidx] = 0.0f;
        return;
    }

    int spx, spy, spz;
    PBA_DECODE(voronoi, spx, spy, spz);

    // Map PBA nearest-site coords → original coords
    int sox, soy, soz;
    if      (z_axis == 0) { sox = spz; soy = spx; soz = spy; }
    else if (z_axis == 1) { sox = spx; soy = spz; soz = spy; }
    else                  { sox = spx; soy = spy; soz = spz; }

    // JFA-compatible packed: (z<<20)|(y<<10)|x
    d_packed_index[oidx] = ((unsigned int)soz << 20) | ((unsigned int)soy << 10) | (unsigned int)sox;

    if (d_distance) {
        float dx = (float)(sox - (int)ox);
        float dy = (float)(soy - (int)oy);
        float dz = (float)(soz - (int)oz);
        d_distance[oidx] = sqrtf(dx*dx + dy*dy + dz*dz);
    }
}

// ============================================================================
// Fused kernel: boundary detection + sign map + PBA init in one pass.
// Replaces get_boundary_and_sign_map + pba_init_from_boundary.
// Writes d_boundary, d_sign_map (original coords), and d_voronoi (PBA coords).
// d_voronoi must be pre-filled with a NOTSITE value for padding regions.
// ============================================================================
__global__ void pba_boundary_sign_init(
    const int* d_quant_inds,
    char* d_boundary, char* d_sign_map,
    int* d_voronoi,
    uint width, uint height, uint depth,
    int xy_size, int z_axis)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= (int)width || y >= (int)height || z >= (int)depth) return;

    size_t idx = (size_t)x + (size_t)y * width + (size_t)z * width * height;
    char is_boundary = 0;

    if (x == 0 || x == (int)width-1 || y == 0 || y == (int)height-1 ||
        z == 0 || z == (int)depth-1) {
        d_boundary[idx] = 0;
    } else {
        int cur   = d_quant_inds[idx];
        int left  = d_quant_inds[(x-1) + y*width + z*width*height];
        int right = d_quant_inds[(x+1) + y*width + z*width*height];
        int up    = d_quant_inds[x + (y-1)*width + z*width*height];
        int down  = d_quant_inds[x + (y+1)*width + z*width*height];
        int front = d_quant_inds[x + y*width + (z-1)*width*height];
        int back  = d_quant_inds[x + y*width + (z+1)*width*height];

        if (cur != left || cur != right || cur != up ||
            cur != down || cur != front || cur != back) {
            is_boundary = 1;
            d_boundary[idx] = 1;

            int signs[6] = {up-cur, down-cur, left-cur, right-cur, front-cur, back-cur};
            double grad_x = abs(right - left) * 0.5;
            double grad_y = abs(up - down) * 0.5;
            double grad_z = abs(front - back) * 0.5;
            double max_grad = grad_x;
            if (grad_y > max_grad) max_grad = grad_y;
            if (grad_z > max_grad) max_grad = grad_z;

            if (max_grad >= 1.0) {
                d_sign_map[idx] = 0;
            } else {
                for (int i = 0; i < 6; i++) {
                    if (signs[i] != 0) {
                        d_sign_map[idx] = (signs[i] > 0) ? 1 : -1;
                        break;
                    }
                }
            }
        } else {
            d_boundary[idx] = 0;
        }
    }

    // PBA init: map original coords to PBA coords and write voronoi
    int px, py, pz;
    if      (z_axis == 0) { px = y;  py = z;  pz = x; }
    else if (z_axis == 1) { px = x;  py = z;  pz = y; }
    else                  { px = x;  py = y;  pz = z; }

    int pid = PBA_IDX(px, py, pz, xy_size);
    d_voronoi[pid] = is_boundary ? PBA_ENCODE(px, py, pz, 0, 0) : PBA_MARKER;
}

// ============================================================================
// Host helpers
// ============================================================================
inline int pba_next_mult(int v, int m) {
    return ((v + m - 1) / m) * m;
}

inline int pba_choose_axis(uint w, uint h, uint d, int &xy_size, int &z_size)
{
    int dims[3] = {(int)w, (int)h, (int)d};
    long long best_vol = LLONG_MAX;
    int best_z = 2;

    for (int za = 0; za < 3; za++) {
        int d1 = dims[(za + 1) % 3];
        int d2 = dims[(za + 2) % 3];
        int xy = pba_next_mult(d1 > d2 ? d1 : d2, 32);
        int z  = pba_next_mult(dims[za], 4);
        if (xy < 32) xy = 32;
        if (z  < 4)  z  = 4;
        long long vol = (long long)xy * xy * z;
        if (vol < best_vol) {
            best_vol = vol;
            best_z = za;
            xy_size = xy;
            z_size = z;
        }
    }
    return best_z;
}

// Returns the required buffer size in BYTES for one PBA ping-pong buffer.
// Caller should allocate two buffers of this size.
inline size_t pba_buffer_size(uint w, uint h, uint d)
{
    int xy_size, z_size;
    pba_choose_axis(w, h, d, xy_size, z_size);
    return (size_t)xy_size * xy_size * z_size * sizeof(int);
}

// ============================================================================
// Main entry point — with pre-allocated buffers (no cudaMalloc/Free overhead)
// d_buf0, d_buf1: two device buffers, each at least pba_buffer_size() bytes.
// Pass nullptr for both to fall back to internal allocation.
// ============================================================================
void edt_3d_pba(
    char* d_boundary, int* index, float* distance,
    uint width, uint height, uint depth,
    int* d_buf0, int* d_buf1)
{
    int xy_size, z_size;
    int z_axis = pba_choose_axis(width, height, depth, xy_size, z_size);

    if (xy_size > 1024) {
        printf("[PBA+] ERROR: xy_size=%d exceeds 1024 (10-bit coordinate limit)\n", xy_size);
        return;
    }

    long long padded_n = (long long)xy_size * xy_size * z_size;
    size_t padded_bytes = (size_t)padded_n * sizeof(int);

    // Use caller buffers or allocate internally
    bool own_bufs = (d_buf0 == nullptr || d_buf1 == nullptr);
    int* d_buf[2] = {d_buf0, d_buf1};
    if (own_bufs) {
        cudaError_t e1 = cudaMalloc(&d_buf[0], padded_bytes);
        cudaError_t e2 = cudaMalloc(&d_buf[1], padded_bytes);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            printf("[PBA+] ERROR: alloc failed (%zu MB each): %s\n",
                   padded_bytes >> 20, cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
            cudaFree(d_buf[0]); cudaFree(d_buf[1]);
            return;
        }
    }

    // --- Init ---
    {
        dim3 block(8, 8, 8);
        dim3 grid((xy_size + 7) / 8, (xy_size + 7) / 8, (z_size + 7) / 8);
        pba_init_from_boundary<<<grid, block>>>(d_boundary, d_buf[0],
            width, height, depth, xy_size, z_size, z_axis);
    }

    int cur = 0;

    // --- Phase 1: Flood Z ---
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (xy_size + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelFloodZ<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
        cur = 1 - cur;
    }

    // --- Phase 2a: Maurer Y ---
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (z_size  + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelMaurerAxis<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
    }

    // --- Phase 3a: Color Y (transposes X\xe2\x86\x94Y) ---
    {
        dim3 block(PBA_BLOCKSIZE, 2);
        dim3 grid(xy_size / PBA_BLOCKSIZE, z_size);
        pba_kernelColorAxis<<<grid, block>>>(d_buf[1 - cur], d_buf[cur], xy_size, z_size);
    }

    // --- Phase 2b: Maurer Y (on transposed data) ---
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (z_size  + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelMaurerAxis<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
    }

    // --- Phase 3b: Color Y (transposes back) ---
    {
        dim3 block(PBA_BLOCKSIZE, 2);
        dim3 grid(xy_size / PBA_BLOCKSIZE, z_size);
        pba_kernelColorAxis<<<grid, block>>>(d_buf[1 - cur], d_buf[cur], xy_size, z_size);
    }

    // --- Extract results (no sync needed — same stream guarantees ordering) ---
    {
        dim3 block(64, 4, 2);
        dim3 grid((width  + 63) / 64, (height + 3) / 4, (depth + 1) / 2);
        pba_extract_result<<<grid, block>>>(d_buf[cur], (unsigned int*)index, distance,
            width, height, depth, xy_size, z_size, z_axis);
    }

    if (own_bufs) {
        cudaFree(d_buf[0]);
        cudaFree(d_buf[1]);
    }
}

// Convenience overload — allocates/frees buffers internally
void edt_3d_pba(
    char* d_boundary, int* index, float* distance,
    uint width, uint height, uint depth)
{
    edt_3d_pba(d_boundary, index, distance, width, height, depth, nullptr, nullptr);
}

// ============================================================================
// Fused boundary+EDT: combines boundary detection + PBA compute in one call.
// Replaces separate get_boundary_and_sign_map + edt_3d_pba for EDT round 1.
// Writes d_boundary and d_sign_map as side effects.
// ============================================================================
void edt_3d_pba_fused_boundary(
    const int* d_quant_inds,
    char* d_boundary, char* d_sign_map,
    int* index, float* distance,
    uint width, uint height, uint depth,
    int* d_buf0, int* d_buf1)
{
    int xy_size, z_size;
    int z_axis = pba_choose_axis(width, height, depth, xy_size, z_size);

    if (xy_size > 1024) {
        printf("[PBA+] ERROR: xy_size=%d exceeds 1024\n", xy_size);
        return;
    }

    long long padded_n = (long long)xy_size * xy_size * z_size;
    size_t padded_bytes = (size_t)padded_n * sizeof(int);

    bool own_bufs = (d_buf0 == nullptr || d_buf1 == nullptr);
    int* d_buf[2] = {d_buf0, d_buf1};
    if (own_bufs) {
        cudaError_t e1 = cudaMalloc(&d_buf[0], padded_bytes);
        cudaError_t e2 = cudaMalloc(&d_buf[1], padded_bytes);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            cudaFree(d_buf[0]); cudaFree(d_buf[1]);
            return;
        }
    }

    // Fill PBA buffer with NOTSITE for padding (byte 0x80 → bit31=1, NOTSITE=1)
    cudaMemset(d_buf[0], 0x80, padded_bytes);

    // Fused boundary detection + sign map + PBA init (one pass over original volume)
    {
        dim3 block(8, 8, 8);
        dim3 grid((width + 7) / 8, (height + 7) / 8, (depth + 7) / 8);
        pba_boundary_sign_init<<<grid, block>>>(d_quant_inds,
            d_boundary, d_sign_map, d_buf[0],
            width, height, depth, xy_size, z_axis);
    }

    int cur = 0;

    // PBA phases (same as edt_3d_pba)
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (xy_size + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelFloodZ<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
        cur = 1 - cur;
    }
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (z_size  + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelMaurerAxis<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
    }
    {
        dim3 block(PBA_BLOCKSIZE, 2);
        dim3 grid(xy_size / PBA_BLOCKSIZE, z_size);
        pba_kernelColorAxis<<<grid, block>>>(d_buf[1 - cur], d_buf[cur], xy_size, z_size);
    }
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (z_size  + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelMaurerAxis<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
    }
    {
        dim3 block(PBA_BLOCKSIZE, 2);
        dim3 grid(xy_size / PBA_BLOCKSIZE, z_size);
        pba_kernelColorAxis<<<grid, block>>>(d_buf[1 - cur], d_buf[cur], xy_size, z_size);
    }

    // Extract
    {
        dim3 block(64, 4, 2);
        dim3 grid((width + 63) / 64, (height + 3) / 4, (depth + 1) / 2);
        pba_extract_result<<<grid, block>>>(d_buf[cur], (unsigned int*)index, distance,
            width, height, depth, xy_size, z_size, z_axis);
    }

    if (own_bufs) {
        cudaFree(d_buf[0]);
        cudaFree(d_buf[1]);
    }
}

// ============================================================================
// Downsampled EDT Round 2: 2x2x2 downsample → PBA → float distance output
// ============================================================================

// Downsample boundary by 2x2x2 using logical OR.
// Input: d_boundary (full-res width×height×depth, char with 0/1)
// Output: d_ds_boundary (ds_w×ds_h×ds_d, char with 0/1)
__global__ void pba_downsample_boundary_2x(
    const char* d_boundary, char* d_ds_boundary,
    int width, int height, int depth,
    int ds_w, int ds_h, int ds_d)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int dz = blockIdx.z * blockDim.z + threadIdx.z;
    if (dx >= ds_w || dy >= ds_h || dz >= ds_d) return;

    int fx = dx * 2, fy = dy * 2, fz = dz * 2;
    char val = 0;
    for (int zz = 0; zz < 2 && fz + zz < depth; zz++)
        for (int yy = 0; yy < 2 && fy + yy < height; yy++)
            for (int xx = 0; xx < 2 && fx + xx < width; xx++) {
                size_t fidx = (size_t)(fx+xx) + (size_t)(fy+yy)*width + (size_t)(fz+zz)*width*height;
                if (d_boundary[fidx] != 0) { val = 1; }
            }

    size_t didx = (size_t)dx + (size_t)dy * ds_w + (size_t)dz * ds_w * ds_h;
    d_ds_boundary[didx] = val;
}

// Downsampled PBA EDT: runs PBA on a 2x2x2-downsampled boundary volume.
// Outputs float distance in downsampled coordinates.
// Caller provides pre-allocated PBA buffers (must be sized for the downsampled volume).
// d_ds_packed_out: if non-null, receives coarse packed nearest-neighbor index (caller owns memory).
void edt_3d_pba_downsampled(
    char* d_boundary_full, float* d_ds_distance,
    uint width, uint height, uint depth,
    int* d_buf0, int* d_buf1,
    unsigned int* d_ds_packed_out = nullptr)
{
    int ds_w = ((int)width  + 1) / 2;
    int ds_h = ((int)height + 1) / 2;
    int ds_d = ((int)depth  + 1) / 2;

    // Allocate downsampled boundary
    char* d_ds_boundary = nullptr;
    size_t ds_size = (size_t)ds_w * ds_h * ds_d;
    cudaMalloc(&d_ds_boundary, ds_size * sizeof(char));

    // Downsample
    {
        dim3 block(8, 8, 8);
        dim3 grid((ds_w + 7) / 8, (ds_h + 7) / 8, (ds_d + 7) / 8);
        pba_downsample_boundary_2x<<<grid, block>>>(
            d_boundary_full, d_ds_boundary,
            (int)width, (int)height, (int)depth,
            ds_w, ds_h, ds_d);
    }

    // PBA on downsampled volume — we only need float distance output, not packed index
    // Reuse the index buffer temporarily for PBA internal use
    int xy_size, z_size;
    int z_axis = pba_choose_axis((uint)ds_w, (uint)ds_h, (uint)ds_d, xy_size, z_size);

    if (xy_size > 1024) {
        printf("[PBA+ ds] ERROR: xy_size=%d exceeds 1024\n", xy_size);
        cudaFree(d_ds_boundary);
        return;
    }

    long long padded_n = (long long)xy_size * xy_size * z_size;
    size_t padded_bytes = (size_t)padded_n * sizeof(int);

    // Check if caller buffers are large enough; if not, allocate
    size_t needed = padded_bytes;
    size_t caller_size = pba_buffer_size(width, height, depth);
    bool own_bufs = (d_buf0 == nullptr || d_buf1 == nullptr || needed > caller_size);
    int* d_buf[2] = {d_buf0, d_buf1};
    if (own_bufs) {
        cudaMalloc(&d_buf[0], padded_bytes);
        cudaMalloc(&d_buf[1], padded_bytes);
    }

    // Init
    {
        dim3 block(8, 8, 8);
        dim3 grid((xy_size + 7) / 8, (xy_size + 7) / 8, (z_size + 7) / 8);
        pba_init_from_boundary<<<grid, block>>>(d_ds_boundary, d_buf[0],
            (uint)ds_w, (uint)ds_h, (uint)ds_d, xy_size, z_size, z_axis);
    }

    int cur = 0;

    // PBA phases
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (xy_size + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelFloodZ<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
        cur = 1 - cur;
    }
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (z_size  + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelMaurerAxis<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
    }
    {
        dim3 block(PBA_BLOCKSIZE, 2);
        dim3 grid(xy_size / PBA_BLOCKSIZE, z_size);
        pba_kernelColorAxis<<<grid, block>>>(d_buf[1 - cur], d_buf[cur], xy_size, z_size);
    }
    {
        dim3 block(PBA_BLOCKX, PBA_BLOCKY);
        dim3 grid((xy_size + PBA_BLOCKX - 1) / PBA_BLOCKX,
                  (z_size  + PBA_BLOCKY - 1) / PBA_BLOCKY);
        pba_kernelMaurerAxis<<<grid, block>>>(d_buf[cur], d_buf[1 - cur], xy_size, z_size);
    }
    {
        dim3 block(PBA_BLOCKSIZE, 2);
        dim3 grid(xy_size / PBA_BLOCKSIZE, z_size);
        pba_kernelColorAxis<<<grid, block>>>(d_buf[1 - cur], d_buf[cur], xy_size, z_size);
    }

    // Extract distance (and optionally the packed nearest-neighbor index).
    unsigned int* d_ds_packed = d_ds_packed_out;
    bool own_packed = (d_ds_packed_out == nullptr);
    if (own_packed) cudaMalloc(&d_ds_packed, ds_size * sizeof(unsigned int));
    {
        dim3 block(64, 4, 2);
        dim3 grid((ds_w + 63) / 64, (ds_h + 3) / 4, (ds_d + 1) / 2);
        pba_extract_result<<<grid, block>>>(d_buf[cur], d_ds_packed, d_ds_distance,
            (uint)ds_w, (uint)ds_h, (uint)ds_d, xy_size, z_size, z_axis);
    }
    if (own_packed) cudaFree(d_ds_packed);
    cudaFree(d_ds_boundary);
    if (own_bufs) {
        cudaFree(d_buf[0]);
        cudaFree(d_buf[1]);
    }
}

// Returns the downsampled dimensions for a given full-res volume.
inline void pba_downsample_dims(uint w, uint h, uint d, int &ds_w, int &ds_h, int &ds_d) {
    ds_w = ((int)w + 1) / 2;
    ds_h = ((int)h + 1) / 2;
    ds_d = ((int)d + 1) / 2;
}

#endif // EDT_PBA_HPP
