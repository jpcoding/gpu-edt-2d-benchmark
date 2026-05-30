# Profiling NVIDIA NPP `nppiDistanceTransformPBA` (RTX 5090, CUDA 13.2)

Driver: [`prof/npp_prof.cu`](../prof/npp_prof.cu) — device-resident input, NPP kernels only.

## Headline finding: NPP's distance transform **is** the NUS Parallel Banding Algorithm

nsys shows the internal kernels are `kernelFloodDown/Up`, `kernelPropagateInterband`,
`kernelUpdateVertical`, `kernelProximatePoints`, `kernelCreateForwardPointers`,
`kernelMergeBands`, `kernelDoubleToSingleList`, `kernelColor` — i.e. **the NUS PBA+ 2D kernels**
(the "PBA" in the function name *is* the Parallel Banding Algorithm). NPP adds only two of its
own kernels for format conversion: `initInputVoronoi` (8-bit → short2) and
`generateAdditionalRelativeOutput_16u` (→ 16-bit distance).

So all three implementations in this repo run the **same core algorithm**: NPP and `ours-2D`
are both wrappers around the NUS PBA kernels (NPP's are NVIDIA-tuned), and NUS-PBA+ is the
original. Performance differences are wrapper/tuning, not algorithm.

## nsys — per-kernel GPU time (4096², 20 iters), sorted

| % | kernel | avg (µs) | note |
|---|---|---|---|
| 31.8 | kernelColor | 318.0 | final coloring/scatter (dominant) |
| 13.5 | initInputVoronoi | 135.6 | NPP input conversion |
| 12.1 | kernelProximatePoints | 121.7 | PBA phase 2 |
| 11.9 | kernelDoubleToSingleList | 119.8 | PBA phase 2 |
| 7.6 | generateAdditionalRelativeOutput_16u | 76.1 | NPP output conversion |
| 7.2 | kernelFloodDown | 72.4 | PBA phase 1 |
| 6.6 | kernelUpdateVertical | 66.3 | PBA phase 1 |
| 3.6 | kernelMergeBands (×5) | 7.2 ea | PBA phase 2 band merge |
| 3.0 | kernelFloodUp | 30.6 | PBA phase 1 |
| 1.9 | kernelCreateForwardPointers | 19.3 | PBA phase 2 |
| 0.6 | kernelPropagateInterband | 6.2 | PBA phase 1 |

Σ ≈ **1.0 ms / transform** at 4096² — matches the cudaEvent number in the main benchmark
(0.98 ms), i.e. the harness timing was already the true on-GPU time.

## ncu — per-kernel Speed-of-Light (2048²)

| kernel | µs | SM % | DRAM % | bound by |
|---|---|---|---|---|
| kernelColor | 179.7 | 6.7 | 1.0 | **latency** (scatter via linked list) |
| kernelProximatePoints | 48.2 | 32.4 | 19.7 | mixed |
| kernelDoubleToSingleList | 35.8 | 14.4 | 53.2 | DRAM |
| kernelFloodUp | 26.8 | 7.3 | 35.5 | DRAM/latency |
| kernelFloodDown | 26.3 | 7.4 | 36.2 | DRAM/latency |
| kernelUpdateVertical | 24.3 | 10.7 | 40.4 | DRAM |
| initInputVoronoi | 21.4 | 24.5 | 42.8 | DRAM |
| kernelCreateForwardPointers | 16.0 | 8.6 | 14.0 | latency |
| kernelMergeBands | 8.9 | 0.6 | 1.3 | latency |
| kernelPropagateInterband | 7.0 | 4.4 | 4.3 | latency |

**No kernel is compute-bound** (max SM ≈ 32%); the transform is memory/latency-bound, dominated
by the latency-bound `kernelColor` final scatter. Same structural picture as our 3D PBA.

## Reproduce
```bash
make npp_prof
nsys profile --stats=true -o /tmp/npp_nsys ./npp_prof 4096 20    # per-kernel time
ncu --set basic ./npp_prof 2048 1                                # per-kernel detail
```

## Consequence: band tuning beats NPP

`kernelColor` being latency-bound (SM 6.7%) is an **occupancy** problem, not a bandwidth wall:
its block is `(64, m3)`, and the common default `m3=2` gives only 128 threads/block. Raising the
phase-3 band to the max valid `m3=16` (1024-thread blocks) — plus `m2=64` — restores occupancy.
A correctness-verified sweep ([`prof/tune.cu`](../prof/tune.cu)) on the 5090:

| size | default (m3=2) Gpix/s | tuned (m2=64,m3=16) Gpix/s | speedup | NPP Gpix/s |
|---|---|---|---|---|
| 512²  | 1.70  | 2.54  | 1.50× | 2.46 |
| 1024² | 4.39  | 7.00  | 1.59× | 6.53 |
| 2048² | 9.85  | 15.42 | 1.56× | 13.83 |
| 4096² | 15.27 | 22.77 | 1.49× | 17.10 |

So the **tuned PBA overtakes NPP** — NPP runs the same kernels but with fixed bands it can't adapt.
(`m3=32/64` *look* faster but exceed the 1024-thread block limit → the launch fails; the sweep
rejects them by verifying the output, which is why correctness-checking the tuner mattered.)
