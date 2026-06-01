# GPU 2D Euclidean Distance Transform — implementations head to head

A small, self-contained benchmark comparing GPU implementations of the **exact 2D Euclidean
Distance Transform / Voronoi diagram** on binary images — the vendor library, the academic
reference, our pipeline's interface, and the "portable" alternative — on the same GPU, same
input, compute-only timing, all cross-verified to produce the same field.

| # | implementation | kind | source | license |
|---|---|---|---|---|
| 1 | **NVIDIA NPP** `nppiDistanceTransformPBA` | 2D-native vendor library (internally = PBA) | ships with the CUDA Toolkit | NVIDIA EULA (linked, not redistributed) |
| 2 | **NUS PBA+** `pba2D` | 2D-native academic reference (Parallel Banding Algorithm) | [orzzzjq/Parallel-Banding-Algorithm-plus](https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus) | MIT (`third_party/nus/LICENSE`) |
| 3 | **ours-2D** `edt_2d_pba` | **native 2D** EDT with our 3D-style device interface (over the NUS 2D kernels) | this project | MIT |
| 4 | **ours-3Don2D** `edt_3d_pba` | our **3D** PBA run as `W×H×1` (shortcut, for contrast; ≤1024) | this project | MIT |
| 5 | **FH** `edt_fh` | **Felzenszwalb–Huttenlocher** separable EDT — the *portable* SOTA alternative | this project (`baselines/`) | MIT |

NUS is run at two band settings — **`m3=16`** (occupancy-tuned) and **`m3=2`** (the common
default) — to expose the band-tuning lever directly.

## Is PBA still state of the art? (short answer: yes, on NVIDIA, for exact EDT)

A literature check (2024–2025) found **nothing that beats PBA on GPU for the exact transform**:

- Recent papers report speedups against a **sequential CPU** baseline — typically SciPy's
  `distance_transform_edt`, i.e. **Maurer's algorithm** (Maurer et al. 2003) — not against PBA
  (e.g. *GPU-Based Parallel EDT*, MDPI Mathematics 2025: 52× vs CPU; *Accelerating EDT*, IEEE
  Access 2025: 250×/400× vs CPU). That choice inflates the numbers and quietly avoids the GPU
  state of the art. The irony: **PBA's phase 2 is the *parallelized* Maurer scan** (our
  `pba_kernelMaurerAxis`), so a "GPU-method vs CPU-Maurer" comparison is really a comparison
  against the sequential form of a sub-step of the algorithm it's dodging. A fair claim has to put
  the method next to PBA *on the same GPU* — which is what this benchmark does.
- The one genuinely modern competitor, **[DistanceTransforms.jl](https://github.com/MolloiLab/DistanceTransforms.jl)**
  (IEEE Access 2025), uses the **Felzenszwalb–Huttenlocher** separable algorithm and wins on
  **portability** (CUDA + ROCm + Metal + oneAPI, Julia/Python, DL-loss integration), *not* raw
  NVIDIA speed. It reports no PBA comparison.
- The NUS authors note PBA is *mathematically equivalent* to Felzenszwalb (both O(N)); the
  difference is the GPU mapping. NVIDIA's own [Xavier EDT talk (GTC 2019)](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9165-euclidean-distance-transform-on-xavier.pdf)
  recommends PBA for high resolution and Felzenszwalb only for small images.

This benchmark includes **FH as baseline #5** to make that comparison concrete: on NVIDIA, the
PBA family is **~10× faster than FH** at 4096² (see below), so PBA is the right primitive — and
NPP, being PBA internally, is the same algorithm as the academic code.

## Results — RTX 5090, 1% random sites (compute-only, best of 11)

`Gpix/s` = pixels/10⁹/s (base-1000 count rate). `GiB/s` = 4·pixels/1024³/s (the float distance
field written out, base-1024). Full multi-density log: [`results/rtx5090_thorough.txt`](results/rtx5090_thorough.txt).

```
size   impl              best_ms    Gpix/s     GiB/s   max_err
256    ours-2D            0.0820     0.800     2.979      0.00
256    NUS-PBA+(m3=16)    0.0793     0.826     3.078      0.00
256    NUS-PBA+(m3=2)     0.0976     0.671     2.501      0.00
256    NPP                0.0808     0.812     3.023      0.97
256    FH                 0.3959     0.166     0.617      0.00
512    ours-2D            0.1069     2.452     9.133      0.00
512    NUS-PBA+(m3=16)    0.1042     2.515     9.369      0.00
512    NPP                0.1070     2.450     9.125      0.97
512    FH                 0.8137     0.322     1.200      0.00
1024   ours-2D            0.1575     6.658    24.801      0.00
1024   NUS-PBA+(m3=16)    0.1517     6.912    25.748      0.00
1024   NPP                0.1607     6.525    24.309      0.97
1024   FH                 1.6215     0.647     2.409      0.00
2048   ours-2D            0.2909    14.419    53.716      0.00
2048   NUS-PBA+(m3=16)    0.2713    15.459    57.589      0.00
2048   NUS-PBA+(m3=2)     0.4169    10.060    37.478      0.00
2048   NPP                0.3037    13.811    51.450      0.97
2048   FH                 3.2994     1.271     4.736      0.00
4096   ours-2D            0.8768    19.135    71.285      0.00
4096   NUS-PBA+(m3=16)    0.7304    22.971    85.572      0.00
4096   NUS-PBA+(m3=2)     1.0956    15.313    57.044      0.00
4096   NPP                0.9815    17.094    63.681      0.97
4096   FH                 8.5139     1.971     7.341      0.00
```

**What the numbers say (4096²):**
- **PBA ≫ FH on GPU**: NUS-tuned 23.0, ours-2D 19.1, NPP 17.1 vs **FH 1.97 Gpix/s** — the portable
  separable algorithm is ~10× slower here. PBA's banding wins on GPU, exactly as the literature predicts.
- **Band tuning is a real ~1.5×**: NUS `m3=16` (23.0) vs `m3=2` (15.3). The dominant `kernelColor`
  is latency-bound when under-occupied; the max valid band `m3=16` (1024-thread blocks) fixes it.
- **NPP runs the same PBA** and **adapts** its bands (block dims read from the binary via ncu:
  `m3=16` @1024², `m3=8` @4096²) — so it ties the tuned PBA at small sizes and is overtaken at 4096²
  only because `m3=16` beats its `m3=8` on Blackwell. See [`results/npp_profile_5090.md`](results/npp_profile_5090.md).
- **Everything agrees**: `max_err 0.00` for ours/NUS/**FH** (FH is bit-exact to display precision,
  even at 4096² where squared distances exceed 2²⁴). NPP's `0.97` is its 16-bit *truncated* integer
  output (`<1` off everywhere), i.e. also correct.

The benchmark also sweeps **site density {1, 10, 50}%** and a **real NYX edge map** (35% sites) —
see [`results/rtx5090_thorough.txt`](results/rtx5090_thorough.txt) and
[`results/rtx5090_nyx.txt`](results/rtx5090_nyx.txt). Ranking is stable across all of them.

> **nsys/ncu profiling of NPP** ([`results/npp_profile_5090.md`](results/npp_profile_5090.md))
> shows NPP's `nppiDistanceTransformPBA` runs the **NUS PBA+ kernels** (`kernelFloodDown`,
> `kernelProximatePoints`, `kernelColor`, …) plus its own format-conversion kernels — the "PBA" in
> the name *is* the Parallel Banding Algorithm. The dominant `kernelColor` is latency-bound; no
> kernel is compute-bound.

## The implementations in detail

- **ours-2D** ([`ours/edt_2d.hpp`](ours/edt_2d.hpp)) — a **native 2D EDT** exposing the same device
  interface as this project's 3D EDT (1-byte boundary in; packed nearest-site `index` + `float
  distance` out, all device-resident), so it drops into the 2D version of our pipeline. Its core is
  the NUS PBA+ 2D kernels through a thin device bridge. ~as fast as NUS; the small gap is our
  `boundary → index/distance` conversion kernels (which NUS doesn't run). 16-bit coords ⇒ no 1024 cap.

- **ours-3Don2D** ([`ours/edt_pba.hpp`](ours/edt_pba.hpp)) — our **3D** PBA on a `W×H×1` volume.
  ~3–4× slower and capped at 1024, not from depth padding (empty z-planes are nearly free) but
  **structurally**: the 3D pipeline spends its cheap flood pass on the trivial Z axis, so both real
  axes are resolved by two *expensive* Maurer/Color passes; a native 2D PBA floods a real axis and
  needs only one. Kept as a bit-exact cross-check and to quantify that shortcut.

- **FH** ([`baselines/edt_fh.hpp`](baselines/edt_fh.hpp)) — the **Felzenszwalb–Huttenlocher**
  separable lower-envelope-of-parabolas algorithm, O(N), exact, the algorithm behind the portable
  multi-vendor libraries. Our implementation is *competently coalesced* (transposes between the two
  1D-DT passes) so it isn't a strawman — yet it's still ~10× behind PBA on NVIDIA, because the
  per-line envelope walk has far less parallelism and worse memory locality than PBA's banding.

## Build & run

Requires the **CUDA Toolkit** (with NPP — included by default). Tested with CUDA 12/13 on
`sm_120` (RTX 5090) and `sm_89` (RTX 4070).

```bash
make                 # builds ./bench  (nvcc -arch=native)
./bench              # default: sizes 256..4096 × densities {1,10,50}% (random sites)
./bench 512 4096     # custom sizes (still sweeps the three densities)
./bench nyx field.f32 512   # real data: binary image = quantization edges of a
                            # 512×512 slice of a NYX field (rel_eb=1e-2)
```

CMake is also provided (cross-platform, modern target-based):
```bash
cmake -B build && cmake --build build -j   # builds bench, npp_prof, tune
```

If `nvcc`/NPP are not on your default path:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Methodology

- Input: a binary image — synthetic random sites at fixed seed (1/10/50%) or a real NYX edge map.
- Each implementation gets the **same** sites in its own native input format
  (NPP: 8-bit, sites=0; NUS: `short2` coords / `MARKER`; ours: 1-byte boundary mask; FH: 1-byte mask).
- **Compute-only timing**: inputs uploaded once; only on-GPU EDT kernels are timed (warm-up, then
  **best + median of 11**). Host↔device transfers are excluded — including NUS's per-iteration input
  restore (`pba2DInitializeInput` is a destructive-buffer H2D refill, run *before* the timer each iteration).
- **Verification**: every implementation's distance field is compared element-wise against the
  reference (ours for ≤1024, NUS above); `max_err` is reported.

## Layout

```
bench.cu                 unified harness (generate, run all impls, time, verify)
Makefile / CMakeLists.txt
ours/edt_pba.hpp         our 3D PBA (MIT)
ours/edt_2d.hpp          native-2D driver over the same kernels, MIT
baselines/edt_fh.hpp     Felzenszwalb–Huttenlocher separable EDT (MIT)
third_party/nus/         NUS PBA+ 2D, ported to CUDA 12+ (no <device_functions.h>); MIT
prof/                    npp_prof.cu, tune.cu — NPP profiling + band sweep
results/                 benchmark logs + npp_profile_5090.md
```

## Credits & licenses

- **NUS PBA+** — Parallel Banding Algorithm, *Cao, Tang, Mohamed, Tan (ACM I3D 2010)*; code
  © 2019 School of Computing, NUS, MIT (`third_party/nus/LICENSE`). Only `<device_functions.h>`
  was removed for CUDA 12+; the algorithm/kernels are unchanged.
- **Felzenszwalb–Huttenlocher** — *Distance Transforms of Sampled Functions*, Theory of Computing 2012.
- **NVIDIA NPP** — `nppiDistanceTransformPBA`, part of the CUDA Toolkit; linked, not redistributed.
- **ours**, **FH**, and this harness — MIT (`LICENSE`).
