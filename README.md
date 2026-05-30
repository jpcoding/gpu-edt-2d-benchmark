# GPU 2D Euclidean Distance Transform — three implementations, head to head

A small, self-contained benchmark comparing three GPU implementations of the **exact 2D
Euclidean Distance Transform / Voronoi diagram** on random binary images:

| # | implementation | kind | source | license |
|---|---|---|---|---|
| 1 | **NVIDIA NPP** `nppiDistanceTransformPBA` | 2D-native, vendor library | ships with the CUDA Toolkit | NVIDIA EULA (linked, not redistributed) |
| 2 | **NUS PBA+** `pba2D` | 2D-native, academic reference | [orzzzjq/Parallel-Banding-Algorithm-plus](https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus) | MIT (`third_party/nus/LICENSE`) |
| 3 | **"ours"** `edt_3d_pba` | **3D**-native PBA, run as `W×H×1` | this project | MIT |

All three are fed the **same** random binary image, timed **compute-only** (data already
device-resident, warm-up + best-of-10), and **cross-verified** to produce the same EDT.

## Results (NVIDIA RTX 5090, ~1% random sites)

```
size   impl            time_ms       Mpix/s       Gpix/s    max_err
256    ours             0.2280        287.4        0.287        ref
256    NUS-PBA+         0.1107        592.1        0.592       0.00
256    NPP              0.0808        811.5        0.812       0.97
512    ours             0.4614        568.2        0.568        ref
512    NUS-PBA+         0.1557       1684.1        1.684       0.00
512    NPP              0.1071       2448.5        2.449       0.97
1024   ours             0.9409       1114.5        1.114        ref
1024   NUS-PBA+         0.2381       4403.4        4.403       0.00
1024   NPP              0.1609       6516.4        6.516       0.97
2048   NUS-PBA+         0.4231       9914.0        9.914       0.00
2048   NPP              0.3034      13822.2       13.822       0.97
4096   NUS-PBA+         1.0909      15378.8       15.379       0.00
4096   NPP              0.9814      17095.5       17.096       0.97
```

(full log: [`results/rtx5090.txt`](results/rtx5090.txt))

**Reading the table**
- `max_err` is the largest distance disagreement vs. the reference field. **NUS = 0.00**
  (bit-exact with ours); **NPP = 0.97** — NPP returns the distance as a *truncated* 16-bit
  integer, so it is off by `< 1` everywhere, i.e. correct. All three agree.
- **NPP** is fastest, **NUS PBA+** close behind; both scale to large images.
- **"ours" trails by ~3–6×, on purpose** — see the honest note below. It is included as a
  *correctness cross-check*, not as a competitive 2D code.

## Why "ours" is slower (and capped at 1024)

`edt_3d_pba` is a **3D** Parallel Banding implementation. To run a 2D image it is treated as
a `W×H×1` volume, which incurs two artifacts a native-2D code never pays:

1. **Depth padding 1 → 4.** The 3D pipeline rounds the depth up to a multiple of 4, so it
   processes ~**4× the cells** of the real 2D image.
2. **In-plane cap of 1024.** Coordinates are packed in 10 bits, so each in-plane dimension is
   limited to 1024. Hence "ours" only appears for sizes ≤ 1024.

So this benchmark is *not* a claim that our PBA is competitive in 2D — it is a faithful,
verifiable comparison that (a) our 3D PBA computes the exact EDT (matches NUS bit-for-bit and
NPP within truncation), and (b) quantifies the cost of using a 3D code for a 2D task.

## Build & run

Requires the **CUDA Toolkit** (with NPP — included by default). Tested with CUDA 13 on
`sm_120` (RTX 5090) and `sm_89` (RTX 4070).

```bash
make                 # builds ./bench  (nvcc -arch=native)
./bench              # default sizes: 256 512 1024 2048 4096
./bench 512 4096     # custom sizes
```

If `nvcc`/NPP are not on your default path:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Methodology

- Input: a random binary image, ~1% of pixels are sites (fixed seed → reproducible).
- Each implementation gets the **same** sites in its own native input format
  (NPP: 8-bit, sites = 0; NUS: `short2` coords / `MARKER`; ours: 1-byte boundary mask).
- **Compute-only timing**: inputs are uploaded once; only the on-GPU EDT kernels are timed
  (warm-up launch first, then best of 10). Host↔device transfers are excluded.
- **Verification**: the exact distance field of each implementation is compared element-wise
  against the reference (ours for ≤1024, NUS above); `max_err` is reported.

## Layout

```
bench.cu                 unified harness (generate, run all three, time, verify)
Makefile
ours/edt_pba.hpp         our 3D PBA (MIT)
third_party/nus/         NUS PBA+ 2D, ported to CUDA 12+ (removed <device_functions.h>); MIT
results/rtx5090.txt      benchmark log
```

## Credits & licenses

- **NUS PBA+** — Parallel Banding Algorithm, *Cao, Tang, Mohamed, Tan (ACM I3D 2010)*;
  code © 2019 School of Computing, National University of Singapore, MIT-licensed
  (`third_party/nus/LICENSE`). Only `<device_functions.h>` was removed for CUDA 12+; the
  algorithm/kernels are unchanged.
- **NVIDIA NPP** — `nppiDistanceTransformPBA`, part of the CUDA Toolkit; linked at build
  time, not redistributed here.
- **ours** and this benchmark harness — MIT (`LICENSE`).
