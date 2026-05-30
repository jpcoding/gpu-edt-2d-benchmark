# GPU 2D Euclidean Distance Transform — three implementations, head to head

A small, self-contained benchmark comparing three GPU implementations of the **exact 2D
Euclidean Distance Transform / Voronoi diagram** on random binary images:

| # | implementation | kind | source | license |
|---|---|---|---|---|
| 1 | **NVIDIA NPP** `nppiDistanceTransformPBA` | 2D-native, vendor library | ships with the CUDA Toolkit | NVIDIA EULA (linked, not redistributed) |
| 2 | **NUS PBA+** `pba2D` | 2D-native, academic reference | [orzzzjq/Parallel-Banding-Algorithm-plus](https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus) | MIT (`third_party/nus/LICENSE`) |
| 3 | **"ours"** `edt_2d_pba` / `edt_3d_pba` | **3D** Parallel Banding, in a 2D-specialized driver and run as `W×H×1` | this project | MIT |

All three are fed the **same** random binary image, timed **compute-only** (data already
device-resident, warm-up + best-of-10), and **cross-verified** to produce the same EDT.

## Results (NVIDIA RTX 5090, ~1% random sites)

```
size   impl            time_ms       Mpix/s       Gpix/s    max_err
256    ours-3Don2D      0.2289        286.3        0.286        ref
256    ours-2D          0.2269        288.8        0.289       0.00
256    NUS-PBA+         0.1094        599.2        0.599       0.00
256    NPP              0.0810        809.2        0.809       0.97
512    ours-3Don2D      0.4622        567.2        0.567        ref
512    ours-2D          0.4597        570.3        0.570       0.00
512    NUS-PBA+         0.1553       1687.9        1.688       0.00
512    NPP              0.1068       2455.2        2.455       0.97
1024   ours-3Don2D      0.9404       1115.0        1.115        ref
1024   ours-2D          0.9316       1125.5        1.126       0.00
1024   NUS-PBA+         0.2368       4428.2        4.428       0.00
1024   NPP              0.1607       6526.0        6.526       0.97
2048   NUS-PBA+         0.4230       9915.6        9.916       0.00
2048   NPP              0.3033      13826.9       13.827       0.97
4096   NUS-PBA+         1.0900      15392.6       15.393       0.00
4096   NPP              0.9810      17101.7       17.102       0.97
```

(full log: [`results/rtx5090.txt`](results/rtx5090.txt))

**Reading the table**
- `max_err` is the largest distance disagreement vs. the reference field. **NUS = 0.00**
  (bit-exact with ours); **NPP = 0.97** — NPP returns the distance as a *truncated* 16-bit
  integer, so it is off by `< 1` everywhere, i.e. correct. All three agree.
- **NPP** is fastest, **NUS PBA+** close behind; both scale to large images.
- **The two "ours" rows are bit-exact (max_err 0.00) but ~3.9× slower** at 1024². They are
  included as a *correctness cross-check* of our 3D PBA, not as competitive 2D codes — see why below.

## Why "ours" is slower (and capped at 1024)

The two `ours` rows are this project's **3D** Parallel Banding code applied to a 2D image:
- `ours-3Don2D` — the image as a `W×H×1` volume (the 3D axis chooser pads depth 1 → 4).
- `ours-2D` — a 2D-specialized driver ([`ours/edt_2d.hpp`](ours/edt_2d.hpp)) that drives the
  *same kernels* with `z_size = 1` (no depth padding).

They land within ~1% of each other — so the **depth padding is essentially free** (the extra
z-planes are empty and early-out). The real gap to native-2D PBA is **structural**:

> Our pipeline is `FloodZ → Maurer/Color → Maurer/Color`. For a 2D image the cheap flood pass
> is spent on the *trivial* Z axis, so **both** real axes (X and Y) must be resolved by the two
> *expensive* Maurer/Color passes. A native 2D PBA (NUS) spends its flood on a **real** axis and
> needs only **one** proximate/color pass — roughly half the expensive work.

Closing that gap would require **native 2D kernels** (i.e. re-implementing the 2D algorithm),
not reusing the 3D one. There is also a hard **in-plane cap of 1024** (coordinates are packed in
10 bits), so the `ours` rows only appear for sizes ≤ 1024.

So this benchmark is *not* a claim that our PBA is competitive in 2D. It is a faithful,
verifiable comparison showing (a) our 3D PBA computes the exact EDT (bit-for-bit with NUS,
within truncation of NPP — including via the native-2D driver), and (b) what it costs to use a
3D-structured PBA for a 2D task.

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
ours/edt_2d.hpp          native-2D driver over the same kernels (z_size=1), MIT
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
