# GPU 2D Euclidean Distance Transform — three implementations, head to head

A small, self-contained benchmark comparing three GPU implementations of the **exact 2D
Euclidean Distance Transform / Voronoi diagram** on random binary images:

| # | implementation | kind | source | license |
|---|---|---|---|---|
| 1 | **NVIDIA NPP** `nppiDistanceTransformPBA` | 2D-native, vendor library | ships with the CUDA Toolkit | NVIDIA EULA (linked, not redistributed) |
| 2 | **NUS PBA+** `pba2D` | 2D-native, academic reference | [orzzzjq/Parallel-Banding-Algorithm-plus](https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus) | MIT (`third_party/nus/LICENSE`) |
| 3 | **"ours-2D"** `edt_2d_pba` | **native 2D** EDT with our 3D-style device interface (over the NUS 2D kernels) | this project | MIT |
| 4 | **"ours-3Don2D"** `edt_3d_pba` | our **3D** PBA run as `W×H×1` (shortcut, for contrast) | this project | MIT |

All three are fed the **same** random binary image, timed **compute-only** (data already
device-resident, warm-up + best-of-10), and **cross-verified** to produce the same EDT.

## Results (NVIDIA RTX 5090, ~1% random sites)

```
size   impl            time_ms       Mpix/s       Gpix/s    max_err
256    ours-3Don2D      0.2279        287.6        0.288        ref
256    ours-2D          0.1128        580.8        0.581       0.00
256    NUS-PBA+         0.1105        593.1        0.593       0.00
256    NPP              0.0807        811.6        0.812       0.97
512    ours-3Don2D      0.4613        568.3        0.568        ref
512    ours-2D          0.1583       1655.6        1.656       0.00
512    NUS-PBA+         0.1556       1684.4        1.684       0.00
512    NPP              0.1074       2441.8        2.442       0.97
1024   ours-3Don2D      0.9384       1117.4        1.117        ref
1024   ours-2D          0.2438       4301.2        4.301       0.00
1024   NUS-PBA+         0.2379       4408.4        4.408       0.00
1024   NPP              0.1611       6508.6        6.509       0.97
2048   ours-2D          0.4424       9481.5        9.482       0.00
2048   NUS-PBA+         0.4239       9895.3        9.895       0.00
2048   NPP              0.3027      13856.3       13.856       0.97
4096   ours-2D          1.2400      13530.4       13.530       0.00
4096   NUS-PBA+         1.0926      15355.5       15.356       0.00
4096   NPP              0.9810      17102.0       17.102       0.97
```

(full log: [`results/rtx5090.txt`](results/rtx5090.txt))

> **nsys/ncu profiling of NPP** ([`results/npp_profile_5090.md`](results/npp_profile_5090.md))
> reveals that NPP's `nppiDistanceTransformPBA` runs the **NUS PBA+ kernels** (`kernelFloodDown`,
> `kernelProximatePoints`, `kernelColor`, …) plus its own format-conversion kernels — the "PBA" in
> the name *is* the Parallel Banding Algorithm. So all three implementations here run the same core
> algorithm; the dominant kernel (`kernelColor`) is latency-bound and no kernel is compute-bound.

**Reading the table**
- `max_err` is the largest distance disagreement vs. the reference field. **ours-2D / NUS = 0.00**
  (bit-exact, including vs. the independent 3D code); **NPP = 0.97** — NPP returns the distance as a
  *truncated* 16-bit integer, so it is `< 1` off everywhere, i.e. correct. Everything agrees.
- **NPP** is fastest; **`ours-2D` and NUS-PBA+ are within ~2–12% of each other** and both scale to
  4096². `ours-2D` runs at every size (no 1024 cap).
- **`ours-3Don2D`** (the 3D code run as `W×H×1`) is ~3.9× slower and capped at 1024 — kept only to
  show the cost of that shortcut.

## The two "ours" entries

- **`ours-2D`** ([`ours/edt_2d.hpp`](ours/edt_2d.hpp)) is a **native 2D EDT** that exposes the same
  device interface as this project's 3D EDT — a 1-byte boundary map in, a packed nearest-site
  `index` + `float distance` out, all device-resident — so it drops into the 2D version of our
  pipeline. Its core is the **NUS PBA+ 2D kernels** (MIT), driven through a thin device bridge,
  exactly as our 3D EDT wraps the NUS 3D kernels. It is therefore ~as fast as NUS; the small gap
  (and its growth at 4096²) is the cost of our `boundary → index/distance` conversion kernels.
  16-bit coordinates ⇒ **no 1024 cap** (up to 32767/axis).

- **`ours-3Don2D`** ([`ours/edt_pba.hpp`](ours/edt_pba.hpp)) is our **3D** PBA applied to a `W×H×1`
  volume. It is ~3.9× slower and capped at 1024, *not* because of depth padding (that is nearly
  free — the padded z-planes are empty), but **structurally**: the 3D pipeline spends its cheap
  flood pass on the trivial Z axis, so both real axes are resolved by two *expensive* Maurer/Color
  passes, where a native 2D PBA spends its flood on a real axis and needs only one. It is kept as a
  correctness cross-check (bit-for-bit with the native path) and to quantify that shortcut.

So: `ours-2D` is a competitive, exact, uncapped native-2D EDT with our pipeline's interface;
`ours-3Don2D` shows what using the 3D code for 2D costs. Both agree bit-for-bit with NUS and within
truncation of NPP.

## Build & run

Requires the **CUDA Toolkit** (with NPP — included by default). Tested with CUDA 13 on
`sm_120` (RTX 5090) and `sm_89` (RTX 4070).

```bash
make                 # builds ./bench  (nvcc -arch=native)
./bench              # default sizes: 256 512 1024 2048 4096   (random sites)
./bench 512 4096     # custom sizes
./bench nyx field.f32 512   # real data: binary image = quantization edges of a
                            # 512x512 slice of a NYX field (rel_eb=1e-2)
```

The `nyx` mode is a real-data correctness check: on a 512² slice of NYX `velocity_x`
(~35% edge sites) on the RTX 5090, `ours-2D` matches NUS/the 3D code bit-for-bit
(`max_err 0.00`) and stays competitive (1.10 vs NUS 1.12 Gpix/s) — see
[`results/rtx5090_nyx.txt`](results/rtx5090_nyx.txt).

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
