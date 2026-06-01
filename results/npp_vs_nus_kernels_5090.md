# NPP vs NUS PBA+, kernel for kernel (RTX 5090, CUDA 13)

NPP's `nppiDistanceTransformPBA` **is** the NUS Parallel Banding Algorithm — same kernels, same
names. This profiles **both** per-kernel on identical ~1% inputs so we can see exactly what
NVIDIA's tuning changes, and what its convenience API costs. Drivers:
[`prof/npp_prof.cu`](../prof/npp_prof.cu), [`prof/nus_prof.cu`](../prof/nus_prof.cu) (NUS at tuned
bands m1=tex/64, m2=64, m3=16). nsys per-kernel average over 20 iterations; ncu for launch configs.

## Per-kernel time, 4096² (avg µs/iter, nsys `cuda_gpu_kern_sum`)

| kernel | NPP | NUS-tuned | note |
|---|--:|--:|---|
| `kernelColor` (phase 3) | **318.8** | **282.0** | NPP `m3=8` vs NUS `m3=16` → NUS 13% faster |
| `kernelProximatePoints` (phase 2) | **121.1** | **104.7** | NPP uses half the phase-2 bands → NUS 16% faster |
| `kernelDoubleToSingleList` | 120.0 | 120.2 | identical |
| `kernelFloodDown` | 72.7 | 71.1 | identical |
| `kernelUpdateVertical` | 66.4 | 67.8 | identical |
| `kernelFloodUp` | 30.6 | 31.5 | identical |
| `kernelCreateForwardPointers` | 19.2 | 15.0 | ~ |
| `kernelMergeBands` | 7.2 | 7.9 | identical |
| `kernelPropagateInterband` | 6.2 | 6.2 | identical |
| **`initInputVoronoi`** (8u→short2) | **135.6** | — | **NPP-only format conversion** |
| **`generateAdditionalRelativeOutput_16u`** (→16-bit dist) | **75.9** | — | **NPP-only format conversion** |
| **Σ total** | **≈ 974** | **≈ 706** | NUS-tuned ~1.38× faster |

## The gap decomposes cleanly (NPP − NUS ≈ 268 µs @4096²)

| source | µs | what it is |
|---|--:|---|
| NPP format-conversion kernels | **+211** | `initInputVoronoi` + `generateAdditionalRelativeOutput` — the price of NPP's 8-bit-in / 16-bit-out convenience API; the PBA core never needs them |
| `kernelColor` band | **+37** | NPP's adaptive heuristic picks `m3=8` at 4096²; `m3=16` (NUS) is faster on Blackwell |
| `kernelProximatePoints` band | **+16** | NPP runs phase 2 with half as many bands |
| everything else | ~0 | the shared PBA kernels are identical in config and time |

So NUS-tuned's end-to-end win over NPP is **~80% format-conversion overhead** and **~20% two
band choices** — *not* a better algorithm. It is literally the same kernels.

## Launch configs that cause the two differences (ncu)

`kernelColor` block `(64, m3)`; `kernelProximatePoints` parallelism = grid.y (phase-2 bands):

| size | kernel | NPP block / grid.y | NUS block / grid.y |
|---|---|---|---|
| 1024² | kernelColor | **1024** (m3=16) | 1024 (m3=16) — *tie* |
| 4096² | kernelColor | **512** (m3=8) | **1024** (m3=16) |
| 1024² | kernelProximatePoints | grid.y **32** | grid.y **64** |
| 4096² | kernelProximatePoints | grid.y **32** | grid.y **64** |

NPP keeps phase-2 at 32 bands and drops phase-3 to `m3=8` at 4096²; NUS-tuned uses 64 phase-2
bands and `m3=16` throughout. At ≤1024² the two pick the same `kernelColor` config and tie there
(matching the end-to-end benchmark, where NPP ties at small sizes and is overtaken at 4096²).

## Trend across sizes (kernelColor / Proximate, avg µs)

| size | Color NPP | Color NUS | Proximate NPP | Proximate NUS |
|---|--:|--:|--:|--:|
| 1024² | 61.3 | 61.4 | 15.0 | 8.5 |
| 2048² | 121.0 | 121.1 | 36.2 | 26.5 |
| 4096² | 318.8 | 282.0 | 121.1 | 104.7 |

`kernelColor` ties until NPP changes the band at 4096²; `kernelProximatePoints` is slower in NPP
at every size (fewer phase-2 bands). Conversion overhead grows with size and dominates NPP's gap.

## Takeaways

1. **Same algorithm, confirmed kernel-for-kernel.** Every PBA phase appears in both with matching
   shared-kernel times. The "PBA" in `nppiDistanceTransformPBA` is exactly the NUS kernels.
2. **NPP's real cost is its API, not its core**: ~211 µs/iter at 4096² of 8u→short2 and →16u
   conversion that a `short2`-native caller (NUS, ours) never pays.
3. **NVIDIA's band heuristic is not Blackwell-optimal at large sizes** (`m3=8` on phase 3,
   32-band phase 2) — tuned PBA beats it on those two kernels.

## Reproduce
```bash
make npp_prof nus_prof
nsys profile --stats=true ./npp_prof 4096 20    # per-kernel time (cuda_gpu_kern_sum)
nsys profile --stats=true ./nus_prof 4096 20
ncu --kernel-name "regex:kernelColor|kernelProximatePoints" --launch-count 2 \
    --metrics launch__block_size,launch__grid_size ./npp_prof 4096 1
```
