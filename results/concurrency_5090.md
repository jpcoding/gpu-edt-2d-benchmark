# Concurrency / batching: recovering idle SMs from latency-bound EDT (RTX 5090)

## The question
A single PBA/FH transform is **latency-bound**, not compute- or bandwidth-bound (ncu Speed-of-Light:
no kernel above ~32% SM; the dominant `kernelColor` sits at **6.7% SM / 1.0% DRAM** — stalled on the
linked-list pointer-chase). So one transform leaves most of the GPU idle. The single-kernel levers
(banding/occupancy) are already tuned out. The remaining engineering lever is **concurrency**: run
many *independent* transforms on separate CUDA streams so their latency overlaps and fills the SMs.

Driver: [`prof/concurrency.cu`](../prof/concurrency.cu). M=8 independent FH transforms, serial (one
stream) vs concurrent (M streams), best of 10. FH is used because it is **reentrant** (all buffers
are per-call); see the caveat below.

## Result (RTX 5090, 170 SMs, M=8)

| size | serial ms | concurrent ms | **speedup** | serial GiB/s | concurrent GiB/s |
|---|--:|--:|--:|--:|--:|
| 256²  | 3.16  | 0.46  | **6.92×** | 0.62 | 4.28 |
| 512²  | 6.54  | 0.86  | **7.57×** | 1.19 | 9.04 |
| 1024² | 13.33 | 1.74  | **7.68×** | 2.34 | 18.00 |
| 2048² | 28.51 | 6.71  | **4.25×** | 4.38 | 18.62 |
| 4096² | 67.70 | 19.49 | **3.47×** | 7.39 | 25.66 |

(GiB/s = aggregate 4·M·pixels / 1024³ / s. serial GiB/s == single-transform throughput, since
serial just runs them one at a time.)

## Reading it
- **The speedup tracks the idle capacity, exactly as the latency-bound model predicts.** At small
  sizes one transform uses a sliver of the 170 SMs, so 8-way streaming approaches the **M=8 ceiling
  (~7.7× at 1024²)** — the GPU was ~85–90% idle per transform and concurrency fills it.
- **Diminishing as the transform grows**: at 4096² each transform already occupies much of the GPU,
  so only ~3.5× of overlap is left to recover. The crossover is visible at 2048² (4.25×).
- **Aggregate throughput** rises from 2.3→18.0 GiB/s at 1024² and 7.4→25.7 GiB/s at 4096².

## Engineering takeaway
For a latency-bound EDT at its single-kernel floor, **throughput is won by feeding the GPU more
independent work, not by tuning the kernel further.** For a pipeline that does many EDTs — e.g. many
2D slices of a volume, or several fields — process them **concurrently on streams**. The single-
transform latency is unchanged; aggregate throughput rises up to ~M× (bounded by how much of the GPU
one transform already uses).

## Caveat: PBA needs a reentrancy refactor to batch
This was measured with **FH** because it is stream-safe (per-call buffers). The NUS/NPP **PBA engine
keeps global device buffers** (`pbaTextures`), so two PBA computes cannot run concurrently as-is —
batching the SOTA kernel first requires making it **stream-safe (per-instance state)**. That refactor
is the concrete engineering work to apply this lever to PBA; the speedup ceiling is the same idle-SM
argument shown here.

## Reproduce
```bash
make concurrency
./concurrency 8 10 256 512 1024 2048 4096    # [M] [iters] [sizes...]
```
