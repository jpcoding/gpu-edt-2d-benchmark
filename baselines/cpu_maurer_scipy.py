#!/usr/bin/env python3
"""CPU exact EDT baseline = SciPy's distance_transform_edt (the Maurer 2003 algorithm).

This is the *same* baseline the recent "GPU EDT" papers report their 52x / 250x / 400x
speedups against. We time it here on the same machine as the GPU numbers so the
"inflated baseline" can be quantified directly: PBA-on-GPU vs this, and (the number
those papers omit) PBA-on-GPU vs PBA-in-NPP.

Input mirrors bench.cu: a binary image with `density`% random sites (foreground), the
distance transform measures distance to the nearest site. Single-threaded (as SciPy is).

Usage: cpu_maurer_scipy.py [sizes...] [--density D] [--iters N]
"""
import sys, time, numpy as np
from scipy import ndimage

def run(sizes, density, iters):
    print(f"# CPU baseline: scipy.ndimage.distance_transform_edt (Maurer 2003), "
          f"density={density}%, best of {iters}")
    print(f"{'size':<6} {'best_ms':>10} {'Mpix/s':>10} {'Gpix/s':>10}")
    rng = np.random.default_rng(12345)
    for S in sizes:
        N = S * S
        sites = rng.integers(0, 100, size=(S, S)) < density   # True = site
        # distance_transform_edt computes distance of each True (nonzero) pixel to the
        # nearest zero. We want distance to nearest *site*, so feed the complement:
        # nonzero where NOT a site -> distance to nearest site.
        field = (~sites).astype(np.uint8)
        ndimage.distance_transform_edt(field)                 # warmup
        best = 1e30
        for _ in range(iters):
            t = time.perf_counter()
            ndimage.distance_transform_edt(field)
            best = min(best, time.perf_counter() - t)
        ms = best * 1e3
        print(f"{S:<6} {ms:>10.4f} {N/1e6/best:>10.1f} {N/1e9/best:>10.4f}")

if __name__ == "__main__":
    sizes, density, iters = [], 1, 5
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--density": density = int(args[i+1]); i += 2
        elif args[i] == "--iters": iters = int(args[i+1]); i += 2
        else: sizes.append(int(args[i])); i += 1
    if not sizes: sizes = [1024, 2048, 4096]
    run(sizes, density, iters)
