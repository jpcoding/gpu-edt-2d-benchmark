NVCC      ?= nvcc
ARCH      ?= native
NVCCFLAGS  = -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -w -Iours -Ibaselines -Ithird_party/nus
LDFLAGS    = -lnppif -lnppc -lnppisu

bench: bench.cu third_party/nus/pba2DHost.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# NPP-only target for profiling with nsys / ncu (see results/npp_profile_5090.md)
npp_prof: prof/npp_prof.cu
	$(NVCC) -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -w $< -o $@ $(LDFLAGS)

nus_prof: prof/nus_prof.cu third_party/nus/pba2DHost.cu
	$(NVCC) -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -w -Ithird_party/nus $^ -o $@

tune: prof/tune.cu third_party/nus/pba2DHost.cu
	$(NVCC) -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -w -Ithird_party/nus $^ -o $@

run: bench
	./bench

clean:
	rm -f bench npp_prof nus_prof tune

.PHONY: run clean
