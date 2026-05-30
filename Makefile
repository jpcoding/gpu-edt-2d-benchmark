NVCC      ?= nvcc
ARCH      ?= native
NVCCFLAGS  = -std=c++17 -O3 -arch=$(ARCH) -Xcompiler -w -Iours -Ithird_party/nus
LDFLAGS    = -lnppif -lnppc -lnppisu

bench: bench.cu third_party/nus/pba2DHost.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

run: bench
	./bench

clean:
	rm -f bench

.PHONY: run clean
