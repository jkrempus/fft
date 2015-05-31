CXX ?= g++

ifdef DEBUG
	CXXFLAGS += -std=c++11 -g
else
	CXXFLAGS += -std=c++11 -Ofast
endif

.PHONY: all
all: avx2

%.o: %.c fft_core.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

test: test.o
	$(CXX) -o $@ $^ $(CFLAGS)

.PHONY: sse
sse: CXXFLAGS += -msse2
sse: test

.PHONY: avx
avx: CXXFLAGS += -mavx
avx: test

.PHONY: avx2
avx2: CXXFLAGS += -mavx2 -mfma
avx2: test

.PHONY: scalar
scalar: CXXFLAGS += -DNO_SIMD
scalar: test

.PHONY: neon
neon: CXXFLAGS += -fPIE -mfpu=neon -mfloat-abi=softfp -mcpu=cortex-a15
neon: test

clean:
	rm *.o
