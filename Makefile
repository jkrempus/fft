CXX ?= g++

ifdef DEBUG
	CXXFLAGS += -std=c++17 -g
	#CXXFLAGS += -fsanitize=address 
	#LDFLAGS += -lasan
else
	CXXFLAGS += -std=c++17 -Ofast -ffunction-sections -mtune=native
endif

ifdef FFTW
	CXXFLAGS += -DHAVE_FFTW
	LDFLAGS += libfftw3f.a
endif

.PHONY: all
all: avx2

-include $(wildcard *.d)

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)
	$(CXX) -MM -MF $(<:.cpp=.d) $< $(CXXFLAGS)

test: test.o
	$(CXX) -o $@ $^ $(LDFLAGS)

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
neon: CXXFLAGS += -fPIE -mfpu=neon -mfloat-abi=softfp
neon: LDFLAGS += -pie
neon: test

clean:
	rm *.o *.d
