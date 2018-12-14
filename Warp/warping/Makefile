EXECUTABLE = warp_image
OBJS = build/mLibSource.o build/main.o
CXX = clang++
LINK = clang++
NVCC = /usr/local/cuda/bin/nvcc
SRC ?= src
CUDAHOME += /usr/local/cuda

ifeq ($(UNAME), Linux)
  S4 = 64
endif

FLAGS += -O3 -g -std=c++11 -I$(SRC) -I$(SRC)/cutil/inc -I../API/release/include -I$(CUDAHOME)/include -I${CUDA_HOME}/include -I../external/mLib/include
NVCCFLAGS += -arch sm_30 $(FLAGS)
LFLAGS += -std=c++11 -L$(CUDAHOME)/lib$(S4) -L${CUDA_HOME}/lib$(S4) -lcudart -ldl
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -Wl,-rpath,$(CUDAHOME)/lib$(S4) -Wl,-rpath,${CUDA_HOME}/lib$(S4) -pthread -ltinfo
  S4 = 64
endif

.SUFFIXES:
.PHONY:	all purge clean

all: $(EXECUTABLE)

build/%.o:	$(SRC)/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) -MP -MD $< -c -o $@
build/%.o:	$(SRC)/%.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(EXECUTABLE):	$(OBJS)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
