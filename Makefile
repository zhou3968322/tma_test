CUTLASS_DIR=../../cutlass
REPO_DIR=.
CXX=nvcc
APP1=main
# APP2=tma_simple


CXXFLAGS=--generate-code=arch=compute_120a,code=sm_120a -std=c++17 -keep -keep-dir ./ptx_dump -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${REPO_DIR}/include/utils --expt-relaxed-constexpr

LDFLAGS=

LDLIBS=-lcuda

OBJECTS_MAIN      := main.o
# OBJECTS_TMA_SIMPLE := tma_simple.o

.SUFFIXES: .o .cu

default: clean $(APP1)

# default: clean $(APP1) $(APP2)

$(APP1): $(OBJECTS_MAIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS_MAIN) $(LDLIBS)

# $(APP2): $(OBJECTS_TMA_SIMPLE)
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS_TMA_SIMPLE) $(LDLIBS)

%.o: %.cu
	$(CXX) -c $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(OBJECTS_MAIN) $(OBJECTS_TMA_SIMPLE) $(APP1)
