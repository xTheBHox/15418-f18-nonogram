
CXX_FILES=main.cpp
CU_FILES=Board2DDevice.cu NonogramLineDevice.cu
OBJ_FILES=main.o Board2DDevice.o NonogramLineDevice.o
EXEC=$(EXECPAR)
EXECPAR=ngram-par
EXECSEQ=ngram-seq
CXX=g++ -m64
CXXFLAGS=-Wall -O3 -std=c++11
DBGFLAGS= -D DEBUG -D PERF 
NVCC=nvcc -m64
NVCCFLAGS=-O3 --gpu-architecture compute_35 -std=c++11
NVCCLIBS=-L /usr/local/depot/cuda-8.0/lib64/ -l cudart
SEQDIR=objseq
PARDIR=objpar

OBJSEQ=$(addprefix $(SEQDIR)/, $(OBJ_FILES))
OBJPAR=$(addprefix $(PARDIR)/, $(OBJ_FILES)) 
#OBJ=$(OBJDIR)/main.o $(OBJDIR)/Board2DDevice.o $(OBJDIR)/NonogramLineDevice.o

.PHONY: dirs clean

default: $(EXECPAR) 

seq: clean $(EXECSEQ)

debug: CXXFLAGS += $(DBGFLAGS)
debug: NVCCFLAGS += $(DBGFLAGS)
debug: clean $(EXECPAR)

debug-seq: CXXFLAGS += $(DBGFLAGS)
debug-seq: clean $(EXECSEQ)

dirs:
		mkdir -p $(SEQDIR)/
		mkdir -p $(PARDIR)/

clean:
		rm -rf $(SEQDIR) $(PARDIR) *~ $(EXECSEQ) $(EXECPAR)
 
$(EXECSEQ): dirs $(OBJSEQ)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJSEQ)
 
$(SEQDIR)/%.o: %.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $<

$(SEQDIR)/%.o: %.cu
	$(CXX) -c -o $@ $(CXXFLAGS) -x c++ $<

$(EXECPAR): dirs $(OBJPAR)
	$(NVCC) -o $@ $(NVCCFLAGS) $(NVCCLIBS) $(OBJPAR)
 
$(PARDIR)/%.o: %.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $<
 
$(PARDIR)/%.o: %.cu
	$(NVCC) -c -o $@ $(NVCCFLAGS) $(NVCCLIBS) $<