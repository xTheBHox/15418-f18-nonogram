
EXEC=ngram
CXX=g++ -m64 -std=c++11
CXXFLAGS=-Wall -O3
DBGFLAGS=-D DEBUG -D PERF 
CXX_FILES=main.cpp
CU_FILES=Board2DDevice.cpp NonogramLineDevice.cpp
OBJDIR=objs
OBJ=$(OBJDIR)/main.o $(OBJDIR)/Board2DDevice.o $(OBJDIR)/NonogramLineDevice.o

.PHONY: dirs clean

default: $(EXEC)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXEC)

$(EXEC): dirs $(OBJ)
	$(CXX) -o $@ $(CFLAGS) $(DBGFLAGS) $(OBJ)
 
$(OBJDIR)/%.o: %.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(DBGFLAGS) $<