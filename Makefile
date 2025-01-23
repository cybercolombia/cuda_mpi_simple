# Compiler and flags
NVCC = nvcc
MPICXX = mpicxx
CXXFLAGS = -O2

# Targets
TARGET = mpi_cuda.exe
CUDA_OBJ = mm_cuda.o
MPI_OBJ = mpi_cuda.o

# Rules
all: $(TARGET)

$(CUDA_OBJ): mm_cuda.cu
	$(NVCC) -c $< -o $@

$(MPI_OBJ): mpi_cuda.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(CUDA_OBJ) $(MPI_OBJ)
	$(MPICXX) $(CXXFLAGS) $^ -o $@ -lcudart

clean:
	rm -f $(CUDA_OBJ) $(MPI_OBJ) $(TARGET)

.PHONY: all clean