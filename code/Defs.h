#ifndef CODE_DEFS_H
#define CODE_DEFS_H

#ifdef __NVCC__
#ifdef DEBUG

#include <cstdio>

#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s at %s:%d\n",
        code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#else // __DEBUG__

#define cudaCheckError(ans) ans

#endif // __DEBUG__
#else // __NVCC__
#ifdef __JETBRAINS_IDE__
#include "CLion.h"
#else
#define __host__
#define __device__
#define __global__
#endif

#endif // __NVCC__

#ifdef PERF
#include <chrono>
#define TIMER_START(timer_name) \
auto __time ## timer_name = std::chrono::high_resolution_clock::now()

#define TIMER_STOP(timer_name) \
std::cout << #timer_name << ": " << \
std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - __time ## timer_name ).count() \
<< " ns" << std::endl

#else
#define TIMER_START(timer_name)
#define TIMER_STOP(timer_name)
#endif

#endif // CODE_DEFS_H
