//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_BOARD2DDEVICE_H
#define CODE_BOARD2DDEVICE_H

#ifdef __JETBRAINS_IDE__
#include "math.h"
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __noinline__
#define __forceinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__
// CUDA Synchronization
inline void __syncthreads() {};
inline void __threadfence_block() {};
inline void __threadfence() {};
inline void __threadfence_system();
inline int __syncthreads_count(int predicate) {return predicate};
inline int __syncthreads_and(int predicate) {return predicate};
inline int __syncthreads_or(int predicate) {return predicate};
template<class T> inline T __clz(const T val) { return val; }
template<class T> inline T __ldg(const T* address){return *address};
#endif

typedef struct {

    unsigned w;
    unsigned h;
    char *data;
    char *dataCM;

} Board2DDevice;

#endif //CODE_BOARD2DDEVICE_H
