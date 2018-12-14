//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_BOARD2DDEVICE_H
#define CODE_BOARD2DDEVICE_H


#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string.h>

#include "Defs.h"

#include "NonogramColor.h"

typedef struct alignas(8) {

    NonogramColor *data;
    NonogramColor *dataCM;
    unsigned w;
    unsigned h;
    bool dirty;
    bool valid;
    bool solved;

} Board2DDevice;


__host__ __device__ __inline__
void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val) {
    B->data[y * B->w + x] = val;
    B->dataCM[x * B->h + y] = val;
    B->dirty = true;
    
}

__host__ __device__ __inline__
NonogramColor board2d_dev_elem_get_rm(const Board2DDevice *B, unsigned x, unsigned y) {
#ifdef DEBUG
    if (x >= B->w || y >= B->h) {
        printf("nglinehyp_dev_cell_solve error: Tried to access (%d, %d)\n", y, x);
    }
#endif
    return B->data[y * B->w + x];
}

__host__ __device__ __inline__
NonogramColor board2d_dev_elem_get_cm(const Board2DDevice *B, unsigned x, unsigned y) {
#ifdef DEBUG
    if (x >= B->w || y >= B->h) {
        printf("nglinehyp_dev_cell_solve error: Tried to access (%d, %d)\n", y, x);
    }
#endif
    return B->dataCM[x * B->h + y];
}

__host__ __device__ __inline__
NonogramColor *board2d_dev_row_ptr_get(const Board2DDevice *B, unsigned index) {
    return &B->data[index * B->w];
}

__host__ __device__ __inline__
NonogramColor *board2d_dev_col_ptr_get(const Board2DDevice *B, unsigned index) {
    return &B->dataCM[index * B->h];
}

__device__
void board2d_dev_init_copy(Board2DDevice *B_dst, const Board2DDevice *B_src);
__device__
void board2d_dev_mutableonly_copy(Board2DDevice *B_dst, const Board2DDevice *B_src);

Board2DDevice *board2d_init_host(unsigned w, unsigned h);
Board2DDevice *board2d_init_dev(Board2DDevice *B_host);
void board2d_free_host(Board2DDevice *B);
void board2d_cleanup_dev(Board2DDevice *B_host, Board2DDevice *B_dev);
Board2DDevice *board2d_deepcopy_host(Board2DDevice *B);
std::ostream &operator<<(std::ostream &os, Board2DDevice *B);

#endif //CODE_BOARD2DDEVICE_H
