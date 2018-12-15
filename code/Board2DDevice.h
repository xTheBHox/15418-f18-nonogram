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
    unsigned pitchRM; // This will be invalid for host board when running parallel
    unsigned pitchCM; // This will be invalid for host board when running parallel
    unsigned w;
    unsigned h;
    bool dirty;
    bool valid;
    bool solved;

} Board2DDevice;



__host__ __inline__
void board2d_host_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val) {
    B->data[y * B->w + x] = val;
    B->dataCM[x * B->h + y] = val;
    B->dirty = true;
}

__host__ __inline__
NonogramColor board2d_host_elem_get_rm(const Board2DDevice *B, unsigned x, unsigned y) {
#ifdef DEBUG
    if (x >= B->w || y >= B->h) {
        printf("nglinehyp_dev_cell_solve error: Tried to access (%d, %d)\n", y, x);
    }
#endif
    return B->data[y * B->w + x];
}

__host__ __inline__
NonogramColor board2d_host_elem_get_cm(const Board2DDevice *B, unsigned x, unsigned y) {
#ifdef DEBUG
    if (x >= B->w || y >= B->h) {
        printf("nglinehyp_dev_cell_solve error: Tried to access (%d, %d)\n", y, x);
    }
#endif
    return B->dataCM[x * B->h + y];
}

__host__ __inline__
NonogramColor *board2d_host_row_ptr_get(const Board2DDevice *B, unsigned index) {
    return &B->data[index * B->w];
}

__host__ __inline__
NonogramColor *board2d_host_col_ptr_get(const Board2DDevice *B, unsigned index) {
    return &B->dataCM[index * B->h];
}

#ifdef __NVCC__

    __device__ __inline__
    void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val) {
        B->data[y * B->pitchRM + x] = val;
        B->dataCM[x * B->pitchCM + y] = val;
        B->dirty = true;
    }

    __device__ __inline__
    NonogramColor board2d_dev_elem_get_rm(const Board2DDevice *B, unsigned x, unsigned y) {
    #ifdef DEBUG
        if (x >= B->w || y >= B->h) {
            printf("nglinehyp_dev_cell_solve error: Tried to access (%d, %d)\n", y, x);
        }
    #endif
        return B->data[y * B->pitchRM + x];
    }

    __device__ __inline__
    NonogramColor board2d_dev_elem_get_cm(const Board2DDevice *B, unsigned x, unsigned y) {
    #ifdef DEBUG
        if (x >= B->w || y >= B->h) {
            printf("nglinehyp_dev_cell_solve error: Tried to access (%d, %d)\n", y, x);
        }
    #endif
        return B->dataCM[x * B->pitchCM + y];
    }

    __device__ __inline__
    NonogramColor *board2d_dev_row_ptr_get(const Board2DDevice *B, unsigned index) {
        return &B->data[index * B->pitchRM];
    }

    __device__ __inline__
    NonogramColor *board2d_dev_col_ptr_get(const Board2DDevice *B, unsigned index) {
        return &B->dataCM[index * B->pitchCM];
    }

#else //__NVCC__

    __inline__
    void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val) {
        board2d_host_elem_set(B, x, y, val);
    }

    __inline__
    NonogramColor board2d_dev_elem_get_rm(const Board2DDevice *B, unsigned x, unsigned y) {
        return board2d_host_elem_get_rm(B, x, y);
    }

    __inline__
    NonogramColor board2d_dev_elem_get_cm(const Board2DDevice *B, unsigned x, unsigned y) {
        return board2d_host_elem_get_cm(B, x, y);
    }

    __inline__
    NonogramColor *board2d_dev_row_ptr_get(const Board2DDevice *B, unsigned index) {
        return board2d_host_row_ptr_get(B, index);
    }

    __inline__
    NonogramColor *board2d_dev_col_ptr_get(const Board2DDevice *B, unsigned index) {
        return board2d_host_col_ptr_get(B, index);
    }

#endif // __NVCC__

__host__ __device__
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
