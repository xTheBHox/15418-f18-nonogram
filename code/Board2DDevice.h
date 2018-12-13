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

typedef struct {

    unsigned w;
    unsigned h;
    bool dirty;
    NonogramColor *data;
    NonogramColor *dataCM;
    bool valid;
    bool solved;

} Board2DDevice;

__device__  __inline__
void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val);
__host__ __device__  __inline__
NonogramColor board2d_dev_elem_get_rm(const Board2DDevice *B, unsigned x, unsigned y);
__host__ __device__  __inline__
NonogramColor board2d_dev_elem_get_cm(const Board2DDevice *B, unsigned x, unsigned y);
__host__ __device__  __inline__
NonogramColor *board2d_dev_row_ptr_get(const Board2DDevice *B, unsigned index);
__host__ __device__  __inline__
NonogramColor *board2d_dev_col_ptr_get(const Board2DDevice *B, unsigned index);
__device__ void board2d_dev_mutableonly_copy(Board2DDevice *B_dst, const Board2DDevice *B_src);

Board2DDevice *board2d_init_host(unsigned w, unsigned h);
Board2DDevice *board2d_init_dev(Board2DDevice *B_host);
void board2d_free_host(Board2DDevice *B);
void board2d_cleanup_dev(Board2DDevice *B_host, Board2DDevice *B_dev);
Board2DDevice *board2d_deepcopy_host(Board2DDevice *B);
std::ostream &operator<<(std::ostream &os, Board2DDevice *B);

#endif //CODE_BOARD2DDEVICE_H
