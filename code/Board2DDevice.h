//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_BOARD2DDEVICE_H
#define CODE_BOARD2DDEVICE_H

#include <cstdlib>
#include <cstdio>
#include <iostream>

#ifdef __NVCC__

#else

#ifdef __JETBRAINS_IDE__
#include "CLion.h"
#else
#define __host__
#define __device__
#define __global__
#endif
#endif

#include "NonogramColor.h"

typedef struct {

    unsigned w;
    unsigned h;
    bool dirty;
    NonogramColor *data;
    NonogramColor *dataCM;

} Board2DDevice;

Board2DDevice *board2d_init_host(unsigned w, unsigned h);
Board2DDevice *board2d_init_dev(Board2DDevice *B_host);
void board2d_free_host(Board2DDevice *B);
void board2d_cleanup_dev(Board2DDevice *B_host, Board2DDevice *B_dev);
std::ostream &operator<<(std::ostream &os, Board2DDevice *B);

void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val);
NonogramColor board2d_dev_elem_get_rm(Board2DDevice *B, unsigned x, unsigned y);
NonogramColor board2d_dev_elem_get_cm(Board2DDevice *B, unsigned x, unsigned y);
NonogramColor *board2d_dev_row_ptr_get(Board2DDevice *B, unsigned index);
NonogramColor *board2d_dev_col_ptr_get(Board2DDevice *B, unsigned index);

#endif //CODE_BOARD2DDEVICE_H
