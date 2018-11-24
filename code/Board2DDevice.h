//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_BOARD2DDEVICE_H
#define CODE_BOARD2DDEVICE_H

#include <cstdlib>
#include <cstdio>
#include <iostream>

#ifdef __JETBRAINS_IDE__
#include "CLion.h"
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

void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, char val);
char board2d_dev_elem_get_rm(Board2DDevice *B, unsigned x, unsigned y);
char board2d_dev_elem_get_cm(Board2DDevice *B, unsigned x, unsigned y);
char *board2d_dev_row_ptr_get(Board2DDevice *B, unsigned index);
char *board2d_dev_col_ptr_get(Board2DDevice *B, unsigned index);

#endif //CODE_BOARD2DDEVICE_H
