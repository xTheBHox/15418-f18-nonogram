//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_NONOGRAMLINEDEVICE_H
#define CODE_NONOGRAMLINEDEVICE_H

#include "NonogramColor.h"
#include "Nonogram.h"

#include "Board2DDevice.h"

#define MAX_RUNS 32

typedef struct {
    unsigned topEnd;
    unsigned botStart;
} BRun;

typedef struct {

    unsigned len;
    bool line_is_row;
    unsigned line_index;
    const char *data;
    unsigned constr_len;
    unsigned constr[MAX_RUNS];
    BRun b_runs[MAX_RUNS];

} NonogramLineDevice;


void ngline_dev_runs_fill(NonogramLineDevice *L, Board2DDevice *B);
void ngline_dev_update(NonogramLineDevice *L, Board2DDevice *B);
void ngline_dev_block_max_size_fill(NonogramLineDevice *L, Board2DDevice *B,
                                    unsigned i, unsigned curr_bblock_len);
void ngline_dev_botStart_propagate(NonogramLineDevice *L, Board2DDevice *B,
                                   unsigned ri, unsigned i);
void ngline_dev_topEnd_propagate(NonogramLineDevice *L, Board2DDevice *B,
                                 unsigned ri, unsigned i);
void ngline_dev_cell_solve(NonogramLineDevice *L, Board2DDevice *B,
                           char color, unsigned i);

#endif //CODE_NONOGRAMLINEDEVICE_H
