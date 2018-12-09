//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_NONOGRAMLINEDEVICE_H
#define CODE_NONOGRAMLINEDEVICE_H

#include <cstdlib>

#include "Defs.h"

#include "NonogramColor.h"
#include "Board2DDevice.h"

#define MAX_RUNS 32

typedef struct {
    unsigned topEnd;
    unsigned botStart;
} BRun;

typedef struct {
    unsigned topStart;
    unsigned start;
    unsigned end;
    unsigned botEnd;
} BBlock;

typedef struct {

    unsigned len;
    bool line_is_row;
    unsigned line_index;
    const NonogramColor *data;
    unsigned constr_len;
    unsigned constr[MAX_RUNS];
    BRun b_runs[MAX_RUNS];

} NonogramLineDevice;

__device__ void ngline_dev_runs_fill(NonogramLineDevice *L, Board2DDevice *B);
__device__ void ngline_dev_update(NonogramLineDevice *L, Board2DDevice *B);
__device__ void ngline_dev_block_max_size_fill(NonogramLineDevice *L, Board2DDevice *B,
                                    unsigned i, unsigned curr_bblock_len);
__device__ void ngline_dev_botStart_propagate(NonogramLineDevice *L, Board2DDevice *B,
                                   unsigned ri, unsigned i);
__device__ void ngline_dev_topEnd_propagate(NonogramLineDevice *L, Board2DDevice *B,
                                 unsigned ri, unsigned i);
__device__ void ngline_dev_cell_solve(NonogramLineDevice *L, Board2DDevice *B,
                           NonogramColor color, unsigned i);

bool ng_init(unsigned w, unsigned h, NonogramLineDevice **Ls, Board2DDevice **B);
void ng_free(NonogramLineDevice *Ls, Board2DDevice *B);
bool ng_constr_add(NonogramLineDevice *Ls, unsigned line_index, unsigned constr);
void ng_solve(NonogramLineDevice *Ls_host, Board2DDevice *B_host);


#endif //CODE_NONOGRAMLINEDEVICE_H
