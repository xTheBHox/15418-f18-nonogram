//
// Created by Benjamin Huang on 11/19/2018.
//

#ifndef CODE_NONOGRAMLINEDEVICE_H
#define CODE_NONOGRAMLINEDEVICE_H

#include <cstdlib>

#include "Defs.h"

#include "NonogramColor.h"
#include "Board2DDevice.h"

#define MAX_RUNS 16

typedef struct {
    unsigned char topEnd;
    unsigned char botStart;
} BRun;

typedef struct {

    unsigned len;
    bool line_is_row;
    unsigned line_index;
    NonogramColor *data;
    bool solved;
    unsigned constr_len;
    unsigned char constr[MAX_RUNS];
    BRun b_runs[MAX_RUNS];

} NonogramLineDevice;

__device__ __inline__
unsigned dev_max(unsigned a, unsigned b) {
    return a > b ? a : b;
}

__device__ __inline__
unsigned dev_min(unsigned a, unsigned b) {
    return a < b ? a : b;
}

__device__
void ngline_dev_cell_solve(NonogramLineDevice *L, Board2DDevice *B,
                           NonogramColor color, unsigned i);
__device__
void ngline_init_dev(NonogramLineDevice *L);

__device__
void ngline_dev_run_solve(NonogramLineDevice *L, Board2DDevice *B);
__device__
void ngline_dev_block_solve(NonogramLineDevice *L, Board2DDevice *B);
__device__
void ngline_dev_mutableonly_copy(NonogramLineDevice *L_dst, const NonogramLineDevice *L_src);

bool ng_linearr_init_host(unsigned w, unsigned h, NonogramLineDevice **Ls);
NonogramLineDevice *ng_linearr_init_dev(unsigned w, unsigned h, NonogramLineDevice *Ls_host);
void ng_linearr_free_dev(NonogramLineDevice *Ls_dev);
NonogramLineDevice *ng_linearr_deepcopy_host(NonogramLineDevice *Ls, unsigned w, unsigned h);
NonogramLineDevice *ng_linearr_deepcopy_dev_double(NonogramLineDevice *Ls, unsigned Ls_size);
void ng_linearr_board_change(NonogramLineDevice *Ls, Board2DDevice *B);

#endif //CODE_NONOGRAMLINEDEVICE_H
