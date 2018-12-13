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

    unsigned len;
    bool line_is_row;
    unsigned line_index;
    const NonogramColor *data;
    bool solved;
    unsigned constr_len;
    unsigned constr[MAX_RUNS];
    BRun b_runs[MAX_RUNS];

} NonogramLineDevice;

__device__ __inline__
unsigned dev_max(unsigned a, unsigned b);
__device__ __inline__
unsigned dev_min(unsigned a, unsigned b);

__device__
void ngline_dev_cell_solve(NonogramLineDevice *L, Board2DDevice *B,
                           NonogramColor color, unsigned i);
__device__
void ngline_init_dev(NonogramLineDevice *L);
__device__ __inline__
bool ngline_dev_run_top_adjust(NonogramLineDevice *L,
                               unsigned &topEnd, unsigned line_len, unsigned run_len);
__device__ __inline__
bool ngline_dev_run_bot_adjust(NonogramLineDevice *L,
                               unsigned &botStart, unsigned line_len, unsigned run_len);
__device__ __inline__
void ngline_dev_run_top_prop(NonogramLineDevice *L);
__device__ __inline__
void ngline_dev_run_bot_prop(NonogramLineDevice *L);
__device__ __inline__
void ngline_dev_run_fill_black(NonogramLineDevice *L, Board2DDevice *B, const BRun *R, unsigned run_len);
__device__ __inline__
void ngline_dev_run_fill_white(NonogramLineDevice *L, Board2DDevice *B, unsigned ri);

__device__
void ngline_dev_run_solve(NonogramLineDevice *L, Board2DDevice *B, unsigned run_index);
__device__
void ngline_dev_block_solve(NonogramLineDevice *L, Board2DDevice *B);
__device__
void ngline_dev_mutableonly_copy(NonogramLineDevice *L_dst, const NonogramLineDevice *L_src);

bool ng_linearr_init_host(unsigned w, unsigned h, NonogramLineDevice **Ls);
NonogramLineDevice *ng_linearr_init_dev(unsigned w, unsigned h, NonogramLineDevice *Ls_host);
void ng_linearr_free_dev(NonogramLineDevice *Ls_dev);
NonogramLineDevice *ng_linearr_deepcopy_host(NonogramLineDevice *Ls, unsigned w, unsigned h);
void ng_linearr_board_change(NonogramLineDevice *Ls, Board2DDevice *B);

#endif //CODE_NONOGRAMLINEDEVICE_H
