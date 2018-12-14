//
// Created by Benjamin Huang on 12/9/2018.
//

#ifndef CODE_HYPOTHETICALBOARD_H
#define CODE_HYPOTHETICALBOARD_H

#include "Board2DDevice.h"
#include "NonogramLineDevice.h"
#ifdef DISP
#include <ncurses.h>
#endif

typedef struct _HypotheticalBoard HypotheticalBoard;

struct _HypotheticalBoard {
    NonogramLineDevice *Ls;
    Board2DDevice *B;
    unsigned row;
    unsigned col;
    NonogramColor guess_color;
};

typedef struct {
    unsigned w;
    unsigned h;
    char *data;
    unsigned r_max;
    unsigned c_max;
} Heuristic;

__device__
void nglinehyp_dev_run_solve(NonogramLineDevice *L, Board2DDevice *B, unsigned run_index);
__device__
void nglinehyp_dev_block_solve(NonogramLineDevice *L, Board2DDevice *B);

__device__
void nghyp_heuristic_cell(const NonogramLineDevice *Ls, const Board2DDevice *B, Heuristic *X, unsigned r, unsigned c);

bool nghyp_heuristic_init(NonogramLineDevice *Ls, Board2DDevice *B, Heuristic **X);
Heuristic *nghyp_heuristic_init_dev(unsigned w, unsigned h);
void nghyp_heuristic_free(Heuristic *X);
void nghyp_heuristic_free_dev(Heuristic *X_dev);

void nghyp_heuristic_fill(const NonogramLineDevice *Ls, const Board2DDevice *B, Heuristic *X, unsigned w, unsigned h);
__device__
void nghyp_heuristic_update(HypotheticalBoard *H, Heuristic *X, NonogramColor color);
void nghyp_heuristic_max(Heuristic *X);
void nghyp_heuristic_max_dev(Heuristic *X, unsigned X_data_len);

HypotheticalBoard nghyp_init(NonogramLineDevice *Ls, Board2DDevice *B);
void nghyp_free(HypotheticalBoard H);
void nghyp_hyp_confirm(HypotheticalBoard *H, Board2DDevice **B, NonogramLineDevice **Ls);
bool nghyp_valid_check(HypotheticalBoard *H, Board2DDevice *B);

__device__ __inline__
void nghyp_hyp_assume(NonogramLineDevice *L_hyp, const NonogramLineDevice *L_global, Board2DDevice *B_global, unsigned i) {

    if (L_global->data[i] == NGCOLOR_UNKNOWN) {
        switch (L_hyp->data[i]) {
        case NGCOLOR_UNKNOWN:
            return;
        case NGCOLOR_WHITE:
            L_global->data[i] = NGCOLOR_HYP_WHITE;
            return;
        case NGCOLOR_BLACK:
            L_global->data[i] = NGCOLOR_HYP_BLACK;
            return;
        default: return;
        }
    }
    
}

__device__ __inline__
void nghyp_hyp_unassume(NonogramLineDevice *L_hyp, const NonogramLineDevice *L_global, Board2DDevice *B_global, unsigned i) {

    if (L_global->data[i] == NGCOLOR_HYP_WHITE) {
        if (L_hyp->data[i] == NGCOLOR_WHITE) {
            L_global->data[i] = NGCOLOR_WHITE;
            B_global->dirty = true;
        }
        else {
            L_global->data[i] = NGCOLOR_UNKNOWN;
        }
        return;
    }
    if (L_global->data[i] == NGCOLOR_HYP_BLACK) {
        if (L_hyp->data[i] == NGCOLOR_BLACK) {
            L_global->data[i] = NGCOLOR_BLACK;
            B_global->dirty = true;
        }
        else {
            L_global->data[i] = NGCOLOR_UNKNOWN;
        }
        return;
    }
    
}

__device__ __inline__
void nghyp_confirm_assume(NonogramLineDevice *L_hyp, const NonogramLineDevice *L_global, Board2DDevice *B_global, unsigned i) {

    if (L_global->data[i] == NGCOLOR_UNKNOWN && L_hyp->data[i] != NGCOLOR_UNKNOWN) {
       L_global->data[i] = L_hyp->data[i];
       B_global->dirty = true;
    }
    
}

__device__ __inline__
void nghyp_confirm_unassume(const NonogramLineDevice *L_global, Board2DDevice *B_global, unsigned i) {

    switch (L_global->data[i]) {
        case NGCOLOR_HYP_WHITE:
            L_global->data[i] = NGCOLOR_WHITE;
            B_global->dirty = true;
            return;
        case NGCOLOR_HYP_BLACK:
            L_global->data[i] = NGCOLOR_BLACK;
            B_global->dirty = true;
        
        default: return;
    }
    
}

void nghyp_common_set(HypotheticalBoard *H1, HypotheticalBoard *H2, Board2DDevice *B);


#endif //CODE_HYPOTHETICALBOARD_H
