//
// Created by Benjamin Huang on 12/9/2018.
//

#ifndef CODE_HYPOTHETICALBOARD_H
#define CODE_HYPOTHETICALBOARD_H

#include "Board2DDevice.h"
#include "NonogramLineDevice.h"

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
void nghyp_heuristic_free(Heuristic *X);

void nghyp_heuristic_fill(const NonogramLineDevice *Ls, const Board2DDevice *B, Heuristic *X);
void nghyp_heuristic_update(HypotheticalBoard *H, Heuristic *X, NonogramColor color);
void nghyp_heuristic_max(Heuristic *X);

HypotheticalBoard nghyp_init(NonogramLineDevice *Ls, Board2DDevice *B);
void nghyp_free(HypotheticalBoard H);
void nghyp_hyp_confirm(HypotheticalBoard *H, Board2DDevice **B, NonogramLineDevice **Ls);
bool nghyp_valid_check(HypotheticalBoard *H, Board2DDevice *B);
void nghyp_common_set(HypotheticalBoard *H1, HypotheticalBoard *H2, Board2DDevice *B);


#endif //CODE_HYPOTHETICALBOARD_H
