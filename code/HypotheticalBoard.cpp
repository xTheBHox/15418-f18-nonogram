//
// Created by Benjamin Huang on 12/9/2018.
//

#include "HypotheticalBoard.h"
#include "NonogramLineDevice.h"
#include "Board2DDevice.h"

__device__ __inline__
void nglinehyp_dev_cell_solve(NonogramLineDevice *L, Board2DDevice *B,
                           NonogramColor color, unsigned i) {
    if (L->data[i] == color) return;
    if (L->data[i] == NGCOLOR_UNKNOWN) {
        unsigned x, y;
        if (L->line_is_row) {
            y = L->line_index;
            x = i;
        }
        else {
            x = L->line_index;
            y = i;
        }
        board2d_dev_elem_set(B, x, y, color);
        return;
    }

    // We have found a contradiction
    B->valid = false;

}


__device__ __inline__
bool nglinehyp_dev_run_top_adjust(NonogramLineDevice *L, Board2DDevice *B,
                                  unsigned &topEnd, unsigned line_len, unsigned run_len) {

    if (topEnd < line_len && L->data[topEnd] == NGCOLOR_BLACK) {
        topEnd++;
        return true;
    }

    for (unsigned i = topEnd; i > topEnd - run_len; i--) {
        if (L->data[i - 1] == NGCOLOR_WHITE) {
            if (i + run_len > line_len) {
                B->valid = false;
                return false;
            }
            topEnd = i + run_len;
            return true;
        }
    }

    if (topEnd > run_len && L->data[topEnd - run_len - 1] == NGCOLOR_BLACK) {
        if (topEnd == line_len) {
            B->valid = false;
            return false;
        }
        topEnd++;
        return true;
    }

    return false;

}

__device__ __inline__
bool nglinehyp_dev_run_bot_adjust(NonogramLineDevice *L, Board2DDevice *B,
                                  unsigned &botStart, unsigned line_len, unsigned run_len) {

    if (botStart > 0 && L->data[botStart - 1] == NGCOLOR_BLACK) {
        botStart--;
        return true;
    }

    for (unsigned i = botStart; i < botStart + run_len; i++) {
        if (L->data[i] == NGCOLOR_WHITE) {
            if (i < run_len) {
                B->valid = false;
                return false;
            }
            botStart = i - run_len;
            return true;
        }
    }

    if (botStart + run_len < line_len && L->data[botStart + run_len] == NGCOLOR_BLACK) {
        if (botStart == 0) {
            B->valid = false;
            return false;
        }
        botStart--;
        return true;
    }

    return false;

}


__device__ __inline__
void nglinehyp_dev_run_fill_black(NonogramLineDevice *L, Board2DDevice *B, const BRun *R, unsigned run_len) {

    if (R->topEnd > R->botStart + run_len) {
        B->valid = false;
        return;
    }
    if (R->topEnd == R->botStart + run_len) {
        if (R->topEnd < L->len) nglinehyp_dev_cell_solve(L, B, NGCOLOR_WHITE, R->topEnd);
        if (R->botStart > 0) nglinehyp_dev_cell_solve(L, B, NGCOLOR_WHITE, R->botStart - 1);
    }
    for (unsigned i = R->botStart; i < R->topEnd; i++) {
        nglinehyp_dev_cell_solve(L, B, NGCOLOR_BLACK, i);
    }

}

__device__ __inline__
void nglinehyp_dev_run_fill_white(NonogramLineDevice *L, Board2DDevice *B, unsigned ri) {
// ri is the index of the black run after the white area
    unsigned prevBotEnd;
    unsigned topStart;
    if (ri == 0) {
        prevBotEnd = 0;
    }
    else {
        prevBotEnd = L->b_runs[ri - 1].botStart + L->constr[ri - 1];
    }
    if (ri == L->constr_len) {
        topStart = L->len;
    }
    else {
        topStart = L->b_runs[ri].topEnd - L->constr[ri];
    }

    for (unsigned i = prevBotEnd; i < topStart; i++) {
        nglinehyp_dev_cell_solve(L, B, NGCOLOR_WHITE, i);
    }

}

__device__
void nglinehyp_dev_run_solve(NonogramLineDevice *L, Board2DDevice *B, unsigned run_index) {

#ifdef DEBUG
    if (run_index >= L->constr_len) return;
#endif

    BRun *R = &L->b_runs[run_index];
    unsigned run_len = L->constr[run_index];
    unsigned line_len = L->len;

    // Adjust the possible start and end points of the runs

    while(nglinehyp_dev_run_top_adjust(L, B, R->topEnd, line_len, run_len)) {
        if (!B->valid) return;
    }
    while(nglinehyp_dev_run_bot_adjust(L, B, R->botStart, line_len, run_len)) {
        if (!B->valid) return;
    }

    // Propagate changes - one thread only!

    ngline_dev_run_top_prop(L);
    ngline_dev_run_bot_prop(L);

    // Fill overlaps

    nglinehyp_dev_run_fill_black(L, B, R, run_len);
    nglinehyp_dev_run_fill_white(L, B, run_index);
    if (run_index == 0) nglinehyp_dev_run_fill_white(L, B, L->constr_len);

}

__device__
void nglinehyp_dev_block_solve(NonogramLineDevice *L, Board2DDevice *B) {

    unsigned block_topStart = 0;
    unsigned block_start;
    unsigned block_end;
    unsigned block_botEnd = 0;
    unsigned ri_first = 0;
    unsigned ri_last = 0;
    unsigned i = 0;

    bool solved = true;

    while (i < L->len) {

        if (L->data[i] == NGCOLOR_UNKNOWN) {
            // No blocks to solve here
            i++;
            solved = false;
            continue;
        }
        if (L->data[i] == NGCOLOR_WHITE) {
            // White blocks, nothing to solve
            i++;
            block_topStart = i;
            continue;
        }

        // At this point we have reached a shaded block
        block_start = i;
        while (L->data[i] == NGCOLOR_BLACK) {
            // Find the end of the shaded block
            i++;
            if (i == L->len) break;
        }
        block_end = i;
        block_botEnd = i;
        // Find the next white block (to determine the maximum possible extent of the shaded block
        while (block_botEnd < L->len && L->data[block_botEnd] != NGCOLOR_WHITE) block_botEnd++;
        // Determine the minimum and maximum length of this block
        unsigned block_len_min = block_end - block_start;
        unsigned block_len_max = block_botEnd - block_topStart;

        // The number of runs that will fit this block
        unsigned run_fit_count = 0;
        // next three are not valid if run_fit_count == 0
        unsigned run_fit_index = 0; // The bottom-most fitting run
        unsigned run_len_min = L->len; // The minimum length of all the fitting runs
        unsigned run_len_max = 0; // The maximum length of all the fitting runs

        // Get the run valid run indexes
        while (ri_first < L->constr_len && L->b_runs[ri_first].botStart + L->constr[ri_first] < block_end) ri_first++;
        ri_last = ri_first;
        while (ri_last < L->constr_len && L->b_runs[ri_last].topEnd - L->constr[ri_last] <= block_start) {
            unsigned run_len = L->constr[ri_last];
            // Check that the run length will fit the block
            if (block_len_min <= run_len && run_len <= block_len_max) {
                if (run_fit_count == 0) {
                    // Make the topmost possible run start no later than this run
                    if (L->b_runs[ri_last].botStart > block_start) {
                        L->b_runs[ri_last].botStart = block_start;
                        B->dirty = true;
                    }
                    run_len_min = run_len;
                    run_len_max = run_len;
                }
                else {
                    run_len_min = std::min(run_len, run_len_min);
                    run_len_max = std::max(run_len, run_len_max);
                }
                run_fit_count++;
                run_fit_index = ri_last;
            }
            ri_last++;
        }

        if (run_fit_count == 0) {
            B->valid = false;
            return;
        }

        // Make the bottommost possible run start no earlier than this run
        if (L->b_runs[run_fit_index].topEnd < block_end){
            L->b_runs[run_fit_index].topEnd = block_end;
            B->dirty = true;
        }

        while (block_end < block_topStart + run_len_min) {
            // If the minimum run length puts the last cell in the shortest possible run further right
            // than the last cell in the block, fill up in between.
            nglinehyp_dev_cell_solve(L, B, NGCOLOR_BLACK, block_end);
            block_end++;
        }
        while (block_start > block_botEnd - run_len_min) {
            // If the minimum run length puts the first cell in the shortest possible run further left
            // than the first cell in the block, fill up in between.
            block_start--;
            nglinehyp_dev_cell_solve(L, B, NGCOLOR_BLACK, block_start);
        }
        if (block_len_min == run_len_max) {
            // If the block is already the maximum run length, then fill up white around it.
            if (block_end != L->len) nglinehyp_dev_cell_solve(L, B, NGCOLOR_WHITE, block_end);
            if (block_start != 0) nglinehyp_dev_cell_solve(L, B, NGCOLOR_WHITE, block_start - 1);
        }
    }

    if (solved) L->solved = true;

}

__device__
unsigned nghyp_heuristic_rc2i(Heuristic *X, unsigned r, unsigned c) {
    return r * X->w + c;
}

__device__
void nghyp_heuristic_set(Heuristic *X, unsigned r, unsigned c, char val) {
    X->data[nghyp_heuristic_rc2i(X, r, c)] = val;
}

__device__
char nghyp_heuristic_get(Heuristic *X, unsigned r, unsigned c) {
    return X->data[nghyp_heuristic_rc2i(X, r, c)];
}

bool nghyp_heuristic_init(NonogramLineDevice *Ls, Board2DDevice *B, Heuristic **pX) {

    Heuristic *X = (Heuristic *)malloc(sizeof(Heuristic));

    if (X == NULL) {
        fprintf(stderr, "Failed to allocate heuristic struct\n");
        return false;
    }

    X->w = B->w;
    X->h = B->h;
    X->data = (char *)calloc(B->w * B->h, sizeof(char));

    if (X->data == NULL) {
        fprintf(stderr, "Failed to allocate heuristic data array\n");
        free(X);
        return false;
    }

    *pX = X;

    return true;

}

void nghyp_heuristic_free(Heuristic *X) {

    free(X->data);
    free(X);

}

__device__
void nghyp_heuristic_cell(const NonogramLineDevice *Ls, const Board2DDevice *B, Heuristic *X, unsigned r, unsigned c) {

    if (board2d_dev_elem_get_rm(B, c, r) != NGCOLOR_UNKNOWN) {
        nghyp_heuristic_set(X, r, c, -1);
        return;
    }

    char score = 0;
    NonogramColor color_l = NGCOLOR_WHITE;
    NonogramColor color_r = NGCOLOR_WHITE;
    NonogramColor color_u = NGCOLOR_WHITE;
    NonogramColor color_d = NGCOLOR_WHITE;

    if (c > 0) color_l = board2d_dev_elem_get_rm(B, c - 1, r);
    if (c < X->w - 1) color_r = board2d_dev_elem_get_rm(B, c + 1, r);
    if (r > 0) color_u = board2d_dev_elem_get_rm(B, c, r - 1);
    if (r < X->h - 1) color_d = board2d_dev_elem_get_rm(B, c, r + 1);

    if (color_l == NGCOLOR_WHITE && color_r == NGCOLOR_UNKNOWN ||
            color_l == NGCOLOR_UNKNOWN && color_r == NGCOLOR_WHITE) {

        const NonogramLineDevice *L_r = &Ls[r];
        unsigned ri = 0;
        do {
            while (ri < L_r->constr_len && L_r->b_runs[ri].botStart + L_r->constr[ri] <= c) {
                ri++;
            }
            if (ri == L_r->constr_len) break;
            unsigned min_run_len = L_r->constr[ri];
            ri++;
            while (ri < L_r->constr_len && L_r->b_runs[ri].topEnd <= c + L_r->constr[ri]) {
                min_run_len = std::min(min_run_len, L_r->constr[ri]);
                ri++;
            }
            score += min_run_len;
        } while (false);

    }

    if (color_u == NGCOLOR_WHITE && color_d == NGCOLOR_UNKNOWN ||
        color_u == NGCOLOR_UNKNOWN && color_d == NGCOLOR_WHITE) {

        const NonogramLineDevice *L_c = &Ls[X->h + c];
        unsigned ri = 0;
        do {
            while (ri < L_c->constr_len && L_c->b_runs[ri].botStart + L_c->constr[ri] <= r) {
                ri++;
            }
            if (ri == L_c->constr_len) break;
            unsigned min_run_len = L_c->constr[ri];
            ri++;
            while (ri < L_c->constr_len && L_c->b_runs[ri].topEnd <= r + L_c->constr[ri]) {
                min_run_len = std::min(min_run_len, L_c->constr[ri]);
                ri++;
            }
            score += min_run_len;
        } while (false);

    }

    nghyp_heuristic_set(X, r, c, score);

}

void nghyp_heuristic_fill(const NonogramLineDevice *Ls, const Board2DDevice *B, Heuristic *X) {

    for (unsigned r = 0; r < X->h; r++) {
        for (unsigned c = 0; c < X->w; c++) {
            nghyp_heuristic_cell(Ls, B, X, r, c);
        }
    }

}

void nghyp_heuristic_update(HypotheticalBoard *H, Heuristic *X, NonogramColor color) {

#ifdef DEBUG
    if (board2d_dev_elem_get_rm(H->B, X->c_max, X->r_max) != NGCOLOR_UNKNOWN) return;
#endif
    board2d_dev_elem_set(H->B, X->c_max, X->r_max, color);
    H->row = X->r_max;
    H->col = X->c_max;
    H->guess_color = color;

}

void nghyp_heuristic_max(Heuristic *X) {

    char score_max = 0;
    unsigned r_max = 0, c_max = 0;
    for (unsigned r = 0; r < X->h; r++) {
        for (unsigned c = 0; c < X->w; c++) {
            if (score_max < nghyp_heuristic_get(X, r, c)) {
                score_max = nghyp_heuristic_get(X, r, c);
                r_max = r;
                c_max = c;
            }
        }
    }

    X->r_max = r_max;
    X->c_max = c_max;

    // Clear the heuristic.
    nghyp_heuristic_set(X, r_max, c_max, -1);

}

HypotheticalBoard nghyp_init(NonogramLineDevice *Ls, Board2DDevice *B) {

    HypotheticalBoard H;
    H.Ls = ng_linearr_deepcopy_host(Ls, B->w, B->h);
    H.B = board2d_deepcopy_host(B);
    ng_linearr_board_change(H.Ls, H.B);

    return H;

}

void nghyp_free(HypotheticalBoard H) {

    free(H.Ls);
    board2d_free_host(H.B);

}

void nghyp_hyp_confirm(HypotheticalBoard H, Board2DDevice *B, NonogramLineDevice **Ls) {

    std::swap(H.B->data, B->data);
    std::swap(H.Ls, *Ls);

}

/**
 * Checks if there's a contradiction, and modifies the original board.
 * @param H
 * @param B
 * @return
 */
bool nghyp_valid_check(HypotheticalBoard *H, Board2DDevice *B) {

    if (H->B->valid) return true;

#ifdef DEBUG
    if (board2d_dev_elem_get(B, H->col, H->row) != NGCOLOR_UNKNOWN) {
        fprintf(stderr, "Hypothesis is a conflict\n");
        return false;
    }
#endif

    if (H->guess_color == NGCOLOR_BLACK) board2d_dev_elem_set(B, H->col, H->row, NGCOLOR_WHITE);
    else if (H->guess_color == NGCOLOR_WHITE) board2d_dev_elem_set(B, H->col, H->row, NGCOLOR_BLACK);
    B->dirty = true;
    return false;

}

void nghyp_common_set(HypotheticalBoard *H1, HypotheticalBoard *H2, Board2DDevice *B) {

    for (unsigned r = 0; r < B->h; r++) {
        for (unsigned c = 0; c < B->w; c++) {
            if (board2d_dev_elem_get_rm(B, c, r) == NGCOLOR_UNKNOWN) {
                NonogramColor color = board2d_dev_elem_get_rm(H1->B, c, r);
                if (color != NGCOLOR_UNKNOWN &&
                    color == board2d_dev_elem_get_rm(H1->B, c, r)) {
                    board2d_dev_elem_set(B, c, r, color);
                    B->dirty = true;
                }
            }
        }
    }
}