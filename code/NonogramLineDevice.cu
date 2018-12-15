//
// Created by Benjamin Huang on 11/19/2018.
//
#include "NonogramLineDevice.h"

// #define DEBUG

__device__
void ngline_dev_cell_solve(NonogramLineDevice *L, Board2DDevice *B,
                           NonogramColor color, unsigned i) {
    if (L->data[i] != color) {
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

#ifdef DISP
        mvaddch(y, x, ngramColorToChar(color));
        refresh();
#endif
    }
}

__device__
void ngline_init_dev(NonogramLineDevice *L) {

    if (L->constr_len == 0) {
        return;
    }

    // Find the topmost line configuration for each run.

    unsigned topSum = 0;
    for (unsigned i = 0; i < L->constr_len; i++) {
        topSum += L->constr[i];
        L->b_runs[i].topEnd = topSum;
        topSum++;
    }

#ifndef __NVCC__
#ifdef DEBUG
    printf("topSum=%d\t", topSum);
#endif
#endif
    // Find the bottommost line configuration for each run.

    unsigned botSum = L->len;
    for (unsigned i = L->constr_len - 1; i < L->constr_len; i--) {
        botSum -= L->constr[i];
        L->b_runs[i].botStart = botSum;
        botSum--;
    }

#ifndef __NVCC__
#ifdef DEBUG
    printf("botSum=%d\n", botSum);
#endif
#endif
}

__device__ __inline__
bool ngline_dev_run_top_adjust(NonogramLineDevice *L, unsigned char &topEnd, unsigned line_len, unsigned run_len) {

    if (topEnd < line_len && L->data[topEnd] == NGCOLOR_BLACK) {
        topEnd++;
        return true;
    }

    for (unsigned i = topEnd; i > topEnd - run_len; i--) {
        if (L->data[i - 1] == NGCOLOR_WHITE) {
            topEnd = i + run_len;
            return true;
        }
    }

    if (topEnd > run_len && L->data[topEnd - run_len - 1] == NGCOLOR_BLACK) {
        topEnd++;
        return true;
    }

    return false;

}

__device__ __inline__
bool ngline_dev_run_bot_adjust(NonogramLineDevice *L, unsigned char &botStart, unsigned line_len, unsigned run_len) {

    if (botStart > 0 && L->data[botStart - 1] == NGCOLOR_BLACK) {
        botStart--;
        return true;
    }

    for (unsigned i = botStart; i < botStart + run_len; i++) {
        if (L->data[i] == NGCOLOR_WHITE) {
            botStart = i - run_len;
            return true;
        }
    }

    if (botStart + run_len < line_len && L->data[botStart + run_len] == NGCOLOR_BLACK) {
        botStart--;
        return true;
    }

    return false;

}

__device__ __inline__
void ngline_dev_run_top_prop(NonogramLineDevice *L) {

    for (unsigned ri = 1; ri < L->constr_len; ri++) {
        L->b_runs[ri].topEnd = dev_max(L->b_runs[ri].topEnd, L->b_runs[ri-1].topEnd + L->constr[ri] + 1);
    }

}

__device__ __inline__
void ngline_dev_run_bot_prop(NonogramLineDevice *L) {

    for (unsigned ri = L->constr_len - 2; ri < L->constr_len; ri--) {
        L->b_runs[ri].botStart = dev_min(L->b_runs[ri].botStart, L->b_runs[ri+1].botStart - L->constr[ri] - 1);
    }

}

__device__ __inline__
void ngline_dev_run_fill_black(NonogramLineDevice *L, Board2DDevice *B, const BRun *R, unsigned run_len) {

    for (unsigned i = R->botStart; i < R->topEnd; i++) {
        ngline_dev_cell_solve(L, B, NGCOLOR_BLACK, i);
    }
    if (R->topEnd == R->botStart + run_len) {
        if (R->topEnd < L->len) ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, R->topEnd);
        if (R->botStart > 0) ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, R->botStart - 1);
    }

}

__device__ __inline__
void ngline_dev_run_fill_white(NonogramLineDevice *L, Board2DDevice *B, unsigned ri) {
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
        ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, i);
    }

}

__device__
void ngline_dev_run_solve(NonogramLineDevice *L, Board2DDevice *B) {

#ifdef DEBUG
    if (run_index >= L->constr_len) return;
#endif

    unsigned line_len = L->len;

    // Adjust the possible start and end points of the runs

    for (unsigned ri = 0; ri < L->constr_len; ri++) {
        while(ngline_dev_run_top_adjust(L, L->b_runs[ri].topEnd, line_len, L->constr[ri]));
        while(ngline_dev_run_bot_adjust(L, L->b_runs[ri].botStart, line_len, L->constr[ri]));
    }
    // Propagate changes - one thread only!

    ngline_dev_run_top_prop(L);
    ngline_dev_run_bot_prop(L);

    // Fill overlaps
    for (unsigned ri = 0; ri < L->constr_len; ri++) {
        ngline_dev_run_fill_white(L, B, ri);
        ngline_dev_run_fill_black(L, B, &L->b_runs[ri], L->constr[ri]);
    }
    ngline_dev_run_fill_white(L, B, L->constr_len);

}

__device__
void ngline_dev_block_solve(NonogramLineDevice *L, Board2DDevice *B) {

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
        while (L->b_runs[ri_first].botStart + L->constr[ri_first] < block_end) ri_first++;
        ri_last = ri_first;
        while (ri_last < L->constr_len && L->b_runs[ri_last].topEnd <= block_start + L->constr[ri_last]) {
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
                    run_len_min = dev_min(run_len, run_len_min);
                    run_len_max = dev_max(run_len, run_len_max);
                }
                run_fit_count++;
                run_fit_index = ri_last;
            }
            ri_last++;
        }

        // TODO If checking for contradiction, must check no possible runs
        // Make the bottommost possible run start no earlier than this run
        if (L->b_runs[run_fit_index].topEnd < block_end){
            L->b_runs[run_fit_index].topEnd = block_end;
            B->dirty = true;
        }

        while (block_end < block_topStart + run_len_min) {
            // If the minimum run length puts the last cell in the shortest possible run further right
            // than the last cell in the block, fill up in between.
            ngline_dev_cell_solve(L, B, NGCOLOR_BLACK, block_end);
            block_end++;
        }
        while (block_start > block_botEnd - run_len_min) {
            // If the minimum run length puts the first cell in the shortest possible run further left
            // than the first cell in the block, fill up in between.
            block_start--;
            ngline_dev_cell_solve(L, B, NGCOLOR_BLACK, block_start);
        }
        if (block_len_min == run_len_max) {
            // If the block is already the maximum run length, then fill up white around it.
            if (block_end != L->len) ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, block_end);
            if (block_start != 0) ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, block_start - 1);
        }

    }

    if (solved) L->solved = true;

}

__device__
void ngline_dev_mutableonly_copy(NonogramLineDevice *L_dst, const NonogramLineDevice *L_src) {

    L_dst->solved = L_src->solved;
    for (unsigned i = 0; i < L_src->constr_len; i++) {
        L_dst->b_runs[i] = L_src->b_runs[i];
    }

}

bool ng_linearr_init_host(unsigned w, unsigned h, NonogramLineDevice **Ls) {

    // Allocate the array of line solver structs
    unsigned Ls_len = w + h;
    size_t Ls_size = sizeof(NonogramLineDevice) * Ls_len;
    NonogramLineDevice *Ls_tmp = (NonogramLineDevice *)malloc(Ls_size);

    if (Ls_tmp == NULL) {
        fprintf(stderr, "Failed to allocate host line array\n");
        return false;
    }

    for (unsigned i = 0; i < h; i++) {
        Ls_tmp[i].constr_len = 0;
        Ls_tmp[i].line_index = i;
        Ls_tmp[i].line_is_row = true;
        Ls_tmp[i].len = w;
        Ls_tmp[i].solved = false;
    }

    for (unsigned i = 0; i < w; i++) {
        Ls_tmp[i + h].constr_len = 0;
        Ls_tmp[i + h].line_index = i;
        Ls_tmp[i + h].line_is_row = false;
        Ls_tmp[i + h].len = h;
        Ls_tmp[i + h].solved = false;
    }

    *Ls = Ls_tmp;
    return true;

}

#ifdef __NVCC__
NonogramLineDevice *ng_linearr_init_dev(unsigned w, unsigned h, NonogramLineDevice *Ls_host) {

    void *Ls_dev;
    size_t Ls_size = sizeof(NonogramLineDevice) * (w + h);

    cudaCheckError(cudaMalloc(&Ls_dev, Ls_size));
    cudaCheckError(cudaMemcpy(Ls_dev, (void *)Ls_host, Ls_size, cudaMemcpyHostToDevice));

    return (NonogramLineDevice *)Ls_dev;

}
#endif

#ifdef __NVCC__
void ng_linearr_free_dev(NonogramLineDevice *Ls_dev) {

    cudaCheckError(cudaFree(Ls_dev));

}
#endif

NonogramLineDevice *ng_linearr_deepcopy_host(NonogramLineDevice *Ls, unsigned w, unsigned h) {

    unsigned Ls_len = w + h;
    size_t Ls_size = sizeof(NonogramLineDevice) * Ls_len;
    NonogramLineDevice *Ls_copy = (NonogramLineDevice *)malloc(Ls_size);

    if (Ls_copy == NULL) {
        fprintf(stderr, "Failed to allocate copy of line array\n");
        return NULL;
    }

    memcpy((void *)Ls_copy, (void *)Ls, Ls_size);
    return Ls_copy;

}

#ifdef __NVCC__
NonogramLineDevice *ng_linearr_deepcopy_dev_double(NonogramLineDevice *Ls, unsigned Ls_size) {

    NonogramLineDevice *Ls_dcopy;
    cudaCheckError(cudaMalloc((void **)&Ls_dcopy, 2 * Ls_size));
    cudaCheckError(cudaMemcpy((void *)Ls_dcopy, (void *)Ls, Ls_size, cudaMemcpyDeviceToDevice));
    char *Ls_copy2 = ((char *) Ls_dcopy) + Ls_size;
    cudaCheckError(cudaMemcpy((void *)Ls_copy2, (void *)Ls, Ls_size, cudaMemcpyDeviceToDevice));

    return Ls_dcopy;

}
#endif

void ng_linearr_board_change(NonogramLineDevice *Ls, Board2DDevice *B) {

    for (unsigned i = 0; i < B->h; i++) {
        Ls[i].data = board2d_dev_row_ptr_get(B, i);
    }

    for (unsigned i = 0; i < B->w; i++) {
        Ls[i + B->h].data = board2d_dev_col_ptr_get(B, i);
    }

}
