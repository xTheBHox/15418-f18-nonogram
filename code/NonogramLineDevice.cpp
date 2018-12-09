//
// Created by Benjamin Huang on 11/19/2018.
//
#include "NonogramLineDevice.h"
#include "Board2DDevice.h"

#define DEBUG
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

__device__
void ngline_dev_runs_fill(NonogramLineDevice *L, Board2DDevice *B) {

    unsigned prev_wrun_botStart = 0;

    for (unsigned ri = 0; ri < L->constr_len; ri++) {
        BRun r = L->b_runs[ri];

        while (r.topEnd - L->constr[ri] > prev_wrun_botStart) {
            ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, prev_wrun_botStart);
            prev_wrun_botStart++;
        }

        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngline_dev_cell_solve(L, B, NGCOLOR_BLACK, i);
        }

        prev_wrun_botStart = r.botStart + L->constr[ri];

    }

    while (prev_wrun_botStart < L->len) {
        ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, prev_wrun_botStart);
        prev_wrun_botStart++;
    }

}

__device__
void ngline_dev_update(NonogramLineDevice *L, Board2DDevice *B) {

    if (L->constr_len == 0) return;

    // Walk down the line, and fill in the run structures
    unsigned ri0 = 0; // The minimum index black run we could be in
    unsigned ri1 = 0; // The maximum index black run we could be in

    unsigned curr_bblock_len = 0;
    // unsigned first_nwi = 0;

    // Walk
    unsigned i = L->b_runs[0].topEnd - L->constr[0];
    for (; i < L->len; i++) {

        char color = L->data[i];
        if (color == NGCOLOR_BLACK) {
            curr_bblock_len++;
        }
        else if (color == NGCOLOR_WHITE) {
            curr_bblock_len = 0;
            // first_nwi = i + 1;
        }
        else {
            curr_bblock_len = 0;
            continue;
        }

        while (i >= L->b_runs[ri0].botStart + L->constr[ri0]) ri0++;

        unsigned max_run_len = 0;
        for (unsigned j = ri0; j <= ri1; j++) {
            if (max_run_len < L->constr[j]) max_run_len = L->constr[j];
        }
        while (ri1 + 1 < L->constr_len && i >= L->b_runs[ri1 + 1].topEnd - L->constr[ri1 + 1]) {
            ri1++;
            if (max_run_len < L->constr[ri1]) max_run_len = L->constr[ri1];
        }

        if (ri0 == L->constr_len) {
            // We have finished all the shaded regions
            break;
        }

        // Check if we are in an already confirmed shaded region
        if (L->b_runs[ri0].botStart <= i && i < L->b_runs[ri0].topEnd) {
            continue;
        }

        // Check if we are in an already confirmed unshaded region
        if (ri0 > ri1) continue;

        // If we get here, we have a determined cell that has not been assigned to a run.

        // Try to assign to a run.
        if (color == NGCOLOR_BLACK) {
            if (ri0 == ri1) { // Can fix

                ngline_dev_botStart_propagate(L, B, ri0, i);
                ngline_dev_topEnd_propagate(L, B, ri0, i + 1);

            }
        }
        else { // if (color == Nonogram::Color::WHITE) {
            if (i >= L->b_runs[ri0].botStart) {
                ngline_dev_botStart_propagate(L, B, ri0, i - L->constr[ri0]);
            }
            if (i < L->b_runs[ri1].topEnd) {
                ngline_dev_topEnd_propagate(L, B, ri1, i + L->constr[ri1] + 1);
            }
        }

        // We are looking for shaded blocks that have not been assigned to a run.
        if (curr_bblock_len == max_run_len) {
            ngline_dev_block_max_size_fill(L, B, i, curr_bblock_len);
        }

    }
}

__device__ __inline__
bool ngline_dev_run_top_adjust(NonogramLineDevice *L, unsigned &topEnd, unsigned line_len, unsigned run_len) {

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
bool ngline_dev_run_bot_adjust(NonogramLineDevice *L, unsigned &botStart, unsigned line_len, unsigned run_len) {

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
        L->b_runs[ri].topEnd = std::max(L->b_runs[ri].topEnd, L->b_runs[ri-1].topEnd + L->constr[ri] + 1);
    }

}

__device__ __inline__
void ngline_dev_run_bot_prop(NonogramLineDevice *L) {

    for (unsigned ri = L->constr_len - 2; ri < L->constr_len; ri--) {
        L->b_runs[ri].botStart = std::min(L->b_runs[ri].botStart, L->b_runs[ri+1].botStart - L->constr[ri] - 1);
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
void ngline_dev_run_solve(NonogramLineDevice *L, Board2DDevice *B, unsigned run_index) {

#ifdef DEBUG
    if (run_index >= L->constr_len) return;
#endif

    BRun *R = &L->b_runs[run_index];
    unsigned run_len = L->constr[run_index];

    // Adjust the possible start and end points of the runs

    while(ngline_dev_run_top_adjust(L, R->topEnd, run_len, L->constr[run_index]));
    while(ngline_dev_run_bot_adjust(L, R->botStart, run_len, L->constr[run_index]));

    // Propagate changes - one thread only!

    ngline_dev_run_top_prop(L);
    ngline_dev_run_bot_prop(L);

    // Fill overlaps

    ngline_dev_run_fill_black(L, B, R, run_len);
    ngline_dev_run_fill_white(L, B, run_index);
    if (run_index == 0) ngline_dev_run_fill_white(L, B, L->constr_len);

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

    while (i < L->len) {

        if (L->data[i] == NGCOLOR_UNKNOWN) {
            i++;
            continue;
        }
        if (L->data[i] == NGCOLOR_WHITE) {
            i++;
            block_topStart = i;
            continue;
        }
        // Must be black
        block_start = i;
        while (L->data[i] == NGCOLOR_BLACK) {
            i++;
            if (i == L->len) break;
        }
        block_end = i;
        block_botEnd = i;
        while (block_botEnd < L->len && L->data[block_botEnd] != NGCOLOR_WHITE) block_botEnd++;
        // Determine the minimum and maximum length of this block
        unsigned block_len_min = block_end - block_start;
        unsigned block_len_max = block_botEnd - block_topStart;

        unsigned run_fit_count = 0;
        // next three are not valid if run_fit_count == 0
        unsigned run_fit_index = 0;
        unsigned run_len_min = L->len;
        unsigned run_len_max = 0;

        // Get the run indices
        while (L->b_runs[ri_first].botStart + L->constr[ri_first] < block_end) ri_first++;
        ri_last = ri_first;
        while (ri_last < L->constr_len && L->b_runs[ri_last].topEnd - L->constr[ri_last] <= block_start) {
            unsigned run_len = L->constr[ri_last];
            if (block_len_min < run_len < block_len_max) {
                if (run_fit_count == 0) {
                    if (L->b_runs[ri_last].botStart > block_start) {
                        L->b_runs[ri_last].botStart = block_start;
                        B->dirty = true;
                    }
                }
                run_fit_count++;
                run_fit_index = ri_last;
                run_len_min = std::min(run_len, run_len_min);
                run_len_max = std::max(run_len, run_len_max);
            }
            ri_last++;
        }

        // TODO If checking for contradiction, must check no possible runs
        if (L->b_runs[run_fit_index].topEnd < block_end){
            L->b_runs[run_fit_index].topEnd = block_end;
            B->dirty = true;
        }

        while (block_end < block_topStart + run_len_min) {
            ngline_dev_cell_solve(L, B, NGCOLOR_BLACK, block_end);
            block_end++;
        }
        while (block_start > block_botEnd - run_len_min) {
            block_start--;
            ngline_dev_cell_solve(L, B, NGCOLOR_BLACK, block_start);
        }
        if (block_len_min == run_len_max) {
            if (block_end != L->len) ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, block_end);
            if (block_start != 0) ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, block_start - 1);
        }

    }

}

__device__
void ngline_dev_block_max_size_fill(NonogramLineDevice *L, Board2DDevice *B,
                         unsigned i, unsigned curr_bblock_len) {
    if (i + 1 < L->len) {
        ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, i + 1);
    }
    if (i >= curr_bblock_len) {
        ngline_dev_cell_solve(L, B, NGCOLOR_WHITE, i - curr_bblock_len);
    }

}

__device__
void ngline_dev_botStart_propagate(NonogramLineDevice *L, Board2DDevice *B,
                                      unsigned ri, unsigned i) {
    while (i < L->b_runs[ri].botStart) {
        L->b_runs[ri].botStart = i;
        if (ri == 0) break;
        ri--;
        i -= L->constr[ri] + 1;
    }
}

__device__
void ngline_dev_topEnd_propagate(NonogramLineDevice *L, Board2DDevice *B,
                                    unsigned ri, unsigned i) {
    while (i > L->b_runs[ri].topEnd) {
        L->b_runs[ri].topEnd = i;
        ri++;
        if (ri == L->constr_len) break;
        i += L->constr[ri] + 1;
    }
}

__device__
void ngline_dev_update2(NonogramLineDevice *L, Board2DDevice *B) {

    NonogramColor currColor = NGCOLOR_WHITE;

    unsigned currBlockLen;

    unsigned i = L->b_runs[0].topEnd - L->constr[0];

    unsigned prev_wi = i;
    unsigned prev_bi = i;


    if (L->constr_len == 0) return;

    // Walk down the line, and fill in the run structures
    unsigned ri0 = 0; // The minimum index black run we could be in
    unsigned ri1 = 0; // The maximum index black run we could be in

    unsigned curr_bblock_len = 0;
    // unsigned first_nwi = 0;

    // Walk
    for (; i < L->len; i++) {

        char color = L->data[i];
        if (color == NGCOLOR_BLACK) {
            curr_bblock_len++;
        }
        else if (color == NGCOLOR_WHITE) {
            curr_bblock_len = 0;
            // first_nwi = i + 1;
        }
        else {
            curr_bblock_len = 0;
            continue;
        }

        while (i >= L->b_runs[ri0].botStart + L->constr[ri0]) ri0++;

        unsigned max_run_len = 0;
        for (unsigned j = ri0; j <= ri1; j++) {
            if (max_run_len < L->constr[j]) max_run_len = L->constr[j];
        }
        while (ri1 + 1 < L->constr_len && i >= L->b_runs[ri1 + 1].topEnd - L->constr[ri1 + 1]) {
            ri1++;
            if (max_run_len < L->constr[ri1]) max_run_len = L->constr[ri1];
        }

        if (ri0 == L->constr_len) {
            // We have finished all the shaded regions
            break;
        }

        // Check if we are in an already confirmed shaded region
        if (L->b_runs[ri0].botStart <= i && i < L->b_runs[ri0].topEnd) {
            continue;
        }

        // Check if we are in an already confirmed unshaded region
        if (ri0 > ri1) continue;

        // If we get here, we have a determined cell that has not been assigned to a run.

        // Try to assign to a run.
        if (color == NGCOLOR_BLACK) {
            if (ri0 == ri1) { // Can fix

                ngline_dev_botStart_propagate(L, B, ri0, i);
                ngline_dev_topEnd_propagate(L, B, ri0, i + 1);

            }
        }
        else { // if (color == Nonogram::Color::WHITE) {
            if (i >= L->b_runs[ri0].botStart) {
                ngline_dev_botStart_propagate(L, B, ri0, i - L->constr[ri0]);
            }
            if (i < L->b_runs[ri1].topEnd) {
                ngline_dev_topEnd_propagate(L, B, ri1, i + L->constr[ri1] + 1);
            }
        }

        // We are looking for shaded blocks that have not been assigned to a run.
        if (curr_bblock_len == max_run_len) {
            ngline_dev_block_max_size_fill(L, B, i, curr_bblock_len);
        }

    }


}

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
    }
}

#ifdef __NVCC__
__global__
void ngline_row_solve_kernel(Board2DDevice *B, NonogramLineDevice *Ls) {
    unsigned i = blockIdx.x;
#else
void ngline_row_solve_kernel(Board2DDevice *B, NonogramLineDevice *Ls, unsigned i) {
#endif

    NonogramLineDevice *L = &Ls[i];
    for (unsigned ri = 0; ri < L->constr_len; ri++) {
        ngline_dev_run_solve(L, B, ri);
    }
    ngline_dev_block_solve(L, B);

}

#ifdef __NVCC__
__global__
void ngline_col_solve_kernel(Board2DDevice *B, NonogramLineDevice *Ls) {
    unsigned i = blockIdx.x;
#else
void ngline_col_solve_kernel(Board2DDevice *B, NonogramLineDevice *Ls, unsigned i) {
#endif

    NonogramLineDevice *L = &Ls[B->h + i];
    for (unsigned ri = 0; ri < L->constr_len; ri++) {
        ngline_dev_run_solve(L, B, ri);
    }
    ngline_dev_block_solve(L, B);

}

#ifdef __NVCC__
__global__
void ngline_init_kernel(Board2DDevice *B, NonogramLineDevice *Ls) {
    unsigned i = blockIdx.x;
#else
void ngline_init_kernel(Board2DDevice *B, NonogramLineDevice *Ls, unsigned i) {
#endif

    NonogramLineDevice *L = &Ls[i];

    if (L->line_is_row) {
        L->data = board2d_dev_row_ptr_get(B, L->line_index);
#ifndef __NVCC__
#ifdef DEBUG
        printf("Init row %d len=%d constr_len=%d\t", L->line_index, L->len, L->constr_len);
#endif
#endif
    }
    else {
        L->data = board2d_dev_col_ptr_get(B, L->line_index);
#ifndef __NVCC__
#ifdef DEBUG
        printf("Init col %d len=%d constr_len=%d\t", L->line_index, L->len, L->constr_len);
#endif
#endif
    }

    ngline_init_dev(L);

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
    }

    for (unsigned i = 0; i < w; i++) {
        Ls_tmp[i + h].constr_len = 0;
        Ls_tmp[i + h].line_index = i;
        Ls_tmp[i + h].line_is_row = false;
        Ls_tmp[i + h].len = h;
    }

    *Ls = Ls_tmp;
    return true;

}

NonogramLineDevice *ng_linearr_init_dev(unsigned w, unsigned h, NonogramLineDevice *Ls_host) {

#ifdef DEBUG
    std::cout << __func__ << " called" << std::endl;
#endif
#ifdef __NVCC__
    void *Ls_dev;
    size_t Ls_size = sizeof(NonogramLineDevice) * (w + h);

    cudaCheckError(cudaMalloc(&Ls_dev, Ls_size));
    cudaCheckError(cudaMemcpy(Ls_dev, (void *)Ls_host, Ls_size, cudaMemcpyHostToDevice));

    return (NonogramLineDevice *)Ls_dev;
#else
    return Ls_host;
#endif

}

void ng_linearr_free_dev(NonogramLineDevice *Ls_dev) {

#ifdef __NVCC__
    cudaCheckError(cudaFree(Ls_dev));
#else
    return;
#endif

}

bool ng_init(unsigned w, unsigned h, NonogramLineDevice **Ls, Board2DDevice **B) {

    if (!ng_linearr_init_host(w, h, Ls)) return false;
    *B = board2d_init_host(w, h);
    if (*B == NULL) return false;
    return true;

}

void ng_free(NonogramLineDevice *Ls, Board2DDevice *B) {

    free(Ls);
    board2d_free_host(B);

}

bool ng_constr_add(NonogramLineDevice *Ls, unsigned line_index, unsigned constr) {

    NonogramLineDevice *L = &Ls[line_index];

    if (L->constr_len >= MAX_RUNS) {
        return false;
    }

    L->constr[L->constr_len] = constr;
    L->constr_len++;

    return true;

}

void ng_solve(NonogramLineDevice *Ls_host, Board2DDevice *B_host) {
#ifdef DEBUG
    std::cout << "ng_solve called" << std::endl;
#endif
#ifdef PERF
    unsigned perf_iter_cnt = 0;
#endif

    NonogramLineDevice *Ls_dev;
    Board2DDevice *B_dev;

    // Move structures to device memory

#ifdef DEBUG
    std::cout << "Line array initializing..." << std::endl;
#endif
    Ls_dev = ng_linearr_init_dev(B_host->w, B_host->h, Ls_host);

#ifdef DEBUG
    std::cout << "Board initializing..." << std::endl;
#endif
    B_dev = board2d_init_dev(B_host);

    // Initialize the runs

    unsigned block_cnt = B_host->w + B_host->h;

#ifdef DEBUG
    std::cout << "Lines initializing..." << std::endl;
#endif

    TIMER_START(solve_loop);

#ifdef __NVCC__
    ngline_init_kernel<<<block_cnt, 1>>>(B_dev, Ls_dev);
#else
    for (unsigned i = 0; i < block_cnt; i++) {
        ngline_init_kernel(B_dev, Ls_dev, i);
    }
#endif

#ifdef DEBUG
    std::cout << "Lines alternating..." << std::endl;
#endif

    do {

        B_host->dirty = false;
#ifdef __NVCC__
        cudaMemcpy(&B_dev->dirty, &B_host->dirty, sizeof(bool), cudaMemcpyHostToDevice);
        ngline_row_solve_kernel<<<B_host->h, 1>>> (B_dev, Ls_dev);
        ngline_col_solve_kernel<<<B_host->w, 1>>> (B_dev, Ls_dev);
        cudaMemcpy(&B_host->dirty, &B_dev->dirty, sizeof(bool), cudaMemcpyDeviceToHost);
#else
        for (unsigned i = 0; i < B_host->h; i++) {
            ngline_row_solve_kernel(B_dev, Ls_dev, i);
        }
        for (unsigned i = 0; i < B_host->w; i++) {
            ngline_col_solve_kernel(B_dev, Ls_dev, i);
        }

#endif

#ifdef PERF
        perf_iter_cnt++;
#endif
    } while (B_host->dirty);

    TIMER_STOP(solve_loop);

#ifdef PERF
    std::cout << "Total iterations: " << perf_iter_cnt << std::endl;
#endif

    board2d_cleanup_dev(B_host, B_dev);
    ng_linearr_free_dev(Ls_dev);

}
