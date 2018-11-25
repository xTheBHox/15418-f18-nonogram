//
// Created by Benjamin Huang on 11/19/2018.
//

#include "NonogramLineDevice.h"

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
            ngline_dev_cell_solve(L, B, NonogramColor::WHITE, prev_wrun_botStart);
            prev_wrun_botStart++;
        }

        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngline_dev_cell_solve(L, B, NonogramColor::BLACK, i);
        }

        prev_wrun_botStart = r.botStart + L->constr[ri];

    }

    while (prev_wrun_botStart < L->len) {
        ngline_dev_cell_solve(L, B, NonogramColor::WHITE, prev_wrun_botStart);
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
        if (color == NonogramColor::BLACK) {
            curr_bblock_len++;
        }
        else if (color == NonogramColor::WHITE) {
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
        if (color == NonogramColor::BLACK) {
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
void ngline_dev_block_max_size_fill(NonogramLineDevice *L, Board2DDevice *B,
                         unsigned i, unsigned curr_bblock_len) {
    if (i + 1 < L->len) {
        ngline_dev_cell_solve(L, B, NonogramColor::WHITE, i + 1);
    }
    if (i >= curr_bblock_len) {
        ngline_dev_cell_solve(L, B, NonogramColor::WHITE, i - curr_bblock_len);
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
    ngline_dev_update(L, B);
    ngline_dev_runs_fill(L, B);

}

#ifdef __NVCC__
__global__
void ngline_col_solve_kernel(Board2DDevice *B, NonogramLineDevice *Ls) {
    unsigned i = blockIdx.x;
#else
void ngline_col_solve_kernel(Board2DDevice *B, NonogramLineDevice *Ls, unsigned i) {
#endif

    NonogramLineDevice *L = &Ls[B->h + i];
    ngline_dev_update(L, B);
    ngline_dev_runs_fill(L, B);

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
        std::cout << "Iteration " << perf_iter_cnt << std::endl;
#endif
    } while (B_host->dirty);

#ifdef PERF
    std::cout << "Total iterations: " << perf_iter_cnt << std::endl;
#endif

    board2d_cleanup_dev(B_host, B_dev);
    ng_linearr_free_dev(Ls_dev);

}
