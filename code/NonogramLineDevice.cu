//
// Created by Benjamin Huang on 11/19/2018.
//

#include "NonogramLineDevice.h"

__host__
void *ngline_dev_init_host(const unsigned *constr, unsigned constr_len) {

    if (constr_len > MAX_RUNS) return nullptr;

    void *line_dev_v;
    cudaMalloc(&line_dev_v, sizeof(NonogramLineDevice));
    
    NonogramLineDevice *line_dev = (NonogramLineDevice *)line_dev_v;

    void *line_dev_constr = (void *) line_dev->constr;
    cudaMemcpy(line_dev_constr, constr, sizeof(unsigned) * constr_len, cudaMemcpyHostToDevice);

    void *line_dev_constr_len = (void *) &line_dev->constr_len;
    cudaMemcpy(line_dev_constr_len, &constr_len, sizeof(unsigned), cudaMemcpyHostToDevice);

    return line_dev_v;

}

__device__
void ngline_dev_init_dev(NonogramLineDevice *L,
                          unsigned _len, const char *_data,
                          unsigned _index, bool _is_row) {

    L->len = _len;
    L->data = _data;
    L->line_is_row = _is_row;
    L->line_index = _index;

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

    // Find the bottommost line configuration for each run.

    unsigned botSum = L->len;
    for (unsigned i = L->constr_len - 1; i < L->constr_len; i--) {
        botSum -= L->constr[i];
        L->b_runs[i].botStart = botSum;
        botSum--;
    }

}

__device__
void ngline_dev_runs_fill(NonogramLineDevice *L, Board2DDevice *B) {

    unsigned prev_wrun_botStart = 0;

    for (unsigned ri = 0; ri < L->constr_len; ri++) {
        BRun r = L->b_runs[ri];

        while (r.topEnd - L->constr[ri] > prev_wrun_botStart) {
            ngline_dev_cell_solve(L, B, Nonogram::Color::WHITE, prev_wrun_botStart);
            prev_wrun_botStart++;
        }

        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngline_dev_cell_solve(L, B, Nonogram::Color::BLACK, i);
        }

        prev_wrun_botStart = r.botStart + L->constr[ri];

    }

    while (prev_wrun_botStart < L->len) {
        ngline_dev_cell_solve(L, B, Nonogram::Color::WHITE, prev_wrun_botStart);
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
        if (color == Nonogram::Color::BLACK) {
            curr_bblock_len++;
        }
        else if (color == Nonogram::Color::WHITE) {
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
        if (color == Nonogram::Color::BLACK) {
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
        ngline_dev_cell_solve(L, B, Nonogram::Color::WHITE, i + 1);
    }
    if (i >= curr_bblock_len) {
        ngline_dev_cell_solve(L, B, Nonogram::Color::WHITE, i - curr_bblock_len);
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
                           char color, unsigned i) {
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

__device__
NonogramLineDevice *ng_dev_get_ngline(Board2DDevice *B, NonogramLineDevice **Ls) {

    unsigned i = blockIdx.x;
    if (i >= B->h) {
        return Ls[i - B->h];
    }
    else {
        return Ls[i];
    }

}

__global__
void ngline_row_solve_kernel(Board2DDevice *B, NonogramLineDevice **Ls) {

    NonogramLineDevice *L = Ls[blockIdx.x];
    ngline_dev_update(L, B);
    ngline_dev_runs_fill(L, B);

}

__global__
void ngline_col_solve_kernel(Board2DDevice *B, NonogramLineDevice **Ls) {

    NonogramLineDevice *L = Ls[B->h + blockIdx.x];
    ngline_dev_update(L, B);
    ngline_dev_runs_fill(L, B);

}

__global__
void ngline_init_kernel(Board2DDevice *B, NonogramLineDevice **Ls) {
    unsigned i = blockIdx.x;

    unsigned line_index, line_len;
    bool line_is_row;
    const char *line_data;
    if (i >= B->h) {
        line_len = B->w;
        line_index = i - B->h;
        line_is_row = false;
        line_data = board2d_dev_row_ptr_get(B, line_index);
    }
    else {
        line_len = B->h;
        line_index = i;
        line_is_row = true;
        line_data = board2d_dev_col_ptr_get(B, line_index);
    }

    ngline_dev_init_dev(Ls[line_index], line_len, line_data, line_index, line_is_row);

}

__host__
void ng_solve(Nonogram *N) {

    size_t Ls_size = sizeof(NonogramLineDevice *) * (N->w() + N->h());
    NonogramLineDevice **Ls_tmp = (NonogramLineDevice **)malloc(Ls_size);
    unsigned i = 0;
    for (; i < N->h(); i++) {
        const unsigned *constr = N->row_constr[i].data();
        const unsigned constr_len = (unsigned) N->row_constr[i].size();
        Ls_tmp[i] = (NonogramLineDevice *)ngline_dev_init_host(constr, constr_len);
    }

    for (unsigned j = 0; j < N->w(); j++) {
        const unsigned *constr = N->col_constr[j].data();
        const unsigned constr_len = (unsigned) N->col_constr[j].size();
        Ls_tmp[i + j] = (NonogramLineDevice *)ngline_dev_init_host(constr, constr_len);
    }

    void *Ls_tmpptr;
    cudaMalloc(&Ls_tmpptr, Ls_size);
    cudaMemcpy(Ls_tmpptr, Ls_tmp, Ls_size, cudaMemcpyHostToDevice);
    NonogramLineDevice **Ls = (NonogramLineDevice **)Ls_tmpptr;
    Board2DDevice *B = (Board2DDevice *)board2d_to_device(N->board);

    ngline_init_kernel<<<B->w + B->h, 1>>>(B, Ls);

    bool dirty_h;
    do {
        ngline_row_solve_kernel<<< B->h, 1 >>> (B, Ls);
        ngline_row_solve_kernel<<< B->w, 1 >>> (B, Ls);

        cudaMemcpy(&dirty_h, &B->dirty, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (dirty_h);

    board2d_dev_to_host((void *)B, N->board);

}

