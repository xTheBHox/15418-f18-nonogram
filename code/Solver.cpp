//
// Created by Benjamin Huang on 12/10/2018.
//

#include "Solver.h"
#include "Board2DDevice.h"

#define DEBUG

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
        printf("Init row %d len=%d constr_len=%d\n", L->line_index, L->len, L->constr_len);
#endif
#endif
    }
    else {
        L->data = board2d_dev_col_ptr_get(B, L->line_index);
#ifndef __NVCC__
#ifdef DEBUG
        printf("Init col %d len=%d constr_len=%d\n", L->line_index, L->len, L->constr_len);
#endif
#endif
    }

    ngline_init_dev(L);

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

bool ng_solve_loop(NonogramLineDevice *Ls, Board2DDevice *B) {

    do {
        B->dirty = false;

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            for (unsigned ri = 0; ri < L->constr_len; ri++) {
                ngline_dev_run_solve(L, B, ri);
            }
        }

        if (B->dirty) continue;

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            ngline_dev_block_solve(L, B);
        }

    } while (B->dirty);

    for (unsigned i = 0; i < B->h; i++) {
        // check that all the rows are solved. Don't need to check columns
        NonogramLineDevice *L = &Ls[i];
        if (!L->solved) return false;
    }

    return true;

}

bool nghyp_solve_loop(NonogramLineDevice *Ls, Board2DDevice *B) {

    do {
        B->dirty = false;

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            for (unsigned ri = 0; ri < L->constr_len; ri++) {
                nglinehyp_dev_run_solve(L, B, ri);
            }
            if (!B->valid) return false;
        }

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            nglinehyp_dev_block_solve(L, B);
            if (!B->valid) return false;
        }

        std::cout << std::endl << B;

    } while (B->dirty);

    for (unsigned i = 0; i < B->h; i++) {
        // check that all the rows are solved. Don't need to check columns
        NonogramLineDevice *L = &Ls[i];
        if (!L->solved) return false;
    }

    return true;

}

void ng_solve_seq(NonogramLineDevice *Ls_host, Board2DDevice *B_host) {

    NonogramLineDevice *Ls_dev = Ls_host;
    Board2DDevice *B_dev = B_host;

    // Initialize the runs
    unsigned block_cnt = B_host->w + B_host->h;
    for (unsigned i = 0; i < block_cnt; i++) {
        ngline_init_kernel(B_dev, Ls_dev, i);
    }

    bool solved = false;

    while (!solved) {

        solved = ng_solve_loop(Ls_dev, B_dev);
        if (solved) break;

        Heuristic *X;
        nghyp_heuristic_init(Ls_dev, B_dev, &X);
        nghyp_heuristic_fill(Ls_dev, B_dev, X);

#ifdef DEBUG
        std::cout << "Simple solving dead-end:" << std::endl;
        std::cout << B_dev;
#endif

        while (!B_dev->dirty) {
            nghyp_heuristic_max(X);

            HypotheticalBoard H_b = nghyp_init(Ls_dev, B_dev);
            nghyp_heuristic_update(&H_b, X, NGCOLOR_BLACK);

            solved = nghyp_solve_loop(H_b.Ls, H_b.B);

#ifdef DEBUG
            std::cout << "Hypothesis on black (" << H_b.row << ", " << H_b.col << ")" << std::endl;
            std::cout << "Sovled: " << solved << "\tValid: " << H_b.B->valid << std::endl;
            std::cout << H_b.B;
#endif

            if (solved) {
                // Copy to actual board
                nghyp_hyp_confirm(H_b, B_dev, &Ls_dev);
                nghyp_free(H_b);
                break;
            } else {
                // Check for contradiction
                if (!nghyp_valid_check(&H_b, B_dev)) {
                    nghyp_free(H_b);
                    break;
                }
            }

            HypotheticalBoard H_w = nghyp_init(Ls_dev, B_dev);
            nghyp_heuristic_update(&H_w, X, NGCOLOR_WHITE);

            solved = nghyp_solve_loop(H_w.Ls, H_w.B);

#ifdef DEBUG
            std::cout << "Hypothesis on white (" << H_w.row << ", " << H_w.col << ")" << std::endl;
            std::cout << "Sovled: " << solved << "\tValid: " << H_w.B->valid << std::endl;
            std::cout << H_w.B;
#endif

            if (solved) {
                // Copy to actual board
                nghyp_hyp_confirm(H_w, B_dev, &Ls_dev);
                nghyp_free(H_b);
                nghyp_free(H_w);
                break;
            } else {
                // Check for contradiction
                if (!nghyp_valid_check(&H_w, B_dev)) {
                    nghyp_free(H_b);
                    nghyp_free(H_w);
                    break;
                }
            }
            nghyp_common_set(&H_b, &H_w, B_dev);
            nghyp_free(H_b);
            nghyp_free(H_w);

        }

        nghyp_heuristic_free(X);

        // Check for duplicates if not solved

    }

    board2d_cleanup_dev(B_host, B_dev);
    ng_linearr_free_dev(Ls_dev);

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
