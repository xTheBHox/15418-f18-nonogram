//
// Created by Benjamin Huang on 12/10/2018.
//

#include "Solver.h"

// #define DEBUG

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

    NonogramLineDevice *L = &Ls[i];

    if (L->line_is_row) {
        L->data = board2d_dev_row_ptr_get(B, L->line_index);
    }
    else {
        L->data = board2d_dev_col_ptr_get(B, L->line_index);
    }

    ngline_init_dev(L);

}
#else
void ngline_init(Board2DDevice *B, NonogramLineDevice *Ls, unsigned i) {

    NonogramLineDevice *L = &Ls[i];

    if (L->line_is_row) {
        L->data = board2d_dev_row_ptr_get(B, L->line_index);

#ifdef DEBUG
        printf("Init row %d len=%d constr_len=%d\n", L->line_index, L->len, L->constr_len);
#endif

    }
    else {
        L->data = board2d_dev_col_ptr_get(B, L->line_index);

#ifdef DEBUG
        printf("Init col %d len=%d constr_len=%d\n", L->line_index, L->len, L->constr_len);
#endif
    }

    ngline_init_dev(L);

}
#endif

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

#ifdef __NVCC__
__global__ void
ng_solve_loop_kernel(NonogramLineDevice *Ls_global, Board2DDevice *B_global,
        unsigned Ls_size, unsigned B_data_size) {

    unsigned i = threadIdx.x;
    // BEGIN Shared memory copy. Need all of Ls and B in shared memory.
    // Get pointers
    extern __shared__ int _smem[];
    char *smem = (char *) _smem;
    NonogramLineDevice *Ls = (NonogramLineDevice *)smem;
    Board2DDevice *B = (Board2DDevice *)(smem + Ls_size + B_data_size);

    if (i == 0) {
        *B = *B_global;
        B->data = (NonogramColor *)(smem + Ls_size);
        B->dataCM =(NonogramColor *)(smem + Ls_size + B_data_size / 2);
        // WARNING DO NOT reference B->solved before updated by all threads!!!
        B->solved = true;
    }

    // Copy Ls;
    NonogramLineDevice *L_global = &Ls_global[i];
    NonogramLineDevice *L = &Ls[i];
    *L = *L_global;

    // Need to make sure master finished copying B and the B pointers
    __syncthreads();

    // Update the Ls data pointers
    if (L->line_is_row) {
        L->data = board2d_dev_row_ptr_get(B, L->line_index);
    }
    else {
        L->data = board2d_dev_col_ptr_get(B, L->line_index);
    }

    // Copy respective board row/col. WARNING DOES NOT RESPECT INTERFACE!!!
    for (unsigned j = 0; j < L->len; j++) {
        L->data[j] = Ls_global[i].data[j];
    }

    __syncthreads();
    // END Shared memory copy.

    do {
    /*
    // Because of the nature of a solvable Nonogram, it is possible to
    // simultaneously do the rows and columns because they will only ever
    // write correct values.
    if (!L->solved) {
        for (unsigned ri = 0; ri < L->constr_len; ri++) {
            ngline_dev_run_solve(L, B, ri);
        }
    }
    */
        if (L->line_is_row) { // Solve rows
            if (!L->solved) {
                for (unsigned ri = 0; ri < L->constr_len; ri++) {
                    ngline_dev_run_solve(L, B, ri);
                }
            }
        }
        __syncthreads();
        if (!L->line_is_row) { // Solve columns
            if (!L->solved) {
                for (unsigned ri = 0; ri < L->constr_len; ri++) {
                    ngline_dev_run_solve(L, B, ri);
                }
            }
        }
        __syncthreads();

        if (B->dirty) continue;

        if (!L->solved) ngline_dev_block_solve(L, B);

        __syncthreads();

    } while (B->dirty);

    if (L->line_is_row && !L->solved) B->solved = false;
    // Now B->solved is valid

    // BEGIN Global memory copy-back

    ngline_dev_mutableonly_copy(L_global, L);
    for (unsigned j = 0; j < L->len; j++) {
        L->data[j] = Ls_global[i].data[j];
    }

    if (i == 0) {
        board2board2d_dev_mutableonly_copy(B_global, B);
    }

    // END Global memory copy-back

}
#else
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
#endif

bool nghyp_solve_loop(NonogramLineDevice *Ls, Board2DDevice *B) {

    do {
        B->dirty = false;

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            for (unsigned ri = 0; ri < L->constr_len; ri++) {
                nglinehyp_dev_run_solve(L, B, ri);
                if (!B->valid) return false;
            }
        }

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            nglinehyp_dev_block_solve(L, B);
            if (!B->valid) return false;
        }

    } while (B->dirty);

    for (unsigned i = 0; i < B->h; i++) {
        // check that all the rows are solved. Don't need to check columns
        NonogramLineDevice *L = &Ls[i];
        if (!L->solved) return false;
    }

    return true;

}

#ifdef __NVCC__
void ng_solve_par(NonogramLineDevice *Ls_host, Board2DDevice *B_host) {
#ifdef DEBUG
    std::cout << "ng_solve called" << std::endl;
#endif

    unsigned thread_cnt = B_host->w + B_host->h;
    unsigned B_data_size = 2 * B_host->w * B_host->h * sizeof(NonogramColor);
    unsigned Ls_size = thread_cnt * sizeof(NonogramLineDevice);
    unsigned smem_size = Ls_size + B_data_size + sizeof(Board2DDevice);

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

#ifdef DEBUG
    std::cout << "Lines initializing..." << std::endl;
#endif

    TIMER_START(solve_loop);
    ngline_init_kernel<<<1, thread_cnt>>>(B_dev, Ls_dev);

#ifdef DEBUG
    std::cout << "Lines alternating..." << std::endl;
#endif

    do {
        //cudaMemcpy(&B_dev->dirty, &B_host->dirty, sizeof(bool), cudaMemcpyHostToDevice);
        ng_solve_loop_kernel<<<1, thread_cnt, smem_size>>>(Ls_dev, B_dev, Ls_data_size, B_data_size);
        //cudaMemcpy(&B_host->dirty, &B_dev->dirty, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (B_host->dirty);

    TIMER_STOP(solve_loop);

    board2d_cleanup_dev(B_host, B_dev);
    ng_linearr_free_dev(Ls_dev);

}
#else
void ng_solve_seq(NonogramLineDevice **pLs_dev, Board2DDevice **pB_dev) {

    NonogramLineDevice *Ls_dev = *pLs_dev;
    Board2DDevice *B_dev = *pB_dev;

    TIMER_START(solve_loop);

    // Initialize the runs
    unsigned block_cnt = B_dev->w + B_dev->h;
    for (unsigned i = 0; i < block_cnt; i++) {
        ngline_init(B_dev, Ls_dev, i);
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
                nghyp_hyp_confirm(&H_b, &B_dev, &Ls_dev);
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
                nghyp_hyp_confirm(&H_w, &B_dev, &Ls_dev);
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
            // Check for duplicates if not solved
            nghyp_common_set(&H_b, &H_w, B_dev);
            nghyp_free(H_b);
            nghyp_free(H_w);

            // At this point, try a different hypothesis if it didn't gain any new cells

        }

        nghyp_heuristic_free(X);

    }

    TIMER_STOP(solve_loop);
    *pB_dev = B_dev;
    *pLs_dev = Ls_dev;

}
#endif

void ng_solve(NonogramLineDevice **pLs_host, Board2DDevice **pB_host) {

#ifdef __NVCC__
    ng_solve_par(*pLs_host, *pB_host);
#else
    ng_solve_seq(pLs_host, pB_host);
#endif

}
