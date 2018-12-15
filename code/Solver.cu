//
// Created by Benjamin Huang on 12/10/2018.
//

#include "Solver.h"

// #define DEBUG

#ifdef __NVCC__
__global__
void ngline_init_kernel(Board2DDevice *B, NonogramLineDevice *Ls) {
    unsigned i = threadIdx.x;

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
        unsigned Ls_size, const unsigned B_data_size, const bool Ls_shared) {

    unsigned i = threadIdx.x;
    // BEGIN Shared memory copy. Need all of Ls and B in shared memory.
    // Get pointers
    extern __shared__ int _smem[];
    char *smem = (char *) _smem;
    Board2DDevice *B = (Board2DDevice *)smem;
    NonogramLineDevice *Ls;

    if (Ls_shared) {
        Ls = (NonogramLineDevice *)(smem + sizeof(Board2DDevice));
    }
    else {
        Ls_size = 0;
        Ls = Ls_global;
    }

    if (i == 0) {
        board2d_dev_init_copy(B, B_global);
        B->data = (NonogramColor *)(smem + sizeof(Board2DDevice) + Ls_size);
        B->dataCM =(NonogramColor *)(smem + sizeof(Board2DDevice) + Ls_size + B_data_size / 2);
        // WARNING DO NOT reference B->solved before updated by all threads!!!
        B->solved = true;
    }


    NonogramLineDevice *L_global = &Ls_global[i];
    NonogramLineDevice *L = &Ls[i];
    if (Ls_shared) {
        // Copy Ls;
        *L = *L_global;
    }

    // Need to make sure master finished copying B and the B pointers
    __syncthreads();

    NonogramColor *L_data_global;
    if (Ls_shared) {
        // Update the Ls data pointers
        if (L->line_is_row) {
            L->data = board2d_dev_row_ptr_get(B, L->line_index);
        }
        else {
            L->data = board2d_dev_col_ptr_get(B, L->line_index);
        }

        // Copy respective board row/col. WARNING DOES NOT RESPECT INTERFACE!!!
        for (unsigned j = 0; j < L->len; j++) {
            L->data[j] = L_global->data[j];
        }
    }
    else {
        // Update the Ls data pointers
        L_data_global = L->data;
        if (L->line_is_row) {
            L->data = board2d_dev_row_ptr_get(B, L->line_index);
        }
        else {
            L->data = board2d_dev_col_ptr_get(B, L->line_index);
        }

        for (unsigned j = 0; j < L->len; j++) {
            L->data[j] = L_data_global[j];
        }
    }

    __syncthreads();
    // END Shared memory copy.
#ifdef DEBUG
    if (i == 0) printf("Shared memory initialized. Line solvers starting...\n");
#endif

    do {

        B->dirty = false;

        // Because of the nature of a solvable Nonogram, it is possible to
        // simultaneously do the rows and columns because they will only ever
        // write correct values.
        if (!L->solved) ngline_dev_run_solve(L, B);
        __syncthreads();

        if (B->dirty) continue;

        if (!L->solved) ngline_dev_block_solve(L, B);

        // TODO remove this too
        __syncthreads();

    } while (B->dirty);

    if (L->line_is_row && !L->solved) B->solved = false;
    // Now B->solved is valid
#ifdef DEBUG
    if (i == 0) printf("Line solvers complete. Updating global memory...\n");
#endif
    // BEGIN Global memory copy-back
    if (Ls_shared) {
        ngline_dev_mutableonly_copy(L_global, L);
        for (unsigned j = 0; j < L->len; j++) {
            L_global->data[j] = L->data[j];
        }
    }
    else {
        for (unsigned j = 0; j < L->len; j++) {
            L_data_global[j] = L->data[j];
        }
        L->data = L_data_global;
    }
#ifdef DEBUG
if (i == 0) printf("Line solvers updated.\n");
#endif
    if (i == 0) {
        board2d_dev_mutableonly_copy(B_global, B);
    }
#ifdef DEBUG
if (i == 0) printf("Board updated.\n");
#endif
    // END Global memory copy-back

}
#else
bool ng_solve_loop(NonogramLineDevice *Ls, Board2DDevice *B) {

    do {
        B->dirty = false;

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            ngline_dev_run_solve(L, B);
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

#ifdef __NVCC__
__global__
void nghyp_solve_loop_kernel(NonogramLineDevice *Ls_global, Board2DDevice *B_global, Heuristic *X,
        unsigned Ls_size, const unsigned B_data_size, const bool Ls_shared,
        NonogramLineDevice *Ls_original,
        unsigned *B_lock, volatile int *B_status) {

    unsigned i = threadIdx.x;

    NonogramColor H_color;
    if (blockIdx.x == 0) {
        H_color = NGCOLOR_BLACK;
    }
    else {
        H_color = NGCOLOR_WHITE;
        if (!Ls_shared) Ls_global = (NonogramLineDevice *)(((char *)Ls_global) + Ls_size);
    }

    // BEGIN Shared memory copy. Need all of Ls and B in shared memory.
    // Get pointers
    extern __shared__ int _smem[];
    char *smem = (char *) _smem;
    Board2DDevice *B = (Board2DDevice *)smem;
    NonogramLineDevice *Ls;

    if (Ls_shared) {
        Ls = (NonogramLineDevice *)(smem + sizeof(Board2DDevice));
    }
    else {
        Ls_size = 0;
        Ls = Ls_global;
    }

    if (i == 0) {
        board2d_dev_init_copy(B, B_global);
        B->data = (NonogramColor *)(smem + sizeof(Board2DDevice) + Ls_size);
        B->dataCM =(NonogramColor *)(smem + sizeof(Board2DDevice) + Ls_size + B_data_size / 2);
        // WARNING DO NOT reference B->solved before updated by all threads!!!
        B->solved = true;
    }


    NonogramLineDevice *L_global;
    if (Ls_shared) L_global = &Ls_global[i];
    else L_global = &Ls_original[i];

    NonogramLineDevice *L = &Ls[i];
    if (Ls_shared) {
        // Copy Ls;
        *L = *L_global;
    }

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
        L->data[j] = L_global->data[j];
    }

    __syncthreads();
    // END Shared memory copy.

    // Set up the hypothesis in memory
#ifdef DEBUG
    if (i == 0) printf("Hypothesis: %d in (%d, %d)\n", H_color, X->r_max, X->c_max);
#endif
    HypotheticalBoard H;
    H.B = B;
    H.Ls = Ls;
    if (i == 0) {
        nghyp_heuristic_update(&H, X, H_color);
    }
    else {
        H.row = X->r_max;
        H.col = X->c_max;
        H.guess_color = H_color;
    }

    do {
        B->dirty = false;
        // Because of the nature of a solvable Nonogram, it is possible to
        // simultaneously do the rows and columns because they will only ever
        // write correct values.

        if (!L->solved) nglinehyp_dev_run_solve(L, B);

        __syncthreads();
        if (!B->valid) {
            break;
        }

        // TOOD decide whether this is useful
        // if (B->dirty) continue;

        if (!L->solved) nglinehyp_dev_block_solve(L, B);

        // TODO remove this too
        __syncthreads();
        if (!B->valid) {
            break;
        }

    } while (B->dirty);

    if (L->line_is_row && !L->solved) B->solved = false;
    // Now B->solved is valid
#ifdef DEBUG
    if (i == 0) {
        printf("Solved: %d\t Valid: %d\n", B->solved, B->valid);
    }
#endif

    if (i == 0) {
        // Try to take the lock
        unsigned k;
        do {
            k = atomicCAS(B_lock, 0, 1);
        } while (k == 1);
    }
    __syncthreads();

    if (*B_lock == 1) {
        // This is the first hypothesis to finish
        if (!B->valid) {
            if (i == 0) *B_status = -1;
        }
        else if (B->solved) {
            // We have solved the board. Don't care about the other hypothesis.
            // Don't care about the lines.
            if (i == 0) {
                *B_status = 1;
                board2d_dev_mutableonly_copy(B_global, B);
            }
            for (unsigned j = 0; j < L->len; j++) {
                L_global->data[j] = L->data[j];
            }
        }
        else {
            // Dead end. Put assumptions in the board.
            if (i == 0) *B_status = 0;
            for (unsigned j = 0; j < L->len; j++) {
                nghyp_hyp_assume(L, L_global, B_global, j);
            }
        }
        __threadfence();
        if (i == 0) {
            *B_lock = 2;
        }
    }
    else {

#ifdef DEBUG
        if (i == 0) {
            if (*B_lock != 2) {
                printf("Invalid lock value!\n");
            }
        }
#endif
        if (*B_status == -1) {
            // The other hypothesis hit a contradiction
#ifdef DEBUG
            if (i == 0) {
                if (!B->valid) printf("Error: Both hypotheses contradict\n");
            }
#endif
            for (unsigned j = 0; j < L->len; j++) {
                nghyp_confirm_assume(L, L_global, B_global, j);
            }
            return;
        }
        if (*B_status == 1) {
            // The other hypothesis solved the board already
#ifdef DEBUG
            if (i == 0) {
                if (B->solved) printf("Error: Both hypotheses solved the board\n");
            }
#endif
            return;
        }
#ifdef DEBUG
        if (i == 0) {
            if (*B_status != 0) {
                printf("Invalid status value!\n");
            }
        }
#endif
        if (!B->valid) {
            for (unsigned j = 0; j < L->len; j++) {
                nghyp_confirm_unassume(L_global, B_global, j);
            }
            nglinehyp_dev_block_solve(L_global, B_global);
            return;
        }
        else if (B->solved) {
            // We have solved the board. Don't care about the other hypothesis.
            // Don't care about the lines.
            if (i == 0) {
                board2d_dev_mutableonly_copy(B_global, B);
            }
            for (unsigned j = 0; j < L->len; j++) {
                L_global->data[j] = L->data[j];
            }
            return;
        }
        else {
            // Dead end. Put assumptions in the board.
            for (unsigned j = 0; j < L->len; j++) {
                nghyp_hyp_unassume(L, L_global, B_global, j);
            }
            nglinehyp_dev_block_solve(L_global, B_global);
            return;
        }
    }

}
void nghyp_solve_loop_kernel_prep(
        NonogramLineDevice *Ls_global, Board2DDevice *B_global, Heuristic *X, const unsigned thread_cnt,
        const unsigned smem_size, const unsigned Ls_size, const unsigned B_data_size,
        const bool Ls_shared) {

#ifdef DEBUG
    std::cout << "Initializing lock..." << std::endl;
#endif
    // Make the locks and the status
    unsigned *B_lock;
    cudaCheckError(cudaMalloc((void **)&B_lock, sizeof(unsigned)));
    cudaCheckError(cudaMemset((void *)B_lock, 0, sizeof(unsigned)));
    int *B_status;
    cudaCheckError(cudaMalloc((void **)&B_status, sizeof(int)));


    if (!Ls_shared) {
#ifdef DEBUG
        std::cout << "Lock initialized. Copying global line solver data..." << std::endl;
#endif

        NonogramLineDevice *Ls_copy = ng_linearr_deepcopy_dev_double(Ls_global, Ls_size);

#ifdef DEBUG
        std::cout << "Hypothesis kernel starting..." << std::endl;
#endif
        nghyp_solve_loop_kernel<<<2, thread_cnt, smem_size>>>(
            Ls_copy, B_global, X,
            Ls_size, B_data_size, Ls_shared,
            Ls_global, B_lock, B_status);
#ifdef DEBUG
        cudaCheckError(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaCheckError(cudaGetLastError());
#endif
        cudaCheckError(cudaFree(Ls_copy));
    }
    else {
#ifdef DEBUG
        std::cout << "Lock initialized. Hypothesis kernel starting..." << std::endl;
#endif
        nghyp_solve_loop_kernel<<<2, thread_cnt, smem_size>>>(
            Ls_global, B_global, X,
            Ls_size, B_data_size, Ls_shared,
            NULL, B_lock, B_status);
#ifdef DEBUG
        cudaCheckError(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaCheckError(cudaGetLastError());
#endif
    }

#ifdef DEBUG
    std::cout << "Hypothesis kernel finished." << std::endl;
#endif
    cudaCheckError(cudaFree(B_lock));
    cudaCheckError(cudaFree(B_status));

}

#else
bool nghyp_solve_loop(NonogramLineDevice *Ls, Board2DDevice *B) {

    do {
        B->dirty = false;

        for (unsigned i = 0; i < B->h + B->w; i++) {
            NonogramLineDevice *L = &Ls[i];
            if (L->solved) continue;
            nglinehyp_dev_run_solve(L, B);
            if (!B->valid) return false;

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
#endif

#ifdef __NVCC__
void ng_solve_par(NonogramLineDevice *Ls_host, Board2DDevice *B_host) {

    unsigned thread_cnt = B_host->w + B_host->h;
    unsigned B_data_size = 2 * B_host->w * B_host->h * sizeof(NonogramColor);
    unsigned Ls_size = thread_cnt * sizeof(NonogramLineDevice);
    unsigned smem_size = Ls_size + B_data_size + sizeof(Board2DDevice);
    bool Ls_shared = true;

    if (2 * smem_size > 48 * 1024) {
        smem_size = B_data_size + sizeof(Board2DDevice);
        Ls_shared = false;
    }

    NonogramLineDevice *Ls_dev;
    Board2DDevice *B_dev;

    // Move structures to device memory

#ifdef DEBUG
    std::cout << "B_data_size: " << B_data_size << std::endl;
    std::cout << "Ls_size: " << Ls_size << std::endl;
    std::cout << "smem_size: " << smem_size << std::endl;
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
    cudaCheckError(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaCheckError(cudaGetLastError());
#endif

    bool solved = false;
    bool dirty = false;
    while (!solved) {

#ifdef DEBUG
        std::cout << "Launching simple kernel..." << std::endl;
#endif
        ng_solve_loop_kernel<<<1, thread_cnt, smem_size>>>(Ls_dev, B_dev, Ls_size, B_data_size, Ls_shared);
#ifdef DEBUG
        cudaCheckError(cudaGetLastError());
        cudaDeviceSynchronize();
        cudaCheckError(cudaGetLastError());
        std::cout << "Simple kernel finished." << std::endl;
#endif
        // Check if the board is solved
        cudaCheckError(cudaMemcpy(&solved, &B_dev->solved, sizeof(bool), cudaMemcpyDeviceToHost));
        if (solved) break;

#ifdef DEBUG
        std::cout << "Simple solving dead-end. Initializing heuristic..." << std::endl;
#endif
        Heuristic *X_dev = nghyp_heuristic_init_dev(B_host->w, B_host->h);
#ifdef DEBUG
        std::cout << "Filling heuristic..." << std::endl;
#endif
        nghyp_heuristic_fill(Ls_dev, B_dev, X_dev, B_host->w, B_host->h);

        do {
#ifdef DEBUG
            std::cout << "Heuristic initialized. Finding maximum..." << std::endl;
#endif
            nghyp_heuristic_max_dev(X_dev, B_host->w * B_host->h);

#ifdef DEBUG
            std::cout << "Cell selected. Preparing hypothesis kernel..." << std::endl;
#endif
            nghyp_solve_loop_kernel_prep(Ls_dev, B_dev, X_dev, thread_cnt,
                    smem_size, Ls_size, B_data_size, Ls_shared);
            cudaCheckError(cudaMemcpy(&solved, &B_dev->solved, sizeof(bool), cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(&dirty, &B_dev->dirty, sizeof(bool), cudaMemcpyDeviceToHost));

            if (solved) {
                break;
            }
        } while (!dirty);

#ifdef DEBUG
        std::cout << "Freeing heuristc..." << std::endl;
#endif
        nghyp_heuristic_free_dev(X_dev);

    }

    TIMER_STOP(solve_loop);

#ifdef DEBUG
    std::cout << "Cleaning up device..." << std::endl;
#endif

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
        nghyp_heuristic_fill(Ls_dev, B_dev, X, X->w, X->h);

#ifdef DEBUG
        std::cout << "Simple solving dead-end:" << std::endl;
        // std::cout << B_dev;
#endif

        while (!B_dev->dirty) {
            nghyp_heuristic_max(X);

            HypotheticalBoard H_b = nghyp_init(Ls_dev, B_dev);
            nghyp_heuristic_update(&H_b, X, NGCOLOR_BLACK);

            solved = nghyp_solve_loop(H_b.Ls, H_b.B);

#ifdef DEBUG
            std::cout << "Hypothesis on black (" << H_b.row << ", " << H_b.col << ")" << std::endl;
            std::cout << "Sovled: " << solved << "\tValid: " << H_b.B->valid << std::endl;
            // std::cout << H_b.B;
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
            // std::cout << H_w.B;
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
#ifdef DISP
    initscr();
#endif

    ng_solve_seq(pLs_host, pB_host);

#ifdef DISP
    mvprintw((*pB_host)->h, 0, "Completed. Press any key to exit.");
    getch();
    endwin();
#endif
#endif

}
