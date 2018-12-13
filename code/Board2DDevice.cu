//
// Created by Benjamin Huang on 11/19/2018.
//

#include "Board2DDevice.h"

Board2DDevice *board2d_init_host(unsigned w, unsigned h) {

    // Allocate the board header
    Board2DDevice *B = (Board2DDevice *)malloc(sizeof(Board2DDevice));

    if (B == NULL) {
        fprintf(stderr, "Failed to allocate board header\n");
        return NULL;
    }

    B->w = w;
    B->h = h;
    B->dirty = true;
    B->valid = true;
    B->solved = false;

    // Allocate the board data array

    size_t b_len = w * h;
    B->data = (NonogramColor *)calloc(2 * b_len, sizeof(NonogramColor));

    if (B->data == NULL) {
        fprintf(stderr, "Failed to allocate board data array\n");
        free(B);
        return NULL;
    }

    B->dataCM = &B->data[b_len];
    return B;

}

Board2DDevice *board2d_init_dev(Board2DDevice *B_host) {

#ifdef __NVCC__
    Board2DDevice B_tmp_var;
    Board2DDevice *B_tmp = &B_tmp_var;
    void *B_dev;
    size_t B_arr_size = sizeof(NonogramColor) * B_host->w * B_host->h;

    *B_tmp = *B_host;
    cudaCheckError(cudaMalloc((void **)&(B_tmp->data), 2 * B_arr_size));

    cudaCheckError(cudaMemcpy((void *)B_tmp->data, (void *)B_host->data, 2 * B_arr_size, cudaMemcpyHostToDevice));
    B_tmp->dataCM = &B_tmp->data[B_host->w * B_host->h];

    cudaCheckError(cudaMalloc(&B_dev, sizeof(Board2DDevice)));
    cudaCheckError(cudaMemcpy(B_dev, (void *)B_tmp, sizeof(Board2DDevice), cudaMemcpyHostToDevice));

    return (Board2DDevice *)B_dev;
#else
    return B_host;
#endif

}

void board2d_free_host(Board2DDevice *B) {

    free(B->data);
    free(B);

}

void board2d_cleanup_dev(Board2DDevice *B_host, Board2DDevice *B_dev) {

#ifdef __NVCC__
    Board2DDevice B_tmp_var;
    Board2DDevice *B_tmp = &B_tmp_var;

    cudaCheckError(cudaMemcpy((void *)B_tmp, (void *)B_dev, sizeof(Board2DDevice), cudaMemcpyDeviceToHost));

    size_t B_arr_size = sizeof(NonogramColor) * B_host->w * B_host->h;
    cudaCheckError(cudaMemcpy((void *)B_host->data, (void *)B_tmp->data, B_arr_size, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree((void *)B_tmp->data));
    cudaCheckError(cudaFree((void *)B_dev));
#else
    return;
#endif

}

Board2DDevice *board2d_deepcopy_host(Board2DDevice *B) {

    Board2DDevice *B_copy = board2d_init_host(B->w, B->h);
    memcpy((void *)B_copy->data, (void *)B->data, 2 * B->w * B->h * sizeof(NonogramColor));
    B_copy->solved = B->solved;
    B_copy->valid = B->valid;
    B_copy->dirty = true;
    return B_copy;

}

std::ostream &operator<<(std::ostream &os, Board2DDevice *B) {

    for (unsigned r = 0; r < B->h; r++) {
        for (unsigned c = 0; c < B->w; c++) {
            char sym = 'X';
            switch (board2d_dev_elem_get_rm(B, c, r)) {
                case NGCOLOR_BLACK: {
                    sym = '#';
                    break;
                }
                case NGCOLOR_UNKNOWN: {
                    sym = '?';
                    break;
                }
                case NGCOLOR_WHITE: {
                    sym = '.';
                    break;
                }
            }
            os << sym;
        }
        os << std::endl;
    }
    return os;

}


__device__
void board2d_dev_init_copy(Board2DDevice *B_dst, const Board2DDevice *B_src) {
    B_dst->w = B_src->w;
    B_dst->h = B_src->h;
    B_dst->dirty = B_src->dirty;
    B_dst->solved = B_src->solved;
    B_dst->valid = B_src->valid;
}

__device__
void board2d_dev_mutableonly_copy(Board2DDevice *B_dst, const Board2DDevice *B_src) {
    B_dst->dirty = B_src->dirty;
    B_dst->solved = B_src->solved;
    B_dst->valid = B_src->valid;
}
