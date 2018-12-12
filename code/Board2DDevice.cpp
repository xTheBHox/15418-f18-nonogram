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
    void *B_dev_data_val;
    void *B_dev_data = &B_dev_data_val;
    cudaCheckError(cudaMemcpy(B_dev_data, (void *)&(B_dev->data), sizeof(void *), cudaMemcpyDeviceToHost));

    size_t B_arr_size = sizeof(NonogramColor) * B_host->w * B_host->h;
    cudaCheckError(cudaMemcpy((void *)B_host->data, B_dev_data_val, B_arr_size, cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(B_dev_data_val));
    cudaCheckError(cudaFree((void *)B_dev));
#else
    return;
#endif

}

Board2DDevice *board2d_deepcopy_host(Board2DDevice *B) {

    Board2DDevice *B_copy = board2d_init_host(B->w, B->h);
    memcpy((void *)B_copy->data, (void *)B->data, 2 * B->w * B->h * sizeof(NonogramColor));
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
void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, NonogramColor val) {
    B->data[y * B->w + x] = val;
    B->dataCM[x * B->h + y] = val;
    B->dirty = true;
}

__host__ __device__
NonogramColor board2d_dev_elem_get_rm(const Board2DDevice *B, unsigned x, unsigned y) {
    return B->data[y * B->w + x];
}

__device__
NonogramColor board2d_dev_elem_get_cm(const Board2DDevice *B, unsigned x, unsigned y) {
    return B->dataCM[x * B->h + y];
}

__device__
NonogramColor *board2d_dev_row_ptr_get(const Board2DDevice *B, unsigned index) {
    return &B->data[index * B->w];
}

__device__
NonogramColor *board2d_dev_col_ptr_get(const Board2DDevice *B, unsigned index) {
    return &B->dataCM[index * B->h];
}