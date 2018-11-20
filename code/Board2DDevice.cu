//
// Created by Benjamin Huang on 11/19/2018.
//

#include "Board2DDevice.h"

__host__
void *board2d_to_device(Board2D<Nonogram::Color> B_host) {

    Board2DDevice tmp;
    tmp.w = B_host.w;
    tmp.h = B_host.h;

    void *B_data_dev;
    size_t B_data_size = sizeof(char) * 2 * B_host.w * B_host.h;
    cudaMalloc(&B_data_dev, B_data_size);
    cudaMemcpy(B_data_dev, B_host.data, B_data_size, cudaMemcpyHostToDevice);

    tmp.data = (char *)B_data_dev;
    tmp.dataCM = &tmp.data[B_host.w * B_host.h];

    void *B_dev;
    cudaMalloc(&B_dev, sizeof(Board2DDevice));
    cudaMemcpy(B_dev, &tmp, sizeof(Board2DDevice), cudaMemcpyHostToDevice);

    return B_dev;

}

__host__
void board2d_dev_to_host(void* B_dev_v, Board2D<Nonogram::Color> B_host) {
    Board2DDevice *B_dev = (Board2DDevice *)B_dev_v;

    if (B_dev->w != B_host.w || B_dev->h != B_host.h) {
        return;
    }

    size_t B_data_size = sizeof(char) * 2 * B_dev->w * B_dev->h;
    cudaMemcpy(B_host.data, B_dev->data, B_data_size, cudaMemcpyDeviceToHost);
}

__host__
void *board2d_dev_init(unsigned w, unsigned h, char val) {

    Board2DDevice tmp;
    tmp.w = w;
    tmp.h = h;

    void *B_data_dev;
    size_t B_data_size = sizeof(char) * 2 * w * h;
    cudaMalloc(&B_data_dev, B_data_size);

    tmp.data = (char *)B_data_dev;
    tmp.dataCM = &tmp.data[w * h];

    void *B_dev;
    cudaMalloc(&B_dev, sizeof(Board2DDevice));
    cudaMemcpy(B_dev, &tmp, sizeof(Board2DDevice), cudaMemcpyHostToDevice);

    return B_dev;

}

__host__
void board2d_dev_free(Board2DDevice *B) {
    cudaFree(B->data);
    cudaFree(B);
}

__device__
void board2d_dev_elem_set(Board2DDevice *B, unsigned x, unsigned y, char val) {
    B->data[y * B->w + x] = val;
    B->dataCM[x * B->h + y] = val;
    B->dirty = true;
}

__device__
char board2d_dev_elem_get_rm(Board2DDevice *B, unsigned x, unsigned y) {
    return B->data[y * B->w + x];
}

__device__
char board2d_dev_elem_get_cm(Board2DDevice *B, unsigned x, unsigned y) {
    return B->dataCM[x * B->h + y];
}

__device__
char *board2d_dev_row_ptr_get(Board2DDevice *B, unsigned index) {
    return &B->data[index * B->w];
}

__device__
char *board2d_dev_col_ptr_get(Board2DDevice *B, unsigned index) {
    return &B->dataCM[index * B->h];
}