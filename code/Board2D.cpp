//
// Created by Benjamin Huang on 11/12/2018.
//

#include "Board2D.h"

template <class E>
void Board2D::elem_set(size_t x, size_t y, E val) {
#ifdef DEBUG
    if (x >= w || y >= h) {
        std::cerr << __func__ << ": OOB" << std::endl;
        return;
    }
#endif
    data[y * w + x] = val;
    dataCM[x * h + y] = val;
}

/**
 * Get an element from the column-major array.
 */
template <class E>
E Board2D::elem_get_cm(size_t x, size_t y) {
#ifdef DEBUG
    if (x >= w || y >= h) {
        std::cerr << __func__ << ": OOB" << std::endl;
        return 0;
    }
#endif
    return dataCM[x * h + y];
}

/**
 * Get an element from the row-major array.
 */
template <class E>
E Board2D::elem_get_rm(size_t x, size_t y) {
#ifdef DEBUG
    if (x >= w || y >= h) {
        std::cerr << __func__ << ": OOB" << std::endl;
        return 0;
    }
#endif
    return data[y * w + x];
}