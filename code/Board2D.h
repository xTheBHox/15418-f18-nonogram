//
// Created by Benjamin Huang on 11/12/2018.
//

#ifndef CODE_BOARD2D_H
#define CODE_BOARD2D_H

#include <cstdlib>
#include <iostream>

template<class E>
class Board2D {
public:
    Board2D(unsigned _w, unsigned _h, E val) : w(_w), h(_h) {
        data = new E[2 * w * h];
        dataCM = &data[w * h];
        for (unsigned i = 0; i < 2 * w * h; i++) {
            data[i] = val;
        }
    }

    ~Board2D() {
        delete data;
    }

    void elem_set(unsigned x, unsigned y, E val);
    E elem_get_rm(unsigned x, unsigned y) const;
    E elem_get_cm(unsigned x, unsigned y) const;

    const E* row_ptr_get(unsigned index) const;
    const E* col_ptr_get(unsigned index) const;

    friend void *board2d_to_device(Board2D B_host);
    friend void board2d_dev_to_host(void* B_dev_v, Board2D B_host);

    // Board dimensions.
    const unsigned w;
    const unsigned h;

private:


    // Pointer to the row-major array (and the combination of both arrays).
    E *data;

    // Pointer to the column-major array.
    E *dataCM;

};


template <class E>
void Board2D<E>::elem_set(unsigned x, unsigned y, E val) {
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
E Board2D<E>::elem_get_cm(unsigned x, unsigned y) const {
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
E Board2D<E>::elem_get_rm(unsigned x, unsigned y) const {
#ifdef DEBUG
    if (x >= w || y >= h) {
        std::cerr << __func__ << ": OOB" << std::endl;
        return 0;
    }
#endif
    return data[y * w + x];
}

template <class E>
const E* Board2D<E>::row_ptr_get(unsigned index) const {
#ifdef DEBUG
    if (index >= h) {
        std::cerr << __func__ << ": OOB" << std::endl;
        return 0;
    }
#endif
    return &data[index * w];
}

template <class E>
const E* Board2D<E>::col_ptr_get(unsigned index) const {
#ifdef DEBUG
    if (index >= w) {
        std::cerr << __func__ << ": OOB" << std::endl;
        return 0;
    }
#endif
    return &dataCM[index * h];
}

#endif //CODE_BOARD2D_H
