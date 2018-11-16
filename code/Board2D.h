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
    Board2D(size_t _w, size_t _h) : w(_w), h(_h) {
        data = new E[2 * w * h];
        dataCM = &data[w * h];
    }

    ~Board2D() {
        delete data;
    }

    void elem_set(size_t x, size_t y, E val);
    E elem_get_rm(size_t x, size_t y);
    E elem_get_cm(size_t x, size_t y);

private:

    // Board dimensions.
    const size_t w;
    const size_t h;

    // Pointer to the row-major array (and the combination of both arrays).
    E *data;

    // Pointer to the column-major array.
    E *dataCM;

};

#endif //CODE_BOARD2D_H
