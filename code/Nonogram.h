//
// Created by Benjamin Huang on 11/12/2018.
//

#ifndef CODE_NONOGRAM_H
#define CODE_NONOGRAM_H

#include <vector>
#include "Board2D.h"

class Nonogram;
class NonogramLine;

class Nonogram {
public:
    enum Color : char {
        UNKNOWN = 0,
        WHITE = -1,
        BLACK = 1
    };

    Nonogram(unsigned w, unsigned h) : board(w, h, UNKNOWN), row_constr(h), col_constr(w) { }

    /**
     * Sets a row constraint in the nonogram.
     * @param row the row index
     * @param constr the constraint vector rvalue
     * @return true if successful, false otherwise
     */
    bool row_constr_set(unsigned row, std::vector<unsigned> &&constr) {
        row_constr[row] = constr;
        return true;
    }

    /**
     * Sets a column constraint in the nonogram.
     * @param col the column index
     * @param constr the constraint vector rvalue
     * @return true if successful, false otherwise
     */
    bool col_constr_set(unsigned col, std::vector<unsigned> &&constr) {
        col_constr[col] = constr;
        return true;
    }

    /**
     * Confirms an unknown cell in the board to color.
     * @param color the correct color
     * @param line_index the line index, from NonogramLine
     * @param i the index in the line, from NonogramLine
     * @param is_row whether this is a row or column, from NonogramLine
     * @return true if successful, false otherwise
     */
    bool cell_confirm(Color color, unsigned line_index, unsigned i, bool is_row);

    unsigned w() const { return board.w; }
    unsigned h() const { return board.h; }


    void solve();
    void line_init();

    friend std::ostream &operator<<(std::ostream &os, Nonogram &N);

private:
    Board2D<Color> board;
    std::vector<std::vector<unsigned>> row_constr;
    std::vector<std::vector<unsigned>> col_constr;

    // solver params
    bool dirty;
    std::vector<NonogramLine> row_solvers;
    std::vector<NonogramLine> col_solvers;

};

/**
 * NonogramLine is a solver for a single nonogram line (row/col).
 */
class NonogramLine {
public:

    /**
     * BRun represents one contiguous run of shaded cells.
     */
    typedef struct {
        unsigned topEnd;
        unsigned botStart;
        unsigned len;
    } BRun;

    /**
     * WRun represents one contiguous run of unshaded cells.
     */
    typedef struct {
        unsigned botStart;
        unsigned topEnd;
    } WRun;

    NonogramLine(
            Nonogram *_ngram, unsigned _len, const Nonogram::Color *_data,
            unsigned _index, bool _is_row, const std::vector<unsigned> &constr);

    /**
     * Fills the straightforward cells into the board.
     */
    void fill_all();

    /**
     * Fills the straightforward cells into the board.
     */
    void fill_add();

    /**
     * Updates the run structs by referencing the current state of the board.
     */
    void update();
    void botStart_propagate(unsigned ri, unsigned i);
    void topEnd_propagate(unsigned ri, unsigned i);

private:
    Nonogram *ngram;
    const unsigned len;
    const bool line_is_row;
    const unsigned line_index;
    const Nonogram::Color *const data;
    std::vector<BRun> b_runs;
    std::vector<WRun> w_runs;
};


#endif //CODE_NONOGRAM_H
