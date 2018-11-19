//
// Created by Benjamin Huang on 11/12/2018.
//

#ifndef CODE_NONOGRAM_H
#define CODE_NONOGRAM_H

#include <vector>
#include "Board2D.h"

// Uncomment this to enforce checks.
// #define DEBUG

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
    bool cell_check_confirm(Color color, unsigned line_index, unsigned i, bool is_row);

    unsigned w() const { return board.w; }
    unsigned h() const { return board.h; }

    void solve();
    friend void ng_solve(Nonogram *N);

    /**
     * Initializes the solver for each line.
     */
    void line_init();

    friend std::ostream &operator<<(std::ostream &os, Nonogram &N);

    // Solver parameters

    // This indicates the board has been modified.
    bool dirty;

    Board2D<Color> board;
    std::vector<std::vector<unsigned>> row_constr;
    std::vector<std::vector<unsigned>> col_constr;

private:
    // Solver parameters

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
     * Updates the line based on any changes since the last run.
     */
    void update();

    /**
    * Fills the cells which the runs have confirmed.
    */
    void runs_fill();

    /**
     * Fills in white cells around a max size block.
     * @param i the ending index of the block
     * @param curr_bblock_len the length of the block
     */
    void block_max_size_fill(unsigned i, unsigned curr_bblock_len);

    /**
     * Propagates a change to the bottommost position of a run to the rest of the runs.
     * @param ri the run index of the run to be changed.
     * @param i the position to change to.
     */
    void botStart_propagate(unsigned ri, unsigned i);

    /**
     * Propagates a change to the topmost position of a run to the rest of the runs.
     * @param ri the run index of the run to be changed.
     * @param i the position to change to.
     */
    void topEnd_propagate(unsigned ri, unsigned i);

    /**
     * Sets a cell, making sure not to dirty the board if no changes are made.
     * @param color the color to set
     * @param i the cell position to set
     */
    void cell_solve(Nonogram::Color color, unsigned i);

private:
    Nonogram *ngram;
    const unsigned len;
    const bool line_is_row;
    const unsigned line_index;
    const Nonogram::Color *const data;
    std::vector<BRun> b_runs;
};


#endif //CODE_NONOGRAM_H
