//
// Created by Benjamin Huang on 11/12/2018.
//

#include "Nonogram.h"

bool Nonogram::cell_confirm(Color color, unsigned line_index, unsigned i, bool is_row) {

#ifdef DEBUG
    if (is_row)
    if (board.elem_get_rm(c, r) != UNKNOWN && board.elem_get_rm(c, r) != color) {
        std::cerr << __func__ << ": Overwrite" << std::endl;
        return;
    }
    else
    if (board.elem_get_cm(c, r) != UNKNOWN && board.elem_get_cm(c, r) != color) {
        std::cerr << __func__ << ": Overwrite" << std::endl;
        return;
    }
#endif
    if (is_row)
        board.elem_set(i, line_index, color);
    else
        board.elem_set(line_index, i, color);

    dirty = true;

    return true;
}

void Nonogram::solve() {

    line_init();

    do {

        dirty = false;

        for (unsigned i = 0; i < row_solvers.size(); i++) {
            row_solvers[i].update();
            row_solvers[i].fill_add();
        }

        for (unsigned i = 0; i < col_solvers.size(); i++) {
            col_solvers[i].update();
            col_solvers[i].fill_add();
        }

    } while (dirty);

}

void Nonogram::line_init() {

    for (unsigned i = 0; i < h(); i++) {
        row_solvers.emplace_back(NonogramLine(this, w(), board.row_ptr_get(i), i, true, row_constr[i]));
    }

    for (unsigned i = 0; i < w(); i++) {
        col_solvers.emplace_back(NonogramLine(this, h(), board.col_ptr_get(i), i, false, col_constr[i]));
    }

}

std::ostream &operator<<(std::ostream &os, Nonogram &N) {

    for (unsigned r = 0; r < N.h(); r++) {
        for (unsigned c = 0; c < N.w(); c++) {
            if (N.board.elem_get_rm(c, r) == Nonogram::Color::BLACK) {
                os << '#';
            }
            else {
                os << ' ';
            }
        }
        os << std::endl;
    }
    return os;

}

NonogramLine::NonogramLine(
        Nonogram *_ngram, unsigned _len, const Nonogram::Color *_data,
        unsigned _index, bool _is_row, const std::vector<unsigned> &constr)
        : ngram(_ngram),
          len(_len),
          line_index(_index),
          line_is_row(_is_row),
          data(_data) {

    b_runs = std::vector<BRun>(constr.size());
    w_runs = std::vector<WRun>(constr.size() + 1);

    if (constr.empty()) {
        w_runs[0] = {0, len};
        return;
    }

    // Find the topmost line configuration for each run.

    unsigned topSum = 0;
    for (unsigned i = 0; i < constr.size(); i++) {
        b_runs[i].len = constr[i];
        topSum += constr[i];
        b_runs[i].topEnd = topSum;
        topSum++;
    }

    w_runs.back() = {len, topSum - 1};

    // Find the bottommost line configuration for each run.

    unsigned botSum = len;
    for (unsigned i = constr.size() - 1; i < constr.size(); i--) {
        botSum -= constr[i];
        b_runs[i].botStart = botSum;
        botSum--;
    }

    w_runs.front() = {botSum + 1, 0};

    // Fill in parameters for the white runs

    for (unsigned i = 1; i < w_runs.size() - 1; i += 2) {
        w_runs[i] = {b_runs[i].botStart - 1, b_runs[i - 1].topEnd + 1};
    }

}

void NonogramLine::fill_all() {

    for (BRun r : b_runs) {
        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngram->cell_confirm(Nonogram::Color::BLACK, line_index, i, line_is_row);
        }
    }

    for (WRun r : w_runs) {
        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngram->cell_confirm(Nonogram::Color::WHITE, line_index, i, line_is_row);
        }
    }

}

void NonogramLine::fill_add() {

    for (BRun r : b_runs) {
        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            if (data[i] != Nonogram::Color::BLACK)
            ngram->cell_confirm(Nonogram::Color::BLACK, line_index, i, line_is_row);
        }
    }

    for (WRun r : w_runs) {
        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            if (data[i] != Nonogram::Color::WHITE)
            ngram->cell_confirm(Nonogram::Color::WHITE, line_index, i, line_is_row);
        }
    }

}

void NonogramLine::update() {

    // Walk down the line, and fill in the run structures
    unsigned ri0 = 0; // The minimum index black run we could be in
    unsigned ri1 = 0; // The maximum index black run we could be in

    // Walk
    unsigned i = b_runs[0].topEnd - b_runs[0].len;
    for (; i < len; i++) {

        Nonogram::Color color = data[i];

        // No information
        if (color == Nonogram::Color::UNKNOWN) continue;

        while (i >= b_runs[ri0].botStart + b_runs[ri0].len) ri0++;
        while (ri1 + 1 < b_runs.size() && i >= b_runs[ri1 + 1].topEnd - b_runs[ri1 + 1].len) ri1++;

        if (ri0 == b_runs.size()) break;

        // Check if we are in a confirmed shaded region
        if (b_runs[ri0].botStart < b_runs[ri0].topEnd) {
            continue;
        }

        // Check if we are in a confirmed unshaded region
        if (ri0 > ri1) continue;

        // These are not fixed

        // Try to fix
        if (color == Nonogram::Color::BLACK) {
            if (ri0 == ri1) { // Can fix

                botStart_propagate(ri0, i);
                topEnd_propagate(ri0, i + 1);

            }
        }

        else { // if (color == Nonogram::Color::WHITE) {
            if (i >= b_runs[ri0].botStart) {
                botStart_propagate(ri0, i - b_runs[ri0].len);
            }
            if (i <= b_runs[ri1].topEnd) {
                topEnd_propagate(ri1, i);
            }

        }

    }

}

void NonogramLine::botStart_propagate(unsigned ri, unsigned i) {
    while (i < b_runs[ri].botStart) {
        b_runs[ri].botStart = i;
        if (ri == 0) break;
        ri--;
        i -= b_runs[ri].len + 1;
    }
}

void NonogramLine::topEnd_propagate(unsigned ri, unsigned i) {
    while (i > b_runs[ri].topEnd) {
        b_runs[ri].topEnd = i;
        ri++;
        if (ri == b_runs.size()) break;
        i += b_runs[ri].len + 1;
    }
}