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

    return true;
}

NonogramLine::NonogramLine(
        Nonogram &_ngram, unsigned _len, const char *_data,
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
        w_runs = {b_runs[i].botStart - 1, b_runs[i - 1].topEnd + 1};
    }

}

void NonogramLine::fill_all() {

    for (BRun r : b_runs) {
        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngram.cell_confirm(Nonogram::Color::BLACK, line_index, i, line_is_row);
        }
    }

    for (WRun r : w_runs) {
        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            ngram.cell_confirm(Nonogram::Color::WHITE, line_index, i, line_is_row);
        }
    }

}

void NonogramLine::update() {

}