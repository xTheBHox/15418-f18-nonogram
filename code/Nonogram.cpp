//
// Created by Benjamin Huang on 11/12/2018.
//

#include "Nonogram.h"

bool Nonogram::cell_confirm(Color color, unsigned line_index, unsigned i, bool is_row) {

#ifdef DEBUG
    if (is_row) {
        if (board.elem_get_rm(i, line_index) != UNKNOWN && board.elem_get_rm(i, line_index) != color) {
            std::cerr << __func__ << ": Overwrite" << std::endl;
            return false;
        }
    }
    else {
        if (board.elem_get_cm(line_index, i) != UNKNOWN && board.elem_get_cm(line_index, i) != color) {
            std::cerr << __func__ << ": Overwrite" << std::endl;
            return false;
        }
    }
#endif
    dirty = true;

    if (is_row)
        board.elem_set(i, line_index, color);
    else
        board.elem_set(line_index, i, color);

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
            char sym;
            switch (N.board.elem_get_rm(c, r)) {
                case Nonogram::Color::BLACK: {
                    sym = '#';
                    break;
                }
                case Nonogram::Color::UNKNOWN: {
                    sym = '?';
                    break;
                }
                case Nonogram::Color::WHITE: {
                    sym = ' ';
                    break;
                }
            }
            os << sym;
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

    unsigned prev_wrun_botStart = 0;

    for (BRun r : b_runs) {

        while (r.topEnd - r.len > prev_wrun_botStart) {
            cell_solve(Nonogram::Color::WHITE, prev_wrun_botStart);
            prev_wrun_botStart++;
        }

        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            cell_solve(Nonogram::Color::BLACK, i);
        }

        prev_wrun_botStart = r.botStart + r.len;

    }

    while (prev_wrun_botStart < len) {
        cell_solve(Nonogram::Color::WHITE, prev_wrun_botStart);
        prev_wrun_botStart++;
    }

}

void NonogramLine::runs_update() {

    // Walk down the line, and fill in the run structures
    unsigned ri0 = 0; // The minimum index black run we could be in
    unsigned ri1 = 0; // The maximum index black run we could be in

    // Walk
    unsigned i = b_runs[0].topEnd - b_runs[0].len;
    for (; i < len; i++) {

        Nonogram::Color color = data[i];

        // No information
        if (color == Nonogram::Color::UNKNOWN) {
            continue;
        }

        while (i >= b_runs[ri0].botStart + b_runs[ri0].len) ri0++;
        while (ri1 + 1 < b_runs.size() && i >= b_runs[ri1 + 1].topEnd - b_runs[ri1 + 1].len) ri1++;

        if (ri0 == b_runs.size()) {
            // We have finished all the shaded regions
            break;
        }

        // Check if we are in an already confirmed shaded region
        if (b_runs[ri0].botStart < b_runs[ri0].topEnd) {
            continue;
        }

        // Check if we are in an already confirmed unshaded region
        if (ri0 > ri1) continue;

        // If we get here, we have a determined cell that has not been assigned to a run.

        // Try to assign to a run.
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

void NonogramLine::update() {

    // Walk down the line, and fill in the run structures
    unsigned ri0 = 0; // The minimum index black run we could be in
    unsigned ri1 = 0; // The maximum index black run we could be in

    unsigned curr_bblock_len = 0;
    unsigned first_nwi = 0;

    // Walk
    unsigned i = b_runs[0].topEnd - b_runs[0].len;
    for (; i < len; i++) {


        Nonogram::Color color = data[i];
        if (color == Nonogram::Color::BLACK) {
            curr_bblock_len++;
        }
        else if (color == Nonogram::Color::WHITE) {
            curr_bblock_len = 0;
            first_nwi = i + 1;
        }
        else {
            curr_bblock_len = 0;
            continue;
        }

        while (i >= b_runs[ri0].botStart + b_runs[ri0].len) ri0++;

        unsigned max_run_len = 0;
        for (unsigned j = ri0; j <= ri1; j++) {
            max_run_len = std::max(b_runs[j].len, max_run_len);
        }
        while (ri1 + 1 < b_runs.size() && i >= b_runs[ri1 + 1].topEnd - b_runs[ri1 + 1].len) {
            ri1++;
            max_run_len = std::max(b_runs[ri1].len, max_run_len);
        }

        if (ri0 == b_runs.size()) {
            // We have finished all the shaded regions
            break;
        }

        // Check if we are in an already confirmed shaded region
        if (b_runs[ri0].botStart <= i && i < b_runs[ri0].topEnd) {
            continue;
        }

        // Check if we are in an already confirmed unshaded region
        if (ri0 > ri1) continue;

        // If we get here, we have a determined cell that has not been assigned to a run.

        // Try to assign to a run.
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
            if (i < b_runs[ri1].topEnd) {
                topEnd_propagate(ri1, i + b_runs[ri1].len + 1);
            }
        }

        // We are looking for shaded blocks that have not been assigned to a run.
        if (curr_bblock_len == max_run_len) {
            block_max_size_fill(i, curr_bblock_len);
        }

    }
}

void NonogramLine::block_max_size_fill(unsigned i, unsigned curr_bblock_len) {
    if (i + 1 < len) {
        cell_solve(Nonogram::Color::WHITE, i + 1);
    }
    if (i >= curr_bblock_len) {
        cell_solve(Nonogram::Color::WHITE, i - curr_bblock_len);
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

void NonogramLine::cell_solve(Nonogram::Color color, unsigned i) {
    if (data[i] != color) {
        ngram->cell_confirm(color, line_index, i, line_is_row);
    }
}