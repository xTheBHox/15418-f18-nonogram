//
// Created by Benjamin Huang on 11/12/2018.
//

#include "Nonogram.h"

bool Nonogram::cell_confirm(NonogramColor color, unsigned line_index, unsigned i, bool is_row) {

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

bool Nonogram::cell_check_confirm(NonogramColor color, unsigned line_index, unsigned i, bool is_row) {

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

    if (is_row) {
        if (board.elem_get_rm(i, line_index) != UNKNOWN && board.elem_get_rm(i, line_index) != color) return false;
        board.elem_set(i, line_index, color);
    }
    else {
        if (board.elem_get_cm(line_index, i) != UNKNOWN && board.elem_get_cm(line_index, i) != color) return false;
        board.elem_set(line_index, i, color);
    }

    return true;
}

void Nonogram::solve() {

    line_init();

    do {

        dirty = false;

        for (unsigned i = 0; i < row_solvers.size(); i++) {
            row_solvers[i].update();
            row_solvers[i].runs_fill();
        }

        for (unsigned i = 0; i < col_solvers.size(); i++) {
            col_solvers[i].update();
            col_solvers[i].runs_fill();
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
            char sym = 'X';
            switch (N.board.elem_get_rm(c, r)) {
                case NonogramColor::BLACK: {
                    sym = '#';
                    break;
                }
                case NonogramColor::UNKNOWN: {
                    sym = '?';
                    break;
                }
                case NonogramColor::WHITE: {
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
        Nonogram *_ngram, unsigned _len, const NonogramColor *_data,
        unsigned _index, bool _is_row, const std::vector<unsigned> &constr)
        : ngram(_ngram),
          len(_len),
          line_is_row(_is_row),
          line_index(_index),
          data(_data) {

    b_runs = std::vector<BRun>(constr.size());

    if (constr.empty()) {
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

    // Find the bottommost line configuration for each run.

    unsigned botSum = len;
    for (unsigned i = constr.size() - 1; i < constr.size(); i--) {
        botSum -= constr[i];
        b_runs[i].botStart = botSum;
        botSum--;
    }

}

void NonogramLine::runs_fill() {

    unsigned prev_wrun_botStart = 0;

    for (BRun r : b_runs) {

        while (r.topEnd - r.len > prev_wrun_botStart) {
            cell_solve(NonogramColor::WHITE, prev_wrun_botStart);
            prev_wrun_botStart++;
        }

        for (unsigned i = r.botStart; i < r.topEnd; i++) {
            cell_solve(NonogramColor::BLACK, i);
        }

        prev_wrun_botStart = r.botStart + r.len;

    }

    while (prev_wrun_botStart < len) {
        cell_solve(NonogramColor::WHITE, prev_wrun_botStart);
        prev_wrun_botStart++;
    }

}

void NonogramLine::update() {

    if (b_runs.empty()) return;

    // Walk down the line, and fill in the run structures
    unsigned ri0 = 0; // The minimum index black run we could be in
    unsigned ri1 = 0; // The maximum index black run we could be in

    unsigned curr_bblock_len = 0;
    // unsigned first_nwi = 0;

    // Walk
    unsigned i = b_runs[0].topEnd - b_runs[0].len;
    for (; i < len; i++) {

        NonogramColor color = data[i];
        if (color == NonogramColor::BLACK) {
            curr_bblock_len++;
        }
        else if (color == NonogramColor::WHITE) {
            curr_bblock_len = 0;
            // first_nwi = i + 1;
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
        if (color == NonogramColor::BLACK) {
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
        cell_solve(NonogramColor::WHITE, i + 1);
    }
    if (i >= curr_bblock_len) {
        cell_solve(NonogramColor::WHITE, i - curr_bblock_len);
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

void NonogramLine::cell_solve(NonogramColor color, unsigned i) {
    if (data[i] != color) {
        ngram->cell_confirm(color, line_index, i, line_is_row);
    }
}
