#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <vector>
#include "Nonogram.h"

#define BUFLEN 255

Nonogram *parse_input_file_mk(std::string fInput) {

    std::fstream F;

    F.open(fInput);

    std::string line;
    unsigned w, h, n;

    std::getline(F, line);
    std::istringstream iss(line);
    iss >> h >> w;

    // Make a new Nonogram
    Nonogram *puzzle = new Nonogram(w, h);

    for (unsigned r = 0; r < h; r++) {
        std::getline(F, line);
        iss = std::istringstream(line);
        std::vector<unsigned> tmp_constr;
        while (iss >> n) {
            tmp_constr.push_back(n);
        }
        puzzle->row_constr_set(r, std::move(tmp_constr));
    }

    std::getline(F, line);

    for (unsigned c = 0; c < w; c++) {
        std::getline(F, line);
        iss = std::istringstream(line);
        std::vector<unsigned> tmp_constr;
        while (iss >> n) {
            tmp_constr.push_back(n);
        }
        puzzle->col_constr_set(c, std::move(tmp_constr));
    }

    F.close();

    return puzzle;

}

int main(int argc, char **argv) {

    int c;
    std::string fInput;
    Nonogram *P = nullptr;
    while ((c = getopt(argc, argv, "f:")) != -1) {
        switch (c) {
            case 'f': {
                fInput.assign(optarg);
                P = parse_input_file_mk(fInput);
                break;
            }
        }
    }

    if (P == nullptr) return -1;

    // Cleanup
    delete P;

    return 0;
}