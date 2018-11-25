#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <vector>

// #define PERF
// #define DEBUG
// #define __NVCC__

#ifdef __NVCC__
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#else // __DEBUG__

#define cudaCheckError(ans) ans

#endif // __DEBUG__
#else // __NVCC__
#ifdef __JETBRAINS_IDE__
#include "CLion.h"
#else
#define __host__
#define __device__
#define __global__
#endif

#endif // __NVCC__

#include "Board2DDevice.h"
#include "NonogramLineDevice.h"

#define BUFLEN 255

bool parse_input_file_mk(std::string fInput, NonogramLineDevice **Ls, Board2DDevice **B) {

#ifdef DEBUG
    std::cout << "Parsing puzzle..." << std::endl;
#endif

    std::fstream F;
    F.open(fInput);
    if (!F) {
        std::cout << "Input file error" << std::endl;
        return false;
    }

    std::string line;
    unsigned w, h, n;

    std::getline(F, line);
    std::istringstream iss(line);
    if (!(iss >> h >> w)) {
        return false;
    }

    // Initialize structs
    if (!ng_init(w, h, Ls, B)) {
        return false;
    }

    for (unsigned r = 0; r < h; r++) {
        std::getline(F, line);
        iss.clear();
        iss.str(line);
#ifdef DEBUG
        std::cout << r << "\t";
#endif
        while (iss >> n) {
#ifdef DEBUG
            std::cout << n << "\t";
#endif
            if (!ng_constr_add(*Ls, r, n)) {
                std::cout << "MAX_RUNS exceeded" << std::endl;
                return false;
            }
        }
#ifdef DEBUG
        std::cout << "\n";
#endif
    }

    std::getline(F, line);

    for (unsigned c = 0; c < w; c++) {
        std::getline(F, line);
        iss.clear();
        iss.str(line);
#ifdef DEBUG
        std::cout << c << "\t";
#endif
        while (iss >> n) {
#ifdef DEBUG
            std::cout << n << "\t";
#endif
            if (!ng_constr_add(*Ls, h + c, n)) {
                std::cout << "MAX_RUNS exceeded" << std::endl;
                return false;
            }
        }
#ifdef DEBUG
        std::cout << "\n";
#endif
    }

    F.close();
    return true;
}

int main(int argc, char **argv) {

#ifdef DEBUG
    std::cout << "DEBUG ";
#endif

#ifdef PERF
    std::cout << "PERF ";
#endif

#ifdef __NVCC__
    std::cout << "__NVCC__ ";
#endif

    std::cout << std::endl;

    int c;
    std::string fInput;
    Board2DDevice *B = nullptr;
    NonogramLineDevice *Ls = nullptr;
    while ((c = getopt(argc, argv, "f:")) != -1) {
        switch (c) {
            case 'f': {
                fInput.assign(optarg);
                if (!parse_input_file_mk(fInput, &Ls, &B)) {
                    return -1;
                }
                break;
            }
        }
    }

    std::cout << "Solving..." << std::endl;
    ng_solve(Ls, B);

    std::cout << "Completed." << std::endl;
    std::cout << B;
    // Cleanup
    ng_free(Ls, B);

    return 0;
}
