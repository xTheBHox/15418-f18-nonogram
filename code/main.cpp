#include <iostream>
#include <fstream>
#include <unistd.h>

int parse_input_file(char *fInput) {

    std::fstream F;
    F.open(fInput);


    F.close();

}

int main(int argc, char **argv) {

    int c;
    char* fInput;
    while ((c = getopt(argc, argv, "f:")) != -1) {
        switch (c) {
            case 'f': {
                fInput = optarg;
                break;
            }
        }
    }



    return 0;
}