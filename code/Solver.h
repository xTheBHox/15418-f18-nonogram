//
// Created by Benjamin Huang on 12/10/2018.
//

#ifndef CODE_SOLVER_H
#define CODE_SOLVER_H

#include "Board2DDevice.h"
#include "NonogramLineDevice.h"
#include "HypotheticalBoard.h"

bool ng_init(unsigned w, unsigned h, NonogramLineDevice **Ls, Board2DDevice **B);
void ng_free(NonogramLineDevice *Ls, Board2DDevice *B);
bool ng_constr_add(NonogramLineDevice *Ls, unsigned line_index, unsigned constr);
void ng_solve(NonogramLineDevice *Ls_host, Board2DDevice *B_host);

#endif //CODE_SOLVER_H
