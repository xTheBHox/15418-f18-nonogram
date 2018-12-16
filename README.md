# Parallel Nonogram Solver

Benjamin Huang (zemingbh) and Eric Sun (ehsun)

Visit the website at [https://xthebhox.github.io/15418-f18-nonogram], or view report.pdf.

A sequential and parallel nonogram solver (in C++11). To compile, use the provided Makefile. The parallel version requires CUDA and the display (if you compile with it) uses ncurses.

Inputs should be speficied with -f, and be in the .mk format (refer to webpbn.com's puzzle export). Essentially it is <#rows> <#cols> on the first line, all the row constraints, line separated, then a # character on its own line, then all the column constraints. See the inputs folder for examples.
