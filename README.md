Benjmain Huang (zemingbh) and Eric Sun (ehsun)

## Summary

This project aims to implement a fast parallel nonogram solver that scales to large puzzles and scales for puzzles with high ambiguity.
We will also consider the design of data structures tha facilitate parallelism for such an application.

## Background

### Nonograms

Nonograms are a logic puzzle built around a rectangular grid divided into cells. The solver's objective is to determine, for each cell, whether the cell is shaded or not. The constraints on the shading of cells are given as number sequences on each row and column, indicating to the solver the number of contiguous regions of shaded cells in that line (row or column) and the number of shaded cells in each contiguous region. See https://en.wikipedia.org/wiki/Nonogram for more details.

Generally, given a single row and column, a solver would be able to determine the result in some of the cells. For a simple example, given such a row on a 16 column board:

``10 2 2 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|``

This indicates that there will be a contiguous region of 10 shaded cells, then 2 shaded cells, then 2 shaded cells.
Since there is at least one unshaded cell between each contiguous region, we can determine the following cells are shaded and not shaded:

``10 2 2 |X|X|X|X|X|X|X|X|X|X| |X|X| |X|X|``

Suppose we have the following row instead:

``10 1 1 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|``

We will not be able to determine the entire row, but only the following cells:

``10 1 1 |.|.|X|X|X|X|X|X|.|.|.|.|.|.|.|.|``

This can be derived since the last two single shaded cells must have two unshaded cells before them, which restricts the first contiguous 10-region to within the first 12 cells. Thus, the 3rd to the 10th cell in the row will definitely be shaded as part of the 10-region. Howveer, we cannot make any other conclusions about the rest of the cells in the row without looking at the rest of the puzzle.

Another example row is given below:

``2 4 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|``

We cannot determine any shading by looking at this row in isolation, since the possible positions of both the 2- and 4-region do not all interesct at any cells. However, if we determine from other parts of the puzzle that the 3rd cell is shaded,

``2 4 |.|.|X|.|.|.|.|.|.|.|.|.|.|.|.|.|``

we can also determine that the first cell is unshaded. This is because if the first cell was shaded our 2-region would touch the 3rd cell and not be valid.

``2 4 | |.|X|.|.|.|.|.|.|.|.|.|.|.|.|.|``

In some cases the entire puzzle will not have any rows and columns that can be solved from the current state, and the solver is required to proceed via an assumption. Since puzzles have a single solution, the solver either arrives at a contradiction in which case the initial assumption is determined to be false, or the solver solves the puzzle (possibly via further assumptions). A state where an assumption is required to proceed is said to be an *ambiguous* state. Generally, the more ambiguous states a puzzle has, the more difficult it is to solve.

## The Challenge

In particular, ambiguous states are difficult for automated solvers because there is very little per-cell heuristic on which assumptions are likely to cause a contradiction or likely to be part of the solution. Ideally, the solver would pick an assumption that will quickly lead to a contradiction to reduce time spent computing states that are not part of the solution. Determining which assumption of all possible assumptions is non-trivial to determine and relies on heuristics.

As such, the challenge will be to find a heuristic that lends itself to parallelism. It is likely that 

## Resources

Using a GPU or a heterogenous configuration is suitable because the problem inherently has many small parts (many cells, many rows, many columns). 

## Goals / Deliverables



## Platform



## Schedule


