# Parallel Nonogram Solver

Benjamin Huang (zemingbh) and Eric Sun (ehsun)

## Summary

This project aims to implement a fast parallel nonogram solver that scales to large puzzles and scales for puzzles with high ambiguity.

We will also consider the design of data structures that facilitate parallelism for such an application.

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

This can be derived since the last two single shaded cells must have two unshaded cells before them, which restricts the first contiguous 10-region to within the first 12 cells. Thus, the 3rd to the 10th cell in the row will definitely be shaded as part of the 10-region. However, we cannot make any other conclusions about the rest of the cells in the row without looking at the rest of the puzzle.

Another example row is given below:

``2 4 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|``

We cannot determine any shading by looking at this row in isolation, since the possible positions of both the 2- and 4-region do not all intersect at any cells. However, if we determine from other parts of the puzzle that the 3rd cell is shaded,

``2 4 |.|.|X|.|.|.|.|.|.|.|.|.|.|.|.|.|``

we can also determine that the first cell is unshaded. This is because if the first cell was shaded our 2-region would touch the 3rd cell and not be valid.

``2 4 | |.|X|.|.|.|.|.|.|.|.|.|.|.|.|.|``

In some cases the entire puzzle will not have any rows and columns that can be solved from the current state, and the solver is required to proceed via an assumption. Since puzzles have a single solution, the solver either arrives at a contradiction in which case the initial assumption is determined to be false, or the solver solves the puzzle (possibly via further assumptions). A state where an assumption is required to proceed is said to be an *ambiguous* state. Generally, the more ambiguous states a puzzle has, the more difficult it is to solve.

## The Challenge

Nonograms are interesting because they are not a particularly mainstream type of logic puzzle, which means existing research is scarcer,  and are not obviously reducible to other problems. There are also much fewer existing implementations of solvers.

With regard to parallelism, it is likely that three areas of the problem will be parallelized: **simple solving**, **lookahead solving**, and **heuristics**.

### Simple solving

Simple solving is a set of strategies similar to the ones described in the background above, where certain cells in a row or column can be determined to be filled or not based solely on the rest of that row or column. For these strategies we can parallelize by row or column, and use shared memory.

Note that if using shared memory, we cannot operate on rows and columns simultaneously. We either have to introduce some synchronization between solving rows and then solving columns, or use one block of shared memory for each and do some reconciliation at the end of each iteration.

Some puzzles can be solved by doing iterations of simple solving techniques until each row and column satisfies the constraints. It will be most efficient to use simple solving techniques in the first phase of our parallel implementation because they can greatly reduce the search space of each puzzle.

### Lookahead solving

Lookahead solving is a strategy that involves making an assumption in the current state, and exploring the tree of future states that result from that assumption. The goal is to prune that tree by finding states with contradictions. Every time an assumption is made, simple solving techniques will be used to either find a contradiction in the resulting state and invalidate the assumption (or in other words, validate the opposite), or get to a new state where more assumptions must be made.

Thus we have two axes of parallelism: simple solving within each branch and testing different assumptions in parallel. Since exploring different assumptions will likely lead to contradictions in drastically varying amounts of time, some sort of dynamic scheduling will be required.

It is also likely that different branches of the tree merge at future times since different assumptions can lead to the same state, and efficient communication between threads will be required to determine if and when this happens in order to avoid redundant work. In a sequential solver, dynamic programming could be used to find common subproblems, and it's likely a similar approach could work here.

### Heuristics

In particular, ambiguous states are difficult for solvers because there is very little per-cell heuristic on which assumptions are likely to cause a contradiction or likely to be part of the solution. Ideally, the solver would pick an assumption that will quickly lead to a contradiction to reduce time spent computing states that are not part of the solution. Determining which assumption of all possible assumptions is non-trivial to determine and relies on heuristics.

As such, the challenge will be to find a heuristic that lends itself well to parallelism. Here, we distinguish heuristics from lookahead solving by defining heuristics as computation involving only the current board state and no future board states.

It is likely that heuristics will involve data structures other than a boolean array representation of the board, and these data structures will also have to be optimized for parallel computation.

## Resources

There have been two past projects on Nonograms and a number of past projects on other logic puzzles (mainly Sudokus).
https://github.com/seansxiao/nonogram-solver (S17, using CPU, only naive heuristics)
http://www.andrew.cmu.edu/user/marjorie/parallelpbn/parallelpbn.html (S15, OpenMP, poor parallel speedup)


## Goals / Deliverables

### Implementations

#### 1. Serial Implementation
- Able to solve puzzles correctly, implementing simple solving techniques and lookahead solving.
- Does not necessarily implement any heuristics for lookahead solving.
- Does not scale well with puzzle size.
- Does not scale well with puzzle ambiguity.

#### 2. Parallel Implementation
- Able to solve puzzles correctly, implementing simple solving techniques parallelized across rows and columns, and lookahead solving parallelized across assumptions.
- Does not necessarily implement any heuristics for lookahead solving.
- Scales well with puzzle size.
- Scales somewhat well with puzzle ambiguity.

#### 3. Good Parallel Implementation
- Able to solve puzzles correctly, implementing simple solving techniques parallelized across rows and columns, and lookahead solving parallelized across assumptions.
- Implements heuristics for lookahead solving.
- Scales well with puzzle size.
- Scales well with puzzle ambiguity.

### Other Deliverables

#### Demo
Animations of puzzles being solved and the steps taken by the solver for various puzzles.

#### Speedup graphs

Show speedup graphs for various puzzle sizes/ambiguities.

### Stretch goals
- Extend to Nonogram variants (additional colors, unordered constraints).
- Extend to other logic puzzles (Slitherlink, etc.)

## Platform

### Machine

We believe GPUs are a good platform for this solver for a number of reasons. One is that, excluding heuristics (which will require us to do more research), the majority of the workload is using simple solving techniques to either find that a certain state has a contradiction somewhere down the line, or that it can be brought to a different state that requires assumptions to be made.

All of the simple solving techniques can be run in parallel by row and column (as described above). Because the same methods will be employed across each row and column, there is the potential for SIMD parallelism to be useful. Additionally, puzzles are small and can take advantage of thread block shared memory, which in GPUs is significantly faster than global memory. Finally, the small size of each puzzle means that they can be copied in and out of memory extremely quickly.

Finally, it may be useful to try writing our solver to run on a processor like a Xeon Phi which has a large number of independent processors but better performance with divergent execution.

### Languages
- C++: Fast, with good library support.
- CUDA: for GPU utilization.

## Schedule

12 Nov: Working solver

19 Nov: Working parallel solver with significant speedups on lookahead solving

26 Nov: Derive heuristics

3 Dec: Implement heuristics

10 Dec: Optimize for parallelism, take benchmarks

15 Dec: Final report due

