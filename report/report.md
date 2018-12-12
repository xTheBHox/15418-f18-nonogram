# Summary
We implemented a GPU-accelerated parallel nonogram solver that can find solutions for puzzles that require lookahead solving. We found that a sequential implementation is faster for the largest puzzles that humans typically solve (99x99), but that for larger puzzles a parallel implementation has better performance.

<!--
#TODOs:
 - Describe actual performance results here.
-->

# Background
Nonograms are logic puzzles built around a rectangular grid divided into cells. The solver's objective is to determine, for each cell, whether the cell is shaded or not. The constraints on the shading of cells are given as number sequences on each row and column, indicating to the solver the number of contiguous regions of shaded cells in that line (row or column) and the number of shaded cells in each contiguous region. An example is shown below:

![Example nonogram.](./w_nonogram.png)



## Data structures
The state of the cells of a nonogram can be represented by a matrix or board whose elements can take one of three different values: white (unfilled), black (filled), or unknown. For each row and column there is an associated set of constraints, which can be represented as a list of numbers.

Note that while the goal is a board with no unknowns that satisfies the associated constraints, it is possible that a particular board will contradict its constraints or be in such a state that a contradiction is implied in the future. This will be explained in more detail below.

## Operations
Solving nonograms is an iterative process. Any given cell being determined means that other cells in its row and column can also possibly be determined with the new information. There are two basic kinds of operations involved in solving nonograms: simple solving, and lookahead solving. 

### Simple solving
One key operation involved in solving nonograms is advancing the state of any particular row or column, which will just be referred to as a line. Given a line and its list of constraints, a simple solve operation will (potentially) be able to change at least one unknown cell to either filled or unfilled. There are a variety of algorithms that can accomplish this, the simplest of which is illustrated below:

![Simple solving.](./simple_solving.png)

In this example, consider a single line with runs of length four and three. By examining the both the leftmost and rightmost arrangements of the runs, it is apparent that in all cases the cells marked in blue will be filled. In this case, it is not also uaranteed that the cells in white are unfilled; those cells will still be unknown. However, if the first column of the puzzle is solved and it is determined that the first cell of the row is filled, then there is enough information to completely solve the row.

Note that many nonograms can be solved by iterated simple solving. By repeatedly alternating between applying simple solvers to the rows, and to the columns, more and more cells will be determined until eventually the entire puzzle is solved.

This iterated simple solving method can be thought of as a single step in which a board is reduced to its most solved form (fewest unknown cells).

### Lookahead solving

For some puzzles, applying simple solving alone is not enough to find a solution. It is possible to get to a state where no additional cells can be determined from the known information. In this case, it is necessary to make a guess about an unknown cell. The process is relatively simple: a guess is made, and then the board is solved with the guess assumed to be true. There are then three possibilities:

1) A valid solution is found, in which case the guess was correct.
2) A contradiction is found, meaning the guess was incorrect.
3) The board is again unable to advance, meaning another guess must be made.

It is clear that lookahead solving is a traditional backtracking algorithm. As with any backtracking problem, the cell and value of the guess is irrelevant to eventually finding a solution—eventually a solution will be found. However, to maximize performance various heuristics can be used to make a guess that is either likely to be correct, or likely to arrive at a contraction quickly, or both.

Note that lookahead solving does affect the correctness of a board while simple solving does not. We consider a board to be correct if there exists some way to fill in all of its unknown cells such that the constraints are satisfied. It is possible for a board to be incorrect and unsolved: this is a board that is eventually guaranteed to have a contradiction after it is solved more. However a board can only become incorrect through guessing. Solving a puzzle starts with the board full of unknowns which is trivially correct.

## Inputs and outputs

The input to the algorithm is the set of constraints. The board starts in an entirely unknown state.

The output of the algorithm is a board in which every cell is either filled or unfilled, that also satisfies all of the constraints given in the input

## Computational expense
Typically nonograms are meant to be solved by humans. This means that they tend to have relatively low problem sizes. There are two main parameters involved in determining the total size of the problem: number of constraints per line, and size in rows/columns.

Typically puzzles will have on the order of 10 or fewer constraints per line. Because many nonograms form basic images, lines with fewer constraints are more frequent. Puzzles will also be relatively small in grid size. The largest ones are on the order of 100x100, meaning there are only 200 total lines.

This tendency to have small puzzles means that any results need to be contextualized. When solving a puzzle meant for humans it's likely that the overhead of any parallelism will dominate.

### Simple solving
Simple solving techniques are relatively expensive individually. They involve iteratively working through the possibilities for a given line until unknown cells can be filled in. The amount of work involved in this process increases as the number of runs in a given line increases, or as the length of the line increases.

In total the simple solving techniques are also expensive because they must be repeatedly applied to every row of the board, as well as every column. As the board grows the expense of running all the solvers increases.

### Lookahead solving
The process of lookahead solving outside of simple solving is not computationally expensive. If heuristics are used they can be be more expensive depending on which algorithm is chosen.

## Workload
The workload for solving a puzzle consists of iterated simple solving and lookahead solving. Within simple solving there are dependencies between the row steps and column steps.

### Parallelism
Simple solving techniques are applied to a single line (row or column) at a time. They read only the constraints for that line, as well as the contents of the cells in the line, and they write only to cells in the line. Thus simple solving on either every row or every cell can be done in parallel both computationally, and data-wise. 

Because the results from solving the rows are necessary for solving the columns and vice versa solving rows and columns in parallel together is impractical.

### Locality
There is spatial locality as the simple solver will need to iterate over the lines. When the lines are rows this allows us to exploit cache locality, and storing the board in a column major format can allow us to potentially use the same locality for the columns.

### SIMD
Depending on how simple solvers are implemented it is possible that SIMD execution could speed up the simple solvers. In particular, a solution that is largely branchless would be able to take advantage of SIMD. This technique would also likely lend itself well to GPU/CUDA execution because of both the SIMD possibilities and the fact that each computation requires very little data, reducing communication overhead.


