# Checkpoint Report
Benjamin Huang (zemingbh) and Eric Sun (ehsun)

## Current progress
Right now we're wrapping up work on the sequential solver. As described in the original project writeup, sequential solving can be roughly divided into two stages: simple solving and lookahead solving. We're currently nearing the end of implementing of simple solving strategies, and coming to the implementation of lookahead solving. Lookahead solving is by far the easier of the two; it boils down to a breadth or depth-first search of the tree of board possibilities. At the moment our solver is capable of solving puzzles that don't involve any guesswork.

Once we implement lookahead solving we can begin adding parallelism. In this case, the key to a good parallel implementation will be writing good simple solvers and designing data structures that allow for efficient access. Actually running the code on a GPU will largely involve turning the individual line solvers into kernels and moving the board data into shared memory.

## Revised goals
In our initial proposal we outlined three levels of implementation: serial, parallel, and good parallel. Given our current progress we believe we will still get through at least the parallel implementation, and hopefully be able to implement some basic heuristics. We also still believe we will have the demo and speedup graphs.

It is still possible but unlikely that we will be able to implement our stretch goals for other Nonogram variants, and additional types of puzzles.

Our revised poster session goal is a working parallel solver that implements at least one heuristic for lookahead solving. It should be able to read puzzles from disk, solve them, and write the answers back. Ideally we could also add some nice visualization as well.

## Poster session details
There are three deliverables we'd like to have for the poster session. One is the working solver mentioned above. Additionally, we want to have:

1. Speedup graphs for puzzles of various sizes/ambiguities
2. Either a video or live demo of the solver working on a puzzle

## Preliminary results
Our solver is currently able to solve puzzles of easy and medium difficulty that do not involve any guessing.

## Issues
Right now we believing getting to our deliverables is just a matter of writing the code and doing the work. Our primary concern is finishing our set of high quality simple solvers. As mentioned above, actually making the code run in parallel will likely not be a particularly difficult task.

## Revised calendar
- 11/19: Working simple solver
- 11/26: Working sequential solver with finished simple solvers and lookahead
- 11/30: Working parallel solver with finished simple solvers and lookahead
- 12/3: Working basic heuristics
- 12/10: Optimized solver, benchmarks, working demo