Benjmain Huang (zemingbh) and Eric Sun (ehsun)

## Summary

This project aims to implement a fast parallel nonogram solver that scales to large puzzles and scales for puzzles with high ambiguity.
We will also consider the design of data structures tha facilitate parallelism for such an application.

## Background

### Nonograms

Nonograms are a logic puzzle built around a rectangular grid divided into cells. The solver's objective is to determine, for each cell, whether the cell is shaded or not. The constraints on the shading of cells are given as number sequences on each row and column, indicating to the solver the number of contiguous regions of shaded cells in that line (row or column) and the number of shaded cells in each contiguous region. See https://en.wikipedia.org/wiki/Nonogram for more details.

Generally, given a single row and column, a solver would be able to determine the result in some of the cells. For example, given such a row on a 16 column board:

``10 2 2 |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|``

This indicates that there will be a contiguous region of 10 shaded cells, then 2 shaded cells, then 2 shaded cells.
Looking at the row in isolation, we can determine the following cells are shaded:

``10 2 2 |X|X|X|X|X|X|X|X|X|X|.|X|X|.|X|X|``

## The Challenge



## Resources

Using a GPU or a heterogenous configuration is suitable because the problem inherently has many small parts (many cells, many rows, many columns). 

## Goals / Deliverables



## Platform



## Schedule


