# http://dingo.sbs.arizona.edu/~sandiway/sudoku/examples.html
"""
Sudoku Brute Force Solver (DFS)
- Scott Mitchell, Boaz Cogan
"""

import sys

def read_puzzle(filename):
    """ Read puzzle in from a file """
    with open(filename) as f:
        puzzle = [l.split() for l in f.readlines()]
    return puzzle

def print_puzzle(puzzle):
    """ Pretty print puzzle """
    for i in range(9):
        for j in range(9):
            if j%3 == 0 and j != 0:
                print("    ", end="")
            print(puzzle[i][j], end="  ")
        if i%3 == 2:
            print()
        print()

def is_valid_row(puzzle, row, num):
    """ Check for valid row """
    for i in range(9):
        if puzzle[row][i] == num:
            return False
    return True


def is_valid_col(puzzle, col, num):
    """ Check for valid column """
    for i in range(9):
        if puzzle[i][col] == num:
            return False
    return True

def is_valid_square(puzzle, col, row, num):
    """ Check for valid sub square """
    for i in range(3):
        for j in range(3):
            if puzzle[row+i][col+j] == num:
                return False
    return True


def is_valid(puzzle, col, row, num):
    """ Check if the current number is in a legal position """
    return is_valid_col(puzzle, col, num) and is_valid_row(puzzle, row, num) and is_valid_square(puzzle, col - col % 3, row - row % 3, num)


def backtrace(puzzle):
    """
    The solving algorithm for the sudoku puzzle.
    Algorithm:
    1. Find blank
    2. Try any possible number
    3. If no numbers work, go back and try again (DFS)
    4. Repeat
    :param puzzle: The puzzle instance
    :return: If puzzle is valid or not
    """
    if filled(puzzle):
        return True

    row,col = find_blank(puzzle)

    for i in range(1,10):
        i = str(i)
        if is_valid(puzzle, col, row, i):
            puzzle[row][col] = i
            if backtrace(puzzle):
                return True
            puzzle[row][col] = '0'
    return False


def filled(puzzle):
    """ Check if the puzzle has no blanks / is solved """
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == '0':
                return False
    return True


def find_blank(puzzle):
    """ Find the next blank cell """
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == '0':
                return i,j
    return None


def main():
    if len(sys.argv) != 2:
        print("Syntax: python3 sudoku.py <filename>")
        sys.exit(0)
    puzzle = read_puzzle(sys.argv[1])
    if backtrace(puzzle):
        print("Solution:\n")
        print_puzzle(puzzle)
    else:
        print("No Solution.")

main()
