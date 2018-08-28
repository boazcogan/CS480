import numpy as np

def read_puzzle(filename):
    with open(filename) as f:
        puzzle = [l.split() for l in f.readlines()]
    return puzzle

def pprint(puzzle):
    for row in puzzle:
        print(row)

def print_puzzle(puzzle):
    for i in range(9):
        for j in range(9):
            if j%3 == 0 and j != 0:
                print("    ", end="")
            print(puzzle[i][j], end="  ")
        if i%3 == 2:
            print()
        print()


def create_puzzle():
    sodoku_puzzle = []
    for i in range(9):
        sodoku_puzzle.append([[],[],[],[],[],[],[],[],[]])
    for i in range(9):
        for j in range(9):
            sodoku_puzzle[i][j] = [input("")]
    print_puzzle(sodoku_puzzle)
    return sodoku_puzzle


def possible_solutions_for_square(puzzle, R, C):
    possible_values = [True for i in range(9)]
    for i in range(9):
        if len(puzzle[R][i]) == 1 and puzzle[R][i][0] != "":
            possible_values[int(puzzle[R][i][0])-1] = False
        if len(puzzle[i][C]) == 1 and puzzle[i][C][0] != "":
            possible_values[int(puzzle[i][C][0])-1] = False
    boxH = R//3
    boxW = C//3
    for i in range(3*boxH,3*(boxH+1)):
        for j in range(3*boxW, (3*boxW+1)):
            if len(puzzle[i][j]) == 1 and puzzle[i][j][0] != "":
                possible_values[int(puzzle[i][j][0])-1] = False
    actual_values = [1,2,3,4,5,6,7,8,9]
    result = np.array(actual_values) *  np.array(possible_values)
    return result



def backtracing(puzzle):
    result = possible_solutions_for_square(puzzle,0,2)
    puzzle[0][2] = result



def main():
    puzzle = create_puzzle()
    backtracing(puzzle)
    print_puzzle(puzzle)
main()
