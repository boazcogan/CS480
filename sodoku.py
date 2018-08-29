# http://dingo.sbs.arizona.edu/~sandiway/sudoku/examples.html

def read_puzzle(filename):
    with open(filename) as f:
        puzzle = [l.split() for l in f.readlines()]
    return puzzle

def print_puzzle(puzzle):
    for i in range(9):
        for j in range(9):
            if j%3 == 0 and j != 0:
                print("    ", end="")
            print(puzzle[i][j], end="  ")
        if i%3 == 2:
            print()
        print()

def is_valid_row(puzzle, R, num):
    for i in range(9):
        if puzzle[R][i] == num:
            return False
    return True


def is_valid_col(puzzle, C, num):
    for i in range(9):
        if puzzle[i][C] == num:
            return False
    return True


#save for later
def is_valid_square(puzzle, C, R, num):
    for i in range(3):
        for j in range(3):
            if puzzle[R+i][C+j] == num:
                return False
    return True


def is_valid(puzzle, C, R, num):
    return is_valid_col(puzzle, C, num) and is_valid_row(puzzle, R, num) and is_valid_square(puzzle, C - C % 3, R - R % 3, num)


def backtracing(puzzle):
    if filled(puzzle):
        return True

    R,C = find_blank(puzzle)

    for i in range(1,10):
        i = str(i)
        if is_valid(puzzle, C, R, i):
            puzzle[R][C] = i
            if backtracing(puzzle):
                return True
            puzzle[R][C] = '0'
    return False


def filled(puzzle):
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == '0':
                return False
    return True


def find_blank(puzzle):
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == '0':
                return i,j
    return None


def main():
    puzzle = read_puzzle("testfile.txt")
    backtracing(puzzle)
    print_puzzle(puzzle)
main()
