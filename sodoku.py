
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

'''
def create_puzzle():
    sodoku_puzzle = []
    for i in range(9):
        sodoku_puzzle.append([[],[],[],[],[],[],[],[],[]])
    for i in range(9):
        for j in range(9):
            sodoku_puzzle[i][j] = [input("")]
    print_puzzle(sodoku_puzzle)
    return sodoku_puzzle
'''
'''
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
'''


def is_valid_row(puzzle, R, num):
    for i in range(9):
        if puzzle[i][R] == num:
            return False
    return True


def is_valid_col(puzzle, C, num):
    for i in range(9):
        if puzzle[C][i] == num:
            return False
    return True


#save for later
def is_valid_square(puzzle, C, R, num):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == num:
                return False
    return True


def is_valid(puzzle, C, R, num):
    return is_valid_col(puzzle, C, num) and is_valid_row(puzzle,R,num) and is_valid_square(puzzle,C,R,num)


def backtracing(puzzle, R, C):
    for i in range(9):
        puzzle[R][C] = i
        if is_valid(puzzle, C, R, num):
            backtracing(puzzle, (R+1)%9, (C+1)%9)




def main():
    #puzzle = create_puzzle()
    puzzle = read_puzzle("testfile.txt")
    backtracing(puzzle)
    print_puzzle(puzzle)
main()
