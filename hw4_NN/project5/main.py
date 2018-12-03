'''
Authors: Boaz Cogan, Scott Mitchell, Mike Dito
'''


import Part1
import Part2
import Part3


def main():
    while(True):
        selection = input("Please select a problem press q to quit: (1,2,3,q):")
        if (selection == '1'):
            Part1.main()
        elif (selection == '2'):
            Part2.main()
        elif (selection == '3'):
            Part3.main()
        elif (selection == 'q'):
            exit()
        else:
            print("Please select a valid option")


if __name__ == '__main__':
    main()
