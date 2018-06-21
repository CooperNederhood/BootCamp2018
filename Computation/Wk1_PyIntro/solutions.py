# solutions.py
"""Volume IB: Testing.
Cooper Nederhood

"""
import math
import sys
import time
import numpy.random as rand 
import box 
import numpy as np

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.


def addition(a, b):
    return a + b


def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def operator(a, b, oper):
    if type(oper) != str:
        raise ValueError("Oper should be a string")
    if len(oper) != 1:
        raise ValueError("Oper should be one character")
    if oper == "+":
        return a + b
    if oper == "/":
        if b == 0:
            raise ValueError("You can't divide by zero!")
        return a/float(b)
    if oper == "-":
        return a-b
    if oper == "*":
        return a*b
    else:
        raise ValueError("Oper can only be: '+', '/', '-', or '*'")

# Problem 3 Write unit test for this class.
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real, imag)

    def __mul__(self, other):
        real = self.real*other.real - self.imag*other.imag
        imag = self.imag*other.real + other.imag*self.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ValueError("Cannot divide by zero")
        bottom = (other.conjugate()*other*1.).real
        top = self*other.conjugate()
        return ComplexNumber(top.real / bottom, top.imag / bottom)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag))

# Problem 5: Write code for the Set game here

def gen_roll(remaining):
    '''
    Returns a roll total given the remianing numbers
    Parameters:
        remaining (list): The list of the numbers that still need to be
            removed before the box can be shut.

    Returns:
        (int) roll
    '''

    if np.sum(remaining) <= 6:
        count = 1
    else:
        count = 2

    return np.sum(rand.randint(low=1, high=7, size=count))


if __name__ == '__main__':

    if len(sys.argv) == 3:
    
        tot_time = int(sys.argv[2])
        player = sys.argv[1]
        print("{} is playing Shut the Box with time {}".format(player, tot_time))
        remaining = list(range(1, 10))
        elapsed = 0

        win = False
        start = time.time()
        elapsed = time.time() - start 

        while tot_time - elapsed > 0:
            roll = gen_roll(remaining)

            print("Numbers left: ", remaining)
            print("Roll: ", roll)

            if not box.isvalid(roll, remaining):
                print("Game over!")
                print()
                break 

            print("Seconds left: ", round(tot_time - elapsed))
            to_elim_input = input("Numbers to eliminate: ")

            to_elim = box.parse_input(to_elim_input, remaining)
            while not to_elim:
                print("Invalid input")
                print()
                elapsed = time.time() - start 
                print("Seconds left: ", round(tot_time - elapsed))
                to_elim_input = input("Numbers to eliminate: ")
                to_elim = box.parse_input(to_elim_input, remaining)

            print()

            remaining = [x for x in remaining if x not in to_elim]

            if len(remaining) == 0:
                win = True 
                break 

            elapsed = time.time() - start 

        score = np.sum(remaining)
        print("Score for player {}: {} points".format(player, score))
        print("Time played: {} seconds".format(elapsed))
        if win:
            print("Congratulations!! You shut the box!!")
        else:
            print("Better luck next time >:)")



