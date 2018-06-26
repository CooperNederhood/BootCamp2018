# test_solutions.py
"""Volume 1B: Testing.
<Name>
<Class>
<Date>
"""

import solutions as soln
import pytest
import math 

# Problem 1: Test the addition and fibonacci functions from solutions.py
def test_addition():
    assert soln.addition(1,2) == 3, "Incorrect!"

def test_smallest_factor():
    assert soln.smallest_factor(10) == 2, "Incorrect!"
    assert soln.smallest_factor(3) == 3, "Incorrect!"
    assert soln.smallest_factor(1) == 1, "Incorrect!"

# Problem 2: Test the operator function from solutions.py
def test_month():
    assert soln.month_length("April") == 30, "Failed 30 month"
    assert soln.month_length("February", True) == 29, "Failed leap year"
    assert soln.month_length("January") == 31, "Failed 31 month"
    assert soln.month_length("February") == 28, "Failed non-leap feb"


# Problem 3: Test the operator function
    # with pytest.raises(TypeError) as excinfo:
    #     soln.operator(2, 0, 0)
    #assert excinfo.value.args[0] == "Oper should be a string"

# def test_operator():
#     # with pytest.raises(TypeError) as excinfo:
#     #     soln.operator(2, 0, 0)
#     # assert excinfo.value.args[0] == "Oper should be a string"
#     # with pytest.raises(ValueError) as excinfo:
#     #     soln.operator(2, 0, ")")
#     assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"
#     assert soln.operator(2, 1, "/") == 2, "Inc /"
#     assert soln.operator(1, 1, "+") == 2, "Inc +"
#     assert soln.operator(1, 1, "*") == 1, "Inc *"
#     assert soln.operator(2, 1, "-") == 1, "Inc -"   with pytest.raises(ValueError) as excinfo:
#         soln.operator(2, 0, ")")
#     assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"
#     assert soln.operator(2, 1, "/") == 2, "Inc /"
#     assert soln.operator(1, 1, "+") == 2, "Inc +"
#     assert soln.operator(1, 1, "*") == 1, "Inc *"
#     assert soln.operator(2, 1, "-") == 1, "Inc -"

# Problem 4: Finish testing the complex number class
@pytest.fixture
def set_up_complex_nums():
    number_1 = soln.ComplexNumber(1, 2)
    number_2 = soln.ComplexNumber(5, 5)
    number_3 = soln.ComplexNumber(2, 9)
    return number_1, number_2, number_3

def test_complex_addition(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 + number_2 == soln.ComplexNumber(6, 7)
    assert number_2 + number_3 == soln.ComplexNumber(7, 14)
    assert number_1 + number_3 == soln.ComplexNumber(3, 11)
    assert number_3 + number_3 == soln.ComplexNumber(4, 18)

def test_complex_multiplication(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 * number_2 == soln.ComplexNumber(-5, 15)
    assert number_1 * number_3 == soln.ComplexNumber(-16, 13)
    assert number_2 * number_3 == soln.ComplexNumber(-35, 55)
    assert number_3 * number_3 == soln.ComplexNumber(-77, 36)
def test_complex_division(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 / number_2 == soln.ComplexNumber(0.3, 0.1)

def test_complex_str(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert str(number_1) == "1+2i", "Not printing correctly"

def test_complex_eq(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    temp =  soln.ComplexNumber(1,2)
    assert number_1 == temp, "failed for equal"

def test_complex_init(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    print (number_1.real==1)
    assert number_1.real == 1 , "Real part not setting up correctly"
    assert number_1.imag == 2, "Imaginary part not setting up correctly"

def test_complex_conjugate(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.conjugate() == soln.ComplexNumber(1, -2)

def test_complex_norm(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.norm() == math.sqrt(5)


# Problem 5: Write test cases for the Set game.
def set_up_cards():
    hand1 = [ "1022", "1122", "1020", "1022", "1122", "0100", "2021",
               "0010", "2201", "2111", "0020", "1102", "0210", "2110", "1020"]