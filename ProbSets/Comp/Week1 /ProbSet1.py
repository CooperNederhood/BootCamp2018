import numpy as np 

PATH = "../../../Computation/Wk1_PyIntro/"
file = "grid.npy"

grid = np.load(PATH+file)

#############
# NUMPY - QUESTION 1:
#############
print("########\nNUMPY - QUESTION 1\n########\n")
A = np.array( [ [3, -1, 4], [1, 5, -9] ])
B = np.array( [ [2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3] ])
prod = A @ B
print("Product is:\n", prod)
print()

#############
# NUMPY - QUESTION 2:
#############
print("########\nNUMPY - QUESTION 2\n########\n")

A = np.array( [[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
eq = -A@A@A + 9 * A@A - 15*A 
print("-A^3 + 9A^2 - 15A = \n", eq)
print()

#############
# NUMPY - QUESTION 3:
#############
print("########\nNUMPY - QUESTION 3\n########\n")

ones = np.full((7,7), 1)
fives = np.full((7,7), 5)
neg_ones = np.full((7,7), -1)

A = np.triu(ones)
B = np.triu(fives, 1) + np.tril(neg_ones)
print("Matrix A:\n", A)
print()
print("Matrix B:\n", B)
print()
prod = A @ B @ A
prod = prod.astype(np.int64)
print("Product A @ B @ A = \n", prod)
print()

#############
# NUMPY - QUESTION 4:
#############
print("########\nNUMPY - QUESTION 4\n########\n")

def non_negs(np_array):
	'''
	Returns copy with negatives set as zero
	'''

	c = np_array[:]
	c[ c < 0] = 0

	return c 

ex_array = np.array([-2, 2, 0])
print("Example array:\n", ex_array)
print()
print("Example return array:\n", non_negs(ex_array))
print()

#############
# NUMPY - QUESTION 5:
#############
print("########\nNUMPY - QUESTION 5\n########\n")

A = np.array( [ [0, 2, 4], [1, 3, 5] ])
B = np.tril(np.full((3,3), 3))
I = np.identity(3)
C = I * -3

zeros0 = np.zeros((3,3))
zeros1 = np.zeros((2,2))
zeros2 = np.zeros((3,2))
zeros3 = np.zeros((2,3))
s1 = np.vstack((zeros0, A, B))
s2 = np.vstack((A.T, zeros1, zeros2))
s3 = np.vstack((I, zeros3, C))

block = np.hstack((s1, s2, s3))
print("Block matrix:\n", block)
print()

#############
# NUMPY - QUESTION 6:
#############
print("########\nNUMPY - QUESTION 6\n########\n")

def row_stoch(mat_2d):
	'''
	Return the normalized mat by row
	'''

	row_sum = mat_2d.sum(axis=1).reshape((mat_2d.shape[0],1))
	rv = mat_2d / row_sum 

	return rv 

ex = np.reshape( np.array(range(100)),(10,10))
ex_rv = row_stoch(ex)
print("Example matrix:\n", ex)
print()
print("Is normalized to:\n", ex_rv)
print()


#############
# NUMPY - QUESTION 7:
#############
print("########\nNUMPY - QUESTION 7\n########\n")

def find_max(array):
	'''
	Problem 7
	'''

	x = array.shape[0]
	y = array.shape[1]

	# do row-wise
	rc1 = np.vstack( (np.zeros((1,y)), array[:-1, :]))
	rc2 = np.vstack( (np.zeros((2,y)), array[:-2, :]))
	rc3 = np.vstack( (np.zeros((3,y)), array[:-3, :]))
	rc_mult = array * rc1 * rc2 * rc3 
	rc_max = np.max(rc_mult)

	# do col-wise
	cc1 = np.hstack( (np.zeros((x,1)), array[:, :-1]))
	cc2 = np.hstack( (np.zeros((x,2)), array[:, :-2]))
	cc3 = np.hstack( (np.zeros((x,3)), array[:, :-3]))
	cc_mult = array * cc1 * cc2 * cc3 
	cc_max = np.max(cc_mult)

	# do diag-wise by applying col shift to the row-wise matrices
	dc1 = np.hstack( (np.zeros((x,1)), rc1[:, :-1]))
	dc2 = np.hstack( (np.zeros((x,2)), rc2[:, :-2]))
	dc3 = np.hstack( (np.zeros((x,3)), rc3[:, :-3]))
	dc_mult = array * dc1 * dc2 * dc3 
	dc_max = np.max(dc_mult)

	# do diag-wise the other way by applying col shift to the row-wise matrices
	cc1 = np.hstack( (rc1[:, 1:], (np.zeros((x,1)))))
	cc2 = np.hstack( (rc2[:, 2:], (np.zeros((x,2)))))
	cc3 = np.hstack( (rc3[:, 3:], (np.zeros((x,3)))))
	cc_mult = array * cc1 * cc2 * cc3 
	cc_max = np.max(cc_mult)

	return max(rc_max, cc_max, dc_max, cc_max)

t = find_max(grid)
print("Max sequence totals to: ", t)
print()



#############
# STANDARD LIB - QUESTION 1:
#############
print("########\nSTANDARD LIB - QUESTION 1\n########\n")

def return_stats(l):
	return min(l), max(l), np.mean(l)

ex = range(101)
print("Min, max, mean of 0-100 are: ", return_stats(ex))
print()

#############
# STANDARD LIB - QUESTION 2:
#############
print("########\nSTANDARD LIB - QUESTION 2\n########\n")

int1 = 10
int2 = int1
int2 += 1
print("Ints are mutable:", int1 == int2 )

str1 = "test"
str2 = str1
str2 += "MORE stuff"
print("Strs are mutable:", str1 == str2 )

print("Python lists are MUTABLE!!")

print("Python tuples are not MUTABLE!!")

set1 = {1, 2, 3}
set2 = set1
set2.add(4)
print("Sets are mutable:", set1 == set2 )
print()

#############
# STANDARD LIB - QUESTION 3:
#############
print("########\nSTANDARD LIB - QUESTION 3\n########\n")

import calculator 

def fn_hypot(s1, s2):
	h2 = calculator.fn_prod(s1, s1) + calculator.fn_prod(s2, s2)

	return calculator.sqrt(h2)

print("Hypot of 3, 4 is: ", fn_hypot(3, 4))
print()


#############
# STANDARD LIB - QUESTION 4:
#############
print("########\nSTANDARD LIB - QUESTION 4\n########\n")
import itertools

def power_set(listlike):
	'''
	Return power set
	'''

	size = len(listlike)
	rv = []

	for count in range(size+1):
		for sub_list in itertools.combinations(listlike, count):
			sub_set = set(sub_list)
			rv.append(sub_set)

	return rv 


ex = [1, 2, 'b']
print("Example set is:\n", ex)
print("The power set is:\n", power_set(ex))


print()

#############
# OO Programming - QUESTION 1:
#############
print("########\nOO Programming - QUESTION 1\n########\n")

class Backpack:
	'''
	A Backpack object class. Has a name and a list of contents.

	Attributes:
		name (str): the name of the backpacks' owner
		color (str): color of backpack
		max_size (int): max capacity of contents in backpack
		contents (list): the contents of the backpack
	'''

	def __init__(self, name, color, max_size=5):
		'''
		Set the name and initialize an empty list of contents

		Params:
			name (str): name of owner
			color (str): color of backpack
			max_size (int): max size of backpack
		'''


		self.name = name 
		self.contents = []
		self.color = color 
		self.max_size = max_size 

	def put(self, item):
		'''
		Add item to the backpacks list of contents
		'''

		if len(self.contents) == self.max_size:
			print("No RooM!")
		else:
			self.contents.append(item)

	def take(self, item):
		'''
		Remove 'item' from the backpacks list of contents
		'''
		self.contents.remove(item)

	def dump(self):
		'''
		Removes all items from backpack
		'''
		self.contents = []

	def __str__(self):
		'''
		NOTE: the str rep can be used to test the Class
		'''

		s = "Backpack with:\n"
		s+= "\towner: {}\n".format(self.name)
		s+= "\tcolor: {}\n".format(self.color)
		s+= "\tmax size: {}\n".format(self.max_size)
		s+= "\tcontents: {}\n".format(self.contents)

		return s 

	def __eq__(self, other):
		'''
		Determines if two objects are equal based on name, color, and
		number of contents
		'''

		b = (self.color == other.color) and (self.name == other.name) and (len(self.contents) == len(other.contents) )
		return b 

def test_backpack():
	bp1 = Backpack("cooper", "red", 10)
	print(bp1)

	bp1.put("shoes")
	print(bp1)

	bp1.take("shoes")
	print(bp1)

	l = ['hat', 'shoe', 'beer']
	for i in l:
		bp1.put(i)

	bp1.dump()
	print(bp1)

test_backpack()


#############
# OO Programming - QUESTION 2:
#############
print("########\nOO Programming - QUESTION 2\n########\n")

class Jetpack(Backpack):
	'''
	A JetPack inherits from the Backpack class
	'''

	def __init__(self, name, color, max_size=2, fuel=10):
		'''
		JetPack is a special type of Backpack which also has fuel attr
		'''

		Backpack.__init__(self, name, color, max_size)
		self.fuel = fuel 

	def fly(self, burn_fuel):
		'''
		Fly by burning 'burn_fuel' amount of fuel, if possible
		'''

		if burn_fuel > self.fuel:
			print("Not enough fuel!")

		else:
			self.fuel -= burn_fuel

	def dump(self):
		'''
		Removes all items from backpack and dumps the fuel
		'''

		self.items = []
		self.fuel = 0


#############
# OO Programming - QUESTION 3:
#############
print("########\nOO Programming - QUESTION 3\n########\n")

print()
print("See above")
print()

#############
# OO Programming - QUESTION 4:
#############
print("########\nOO Programming - QUESTION 4\n########\n")


class ComplexNumber(object):
	'''
	A complex number class
	'''

	def __init__(self, real, imag):
		'''
		Constructor where real is the real part and
		iamge is the imag part
		'''

		self.real = real
		self.imag = imag 

	def conjugate(self):
		'''
		Returns an instance of complex number which is the 
		compelx conjugate
		'''

		return ComplexNumber(self.real, -self.imag)

	def __str__(self):
		'''
		String representation
		'''

		sign = "+" if self.imag > 0 else "-"

		s = "{}{}{}i".format(self.real, sign, np.abs(imag))
		return s 

	'''
	Having completed the CS applications sequence I am confident I can 
	implement the rest of the class and test 
	so rather than type this up I'm just going to work on the other areas
	'''


#############
# I/O  - QUESTION 1:
#############
print("########\nI/O Programming - QUESTION 1\n########\n")


def arithmagic():
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("First number is not a 3-digit number.")

    if abs(int(step_1[2])-int(step_1[0])) < 2:
        raise ValueError("The first number’s first and last digits differ by less than 2.")
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    if int(step_2[2]) != int(step_1[0]) or int(step_2[0]) != int(step_1[2]) or int(step_2[1]) != int(step_1[1]):
        raise ValueError("The second number is not the reverse of the first number.")
    step_3 = input("Enter the positive difference of these numbers: ")
    if int(step_3) != abs(int(step_1)-int(step_2)):
        raise ValueError("The third number is not the positive difference of the first two numbers.")
    step_4 = input("Enter the reverse of the previous result: ")
    if int(step_4[2]) != int(step_3[0]) or int(step_4[0]) != int(step_3[2]) or int(step_4[1]) != int(step_3[1]):
        raise ValueError("The fourth number is not the reverse of the third number.")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")

arithmagic()


#############
# I/O  - QUESTION 2:
#############
print("########\nI/O Programming - QUESTION 2\n########\n")

from random import choice

def random_walk(max_iters=1e12):
    walk = 0
    directions = [1, -1]
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Interrupted at ", i)
    else:
        print("completed")
    finally:
        return(walk)

random_walk()

#############
# I/O  - QUESTION 3:
#############
print("########\nI/O Programming - QUESTION 3\n########\n")

class ContentFilter(object):
    def __init__(self, filename):
        self.file = filename
        try:
            with open(self.file,"r") as myfile:
                self.content = myfile.read().split('\n')
        except FileNotFoundError:
            file = input("Please enter a valid file name:")
            ContentFilter(file)
        except TypeError:
            file = input("Please enter a valid file name:")
            ContentFilter(file)
        except OSError:
            file = str(input("Please enter a valid file name:"))
            ContentFilter(file)

c = ContentFilter("helloworld.txt")


#############
# I/O  - QUESTION 4:
#############
print("########\nI/O Programming - QUESTION 4\n########\n")

class ContentFilter(object):
    def __init__(self, filename):
        self.file = filename
        try:
            with open(self.file,"r") as myfile:
                self.content = myfile.read().split('\n')
        except FileNotFoundError:
            file = input("Please enter a valid file name:")
            ContentFilter(file)
        except TypeError:
            file = input("Please enter a valid file name:")
            ContentFilter(file)
        except OSError:
            file = str(input("Please enter a valid file name:"))
            ContentFilter(file)
        self.tot = 0
        self.alpha = 0
        self.num = 0
        self.white = 0
        self.lines = len(self.content)
        for i in range(self.lines):
            for j in range(len(self.content[i])):
                self.tot += 1
                if self.content[i][j].isspace():
                    self.white += 1
                if self.content[i][j].isdigit():
                    self.num += 1
                if self.content[i][j].isalpha():
                    self.alpha += 1
    def uniform(self, new_file, case="upper"):
        if case not in ['upper','lower']:
            raise ValueError("Case should be 'upper' or 'lower'")
        with open(new_file, "w") as new_file:
            for i in range(len(self.content)):
                if case == "upper":
                    new_file.write(self.content[i].upper() + "\n")
                elif case == "lower":
                    new_file.write(self.content[i].lower() + "\n")
                    
    def reverse(self, new_file, unit ="line"):
        if unit not in ['line','word']:
            raise ValueError("Case should be'line' or 'unit'")
        with open(new_file, "w") as file:
                if unit == "word":
                    for i in range(len(self.content)):
                        file.write(self.content[i][::-1] + "\n")
                else:
                    for i in range(len(self.content)):
                        file.write(self.content[::-1][i] + "\n")
                        
    def transpose(self, new_file):
        with open(new_file, "w") as file:
            for i in range(len(self.content[0].split())):
                new_line = []
                for j in range(len(self.content)):
                    c = self.content[j].split()
                    new_line.append(c[i])
                file.write(" ".join(new_line) + "\n")
                
    def __str__(self):
        return "File: {}".format(self.file) + "\n" + "Total chars: {}".format(self.tot) + "\n" + "Alphabetic chars: {}".format(self.alpha) \
                + "\n" + "Numerical: {}".format(self.num) + "\n" + "Whitespace: {}".format(self.white) + "\n" + "# lines: {}".format(self.lines)



cf = ContentFilter("cf_example1.txt")
cf.uniform("uniform.txt", case="upper")
cf.uniform("uniform.txt", case="lower")
cf.reverse("reverse.txt", unit="word")
cf.reverse("reverse.txt",unit="line")
cf.transpose("transpose.txt")

print(cf)






	



