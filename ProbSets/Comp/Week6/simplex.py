import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import timeit 
from autograd import grad 
from autograd import elementwise_grad

#############
# Simplex - QUESTION 1:
#############
plt.clf()
section = "Simplex"
print("########\n{} - QUESTION 1\n########\n".format(section))


class SimplexSolver():

	def __init__(self, c, A, b):

		''' 
		QUESTION 1:
		Make c and b Nx1 column vectors
		'''

		self.n = c.shape[0]
		self.m = b.shape[0]

		self.c = c.reshape( (self.n, 1) )
		self.A = A
		self.b = b.reshape( (self.m, 1) )

		self.variables = self.build_tracker()
		self.tab = self.build_tab()

		if np.any(A @ np.zeros( (self.n, 1) ) > self.b):
			raise ValueError('Origin not within feasible set')



	def build_tracker(self):
		'''
		QUESTION 2:
		Helper function called in constructor to
		build variable tracker
		'''

		x_index = np.array(range(0, self.n))
		w_index = np.array(range(0, self.m))
		w_index += self.n

		return np.append(w_index, x_index)

	def build_tab(self):
		'''
		QUESTION 3:
		Helper function called in constructor to
		build the tableau
		'''

		A_rows = np.hstack((self.b, self.A, np.identity(self.m), np.zeros((self.m,1))))

		c_row = np.hstack( (np.zeros((1,1)), -self.c.T, np.zeros((1,self.m)), np.ones((1,1))))
		
		return np.vstack( (c_row, A_rows) )

	def get_pivot_index(self):
		'''
		QUESTION 4:
		Function determines (i,j) index for 
		which to pivot
		'''

		# Determine column 
		col = 1
		while self.tab[0,col] >= 0:
			col += 1

		# Determine the row within that column
		if np.all(self.tab[1:,col] <= 0):
			print("Problem is unbounded")
			return None 

		ratios = self.tab[1:,0] / self.tab[1:,col]
		ratios[ ratios<0 ] = np.inf 
		row = np.argmin(ratios) + 1

		return (row, col)

	def do_pivot(self):
		'''
		QUESTION 5:
		Performs a single pivot
		'''

		row, col = self.get_pivot_index()
		enter_inx = col - 1 
		leave_inx = self.tab[row, col]
		#leave_inx = row + 1

		# swap indexes in var list
		print("Enter inx = {}, Leave inx = {}".format(enter_inx, leave_inx))
		enter_loc = np.where(self.variables==enter_inx)[0][0]
		leave_loc = np.where(self.variables==leave_inx)[0][0]
		self.variables[enter_loc], self.variables[leave_loc] = self.variables[leave_loc], self.variables[enter_loc]


		# Do row operations
		self.tab[row, :] /= leave_inx
		assert self.tab[row, col] == 1
		for i in range(self.tab.shape[0]):
			if i != row:
				factor = -self.tab[i, col]/self.tab[row, col]
				self.tab[i, :] += self.tab[row, :]*factor 

		print("#"*20)
		print(self.tab)
		print(self.variables)
		print("#"*20)


	def solve(self):
		'''
		QUESTION #6:
		Solve simplex
		'''

		while np.any(self.tab[0:, 1:] < 0):
			self.do_pivot()

		max_val = self.tab[0,0]


	def __str__(self):

		header = "#"*20
		s = "c = \n{}\n \nA = \n{}\n \nb = \n{}\n \nvars = \n{}\n \nTableau= \n{}".format(self.c, self.A, self.b, self.variables, self.tab)

		return header+"\n"+s+"\n"+header

c = np.array([3, 2])
A = np.array([ [1, -1], [3, 1], [4, 3] ])
b = np.array([2, 5, 7])

ss = SimplexSolver(c, A, b)
print(ss)
ss.solve()
print(ss)

