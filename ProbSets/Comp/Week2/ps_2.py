import numpy as np 
import matplotlib.pyplot as plt 

#############
# Matplotlib - QUESTION 1:
#############
section = "Matplotlib"
print("########\n{} - QUESTION 1\n########\n".format(section))

def prob_1(n):
	'''
	Arbitrary function tasks given some n
	'''

	r = np.random.normal(size=(n,n))
	r_mean = r.mean(axis=1)

	return r_mean.var()

def other_fn(n_range):
	'''
	Second arbitrary function
	'''

	y_vals = np.empty(n_range.size)

	for i in range(y_vals.shape[0]):
		y_vals[i] = prob_1(n_range[i])

	return y_vals


n_range = np.arange(100, 1100, 100)
y_vals = other_fn(n_range)

plt.plot(n_range, y_vals)
plt.title('Mean of std normal draws')
#plt.show()
plt.clf()


#############
# Matplotlib - QUESTION 2:
#############
section = "Matplotlib"
question = "2"
print("########\n{} - QUESTION {}\n########\n".format(section, question))

x_vals = np.linspace(-2*np.pi, 2*np.pi, 1000)
d = {}
d['sin'] =  np.sin(x_vals)
d['cos'] =  np.cos(x_vals)
d['arctan'] =  np.arctan(x_vals)

plt.clf()
for l, y_vals in d.items():
	plt.plot(x_vals, y_vals, label=l)

plt.title('TRIG FUNCTIONS!!!!')
plt.legend()
#plt.show()
plt.clf()


#############
# Matplotlib - QUESTION 3:
#############
section = "Matplotlib"
question = "3"
print("########\n{} - QUESTION {}\n########\n".format(section, question))

fn = lambda x: 1 / (x-1)

def fn_plotter(fn, x_min, x_max):
	'''
	Practice plotting
	'''

	x_vals = np.arange(x_min, x_max)
	i_inf = int(np.argwhere(fn(x_vals) == np.inf)[0])

	x_vals0 = np.arange(x_min, x_vals[i_inf])
	x_vals1 = np.arange(x_vals[i_inf], x_max)

	plt.plot(x_vals0, fn(x_vals0), lw=4, color='m')
	plt.plot(x_vals1, fn(x_vals1), lw=4, color='m')

	plt.xlim(-2, 6)
	plt.ylim(-6,6)
	#plt.show()

fn_plotter(fn, -2, 6)

plt.clf()

#############
# Matplotlib - QUESTION 4:
#############
section = "Matplotlib"
question = "4"
print("########\n{} - QUESTION {}\n########\n".format(section, question))


def sub_plots():
	'''
	Practice doing subplots
	'''
	x_vals = np.linspace(0, 2*np.pi, 100)

	fig, axes = plt.subplots(2,2)

	axes[0][0].plot(x_vals, np.sin(x_vals), '-', color='g')
	plt.axis([0, 2*np.pi, -2, 2])

	axes[1][0].plot(x_vals, np.sin(2*x_vals), '--', color='r')
	plt.axis([0, 2*np.pi, -2, 2])

	axes[0][1].plot(x_vals, 2*np.sin(x_vals), '--', color='b')
	plt.axis([0, 2*np.pi, -2, 2])

	axes[1][1].plot(x_vals, 2*np.sin(2*x_vals), ':', color='m')
	plt.axis([0, 2*np.pi, -2, 2])

	plt.suptitle('SOME SIN GRAPHS!!')

	plt.show()

#sub_plots()

#############
# Matplotlib - QUESTION 5:
#############
section = "Matplotlib"
question = "5"
print("########\n{} - QUESTION {}\n########\n".format(section, question))

root = "../../../Computation/Wk2_DataVis/"
fars = "FARS.npy"

def vis_fars(root, file):
	'''
	Visualize the FARS data
	'''

	TIME = 0
	LONG = 1
	LAT = 2

	fars = np.load(root+file)

	fig, axes = plt.subplots(1, 2)
	axes[0].plot( fars[:,LONG],  fars[:,LAT] , "k,", color='b', alpha=0.5)
	plt.xlabel('Longitude')
	plt.ylabel('Latitude')

	axes[1].hist(fars[:,TIME], bins=24, range=[0, 24])
	plt.xlabel('Hours')

	plt.show()


#vis_fars(root, fars)
plt.clf()

#############
# Matplotlib - QUESTION 6:
#############
section = "Matplotlib"
question = "6"
print("########\n{} - QUESTION {}\n########\n".format(section, question))

def mesh_plot():
	'''
	Plot a function over -2pi to 2pi
	'''

	r = np.linspace(-2*np.pi, 2*np.pi, 100)
	X, Y = np.meshgrid(r, r)

	f = lambda x, y: (np.sin(x) * np.sin(y)) / (x * y)

	Z = f(X, Y)
	plt.subplot(121)
	plt.pcolormesh(X, Y, Z, cmap="viridis")
	plt.colorbar()

	plt.subplot(122)
	plt.contour(X, Y, Z, 20, cmap="coolwarm")
	plt.colorbar()

	fig = plt.gcf()
	fig.set_size_inches(10, 5)

	plt.show()

mesh_plot()



