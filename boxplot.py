import numpy as np
import matplotlib.pyplot as plt

exp = []

# For each experiment folder
for e in xrange(0,3):
	data = []
	# Read numpy arrays
	for i in xrange(0, 10):
		tmp = np.loadtxt("EXP"+str(e+1)+"/SI_Proj/RESULTS/lena/PLOT/Iteration"+str(i+1)+".txt")
		data.append(tmp)
	exp.append(data)

for i in xrange(0, 10):
	data = []
	data = [exp[0][i], exp[1][i], exp[2][i]]
	# Plot
	plt.boxplot(data, notch=True, sym='b+')
	plt.ylabel("Pheromone Deposit")
	ax = plt.axes()
	ax.yaxis.grid() #horizontal lines
	plt.savefig("Iteration"+str(i+1)+".pdf", format='pdf', bbox_inches='tight')
	plt.show()
