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

count = list(xrange(1,11))
exp1 = [np.std(i) for i in exp[0]]
exp2 = [np.std(i) for i in exp[1]]
exp3 = [np.std(i) for i in exp[2]]

# Plot
# Plot
pl_exp1, = plt.plot(count, exp1, linestyle='-', marker='v', color='red', label='Ants initialized on random positions')
pl_exp2, = plt.plot(count, exp2, linestyle='-', marker='s', color='green', label='Ants initialized on most promising solutions')
pl_exp3, = plt.plot(count, exp3, linestyle='-', marker='o', color='blue',  label='Improved search based on ants average pheromone deposit')
plt.legend((pl_exp1, pl_exp2, pl_exp3), 
	(	'Ants initialized on random positions', 
		'Ants initialized on most promising solutions', 
		'Improved search based on ants average pheromone deposit'), 
	loc=2,
	prop={'size':12}, 
	fancybox=True)
plt.grid(True)
plt.xlabel("Iteration Count")
plt.ylabel("Standard Deviation of Pheromone Deposit")
plt.savefig('stddev.pdf')
