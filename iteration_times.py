import numpy as np
import matplotlib.pyplot as plt

exp = []

# For each experiment folder
for e in xrange(0,3):
	tmp = np.loadtxt("EXP"+str(e+1)+"/SI_Proj/RESULTS/mandril/PLOT/elapsed_time.txt")
	exp.append(tmp)

count = list(xrange(1,11))

# Plot
pl_exp1, = plt.plot(count, exp[0], linestyle='-', marker='v', color='red', label='Ants initialized on random positions')
pl_exp2, = plt.plot(count, exp[1], linestyle='-', marker='s', color='green', label='Ants initialized on most promising solutions')
pl_exp3, = plt.plot(count, exp[2], linestyle='-', marker='o', color='blue',  label='Improved search based on ants average pheromone deposit')
plt.legend((pl_exp1, pl_exp2, pl_exp3), 
	(	'Ants initialized on random positions', 
		'Ants initialized on most promising solutions', 
		'Improved search based on ants average pheromone deposit'), 
	loc=2,
	prop={'size':10}, 
	fancybox=True)
plt.grid(True)
plt.xlabel("Iteration Count")
plt.ylabel("Time elapsed for iteration (in milli seconds)")
plt.savefig('elapsed_time.pdf')