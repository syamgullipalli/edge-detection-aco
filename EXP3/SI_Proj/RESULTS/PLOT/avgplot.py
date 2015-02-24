import numpy as np
import matplotlib.pyplot as plt

data = []

# Read numpy arrays
for i in xrange(0, 10):
	tmp = np.loadtxt("Iteration"+str(i+1)+".txt")
	data.append(tmp)

# Plot
plt.plot(list(xrange(1,11)), [np.average(i) for i in data], 'b-s')
plt.xlabel("Iteration Count")
plt.ylabel("Average Pheromone Deposit")
plt.savefig('avgplot.pdf')

del data