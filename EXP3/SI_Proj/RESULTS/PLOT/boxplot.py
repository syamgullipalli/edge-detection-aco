import numpy as np
import matplotlib.pyplot as plt

data = []

# Read numpy arrays
for i in xrange(0, 10):
	tmp = np.loadtxt("Iteration"+str(i+1)+".txt")
	data.append(tmp)

# Box plot
plt.boxplot(data, notch=True, sym='b+')
plt.xlabel("Iteration Count")
plt.ylabel("Pheromone Deposit")
plt.savefig('boxplot.pdf')

del data