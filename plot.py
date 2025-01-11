import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
x = np.load('lac-step-arr.npy')
y = np.load('lac-cost-arr.npy')

# Plot the data
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('lac-step-arr')
plt.ylabel('lac-cost-arr')
plt.title('Plot of lac-step-arr vs lac-cost-arr')
plt.grid(True)
plt.show()