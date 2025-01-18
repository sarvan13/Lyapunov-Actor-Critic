import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
x = np.load('sac-cost-step-arr.npy')
y = np.load('sac-cost-arr.npy')
x = x/1000
sac_mean_rewards = [np.mean(y[np.max((0,i - 50)): i]) for i in range(len(y))]

# Plot the data
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('sac-step-arr')
plt.ylabel('sac-cost-arr')
plt.title('Plot of lac-step-arr vs lac-cost-arr')
plt.grid(True)
plt.show()