import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
# x = np.load('C:/Users/Sarvan/Desktop/School/UVIC/Lyapunov-Actor-Critic/lac-step-arr.npy')
# y = np.load('C:/Users/Sarvan/Desktop/School/UVIC/Lyapunov-Actor-Critic/lac-cost-arr.npy')
x = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/lac-step-arr.npy')
y = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/lac-cost-arr.npy')
y2 = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/lac-lambda-arr.npy')
x = x/1000
sac_mean_rewards = [np.mean(y[np.max((0,i - 50)): i]) for i in range(len(y))]

# Load the .npy files
x2 = np.load('sac-3cost-step-arr.npy')
y2 = np.load('sac-3cost-arr.npy')
x2 = x2/1000
sac_mean_rewards2 = [np.mean(y2[np.max((0,i - 50)): i]) for i in range(len(y2))]

# Plot the data
plt.plot(x, sac_mean_rewards, marker='o', linestyle='-')
plt.plot(x2, sac_mean_rewards2, marker='o', linestyle='-')
plt.xlabel('sac-step-arr')
plt.ylabel('sac-cost-arr')
plt.title('Plot of lac-step-arr vs lac-cost-arr')
plt.grid(True)
plt.show()
plt.plot(x, y2, marker='o', linestyle='-')
plt.xlabel('Steps (1000s)')
plt.ylabel('Lagrange Multiplier')
plt.title('LAC')
plt.grid(True)
plt.show()