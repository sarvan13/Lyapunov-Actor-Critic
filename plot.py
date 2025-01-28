import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
# x = np.load('C:/Users/Sarvan/Desktop/School/UVIC/Lyapunov-Actor-Critic/lac-step-arr.npy')
# y = np.load('C:/Users/Sarvan/Desktop/School/UVIC/Lyapunov-Actor-Critic/lac-cost-arr.npy')
x = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/lac-step-arr.npy')
y = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/lac-cost-arr.npy')
x = x/1000
lac_mean_rewards = [np.mean(y[np.max((0,i - 50)): i]) for i in range(len(y))]

# Plot the data
plt.plot(x, lac_mean_rewards, marker='o', linestyle='-')
plt.xlabel('Steps (1000s)')
plt.ylabel('Cost per Episode')
plt.title('LAC')
plt.grid(True)
plt.show()