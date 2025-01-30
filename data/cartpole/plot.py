import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
# x = np.load('C:/Users/Sarvan/Desktop/School/UVIC/Lyapunov-Actor-Critic/lac-step-arr.npy')
# y = np.load('C:/Users/Sarvan/Desktop/School/UVIC/Lyapunov-Actor-Critic/lac-cost-arr.npy')
x = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/data/cartpole/arrays/lac-step2-arr.npy')
y = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/data/cartpole/arrays/lac-cost2-arr.npy')
x = x/1000
lac_mean_rewards = [np.mean(y[np.max((0,i - 50)): i]) for i in range(len(y))]

x1 = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/data/cartpole/arrays/sac-3cost-step-arr.npy')
y1 = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/data/cartpole/arrays/sac-3cost-arr.npy')
x1 = x1/1000
sac_mean_rewards = [np.mean(y1[np.max((0,i - 50)): i]) for i in range(len(y1))]

# Plot the data
plt.plot(x, lac_mean_rewards, marker='o', linestyle='-', label='LAC')
plt.plot(x1, sac_mean_rewards, marker='o', linestyle='-', label='SAC')
plt.xlabel('Steps (1000s)')
plt.ylabel('Cost per Episode')
plt.title('Inverted Pendulum Performance')
plt.legend()
plt.grid(True)
plt.show()

lambda_arr = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/data/cartpole/arrays/lac-lambda2-arr.npy')
lambda_mean_arr = [np.mean(lambda_arr[np.max((0,i - 50)): i]) for i in range(len(lambda_arr))]
beta_arr = np.load('/home/sarvan/Classes/Lyapunov-Actor-Critic/data/cartpole/arrays/lac-beta2-arr.npy')
beta_mean_arr = [np.mean(beta_arr[np.max((0,i - 50)): i]) for i in range(len(beta_arr))]

plt.plot(x, lambda_mean_arr, marker='o', linestyle='-', label='Lambda')
plt.plot(x, beta_mean_arr, marker='o', linestyle='-', label='Beta')
plt.xlabel('Steps (1000s)')
plt.ylabel('Value')
plt.title('LAC Inverted Pendulum Hyperparameters')
plt.legend()
plt.grid(True)
plt.show()
