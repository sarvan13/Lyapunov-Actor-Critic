from agent import SACAgent
import networks
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Training Loop

def train_sac(env_name='InvertedPendulum-v5', episodes=350, batch_size=256):

    env = gym.make(env_name, render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    reward_arr = []

    agent = SACAgent(max_action, state_dim, action_dim, 3e-4, 3e-4, 3e-4)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.choose_action(state, reparameterize=False)
            next_state, reward, done, _, _ = env.step(action)
            data_point = (state, action, reward, next_state, done)
            agent.remember(data_point)
            state = next_state
            total_reward += reward

            agent.train(batch_size)
            step_count += 1

            if step_count > 5_000:
                if episode < episodes - 1:
                    done = True
                # Test to see how long the last episode runs for
                elif step_count > 50_000:
                    done = True

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
        reward_arr.append(total_reward)

    env.close()
    plt.plot(reward_arr)
    plt.show()
    np.save("sac-full-test.npy", np.array(reward_arr))
    


train_sac()