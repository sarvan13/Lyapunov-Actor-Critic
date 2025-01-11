"""
Taken from Minghao Han
Copied from https://github.com/hithmh/Actor-critic-with-stability-guarantee/blob/master/envs/ENV_V1.py
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np
import pygame

class CartPoleEnv_adv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # Other initialization
        self.render_mode = render_mode  # Add this line to handle render_mode
        self.viewer = None

        self.gravity = 10
        # 1 0.1 0.5 original
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 + 0  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.cons_pos = 4
        self.target_pos = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 10
        # self.max_v=1.5
        # self.max_w=1
        # FOR DATA
        self.max_v = 50
        self.max_w = 50

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.max_v,
            self.theta_threshold_radians * 2,
            self.max_w])

        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.state = None

        if self.render_mode is "Human":
            self.screen_width = 800
            self.screen_height = 400
            self.scale = 100  # Adjust this scale factor based on your observations
            self.cart_color = (0, 0, 255)  # Blue cart
            self.pole_color = (0, 255, 0)  # Green pole
            self.cart_width = 50
            self.cart_height = 30
            self.pole_length = 100  # Length of the pole
            self.pole_width = 10
            # Initialize Pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_params(self, length, mass_of_cart, mass_of_pole, gravity):
        self.gravity = gravity
        self.length = length
        self.masspole = mass_of_pole
        self.masscart = mass_of_cart
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def get_params(self):

        return self.length, self.masspole, self.masscart, self.gravity

    def reset_params(self):

        self.gravity = 10
        self.masscart = 1
        self.masspole = 0.1
        self.length = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def step(self, action, impulse=0, process_noise=np.zeros([5])):
        a = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # self.gravity = np.random.normal(10, 2)
        # self.masscart = np.random.normal(1, 0.2)
        # self.masspole = np.random.normal(0.1, 0.02)
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = np.random.normal(action, 0)# wind
        force = force + process_noise[0] + impulse
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot+ process_noise[2]
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
            theta = theta + self.tau * theta_dot + process_noise[1]
            theta_dot = theta_dot + self.tau * thetaacc + process_noise[3]

            # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w)
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot + process_noise[2]
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
            theta = theta + self.tau * theta_dot + process_noise[1]
            theta_dot = theta_dot + self.tau * thetaacc+ process_noise[3]
            # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            x = x + self.tau * x_dot  + process_noise[2]
            theta_dot = theta_dot + self.tau * thetaacc+ process_noise[3]
            theta = theta + self.tau * theta_dot + process_noise[1]
        self.state = np.array([x, x_dot[0], theta, theta_dot[0]])
        done = abs(x) > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)
        # done = False
        if x < -self.x_threshold \
                or x > self.x_threshold:
            a = 1
        r1 = ((self.x_threshold/10 - abs(x-self.target_pos))) / (self.x_threshold/10)  # -4-----1
        r2 = ((self.theta_threshold_radians / 4) - abs(theta)) / (self.theta_threshold_radians / 4)  # -3--------1
        # r1 = max(10 * (1 - ((x-self.target_pos)/self.x_threshold) **2), 1)
        # r2 = max(10 * (1 - np.abs((theta)/self.theta_threshold_radians)), 1)
        # cost1=(self.x_threshold - abs(x))/self.x_threshold
        e1 = (abs(x)) / self.x_threshold
        e2 = (abs(theta)) / self.theta_threshold_radians
        cost = COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot)
        # cost = 0.1+10*max(0, (self.theta_threshold_radians - abs(theta))/self.theta_threshold_radians) \
        #     #+ 5*max(0, (self.x_threshold - abs(x-self.target_pos))/self.x_threshold)\
        cost = 1* x**2/100 + 20 *(theta/ self.theta_threshold_radians)**2
        l_rewards = 0
        if done:
            cost = 100.
        if abs(x)>self.cons_pos:
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0
        return self.state, cost, done, dict(hit=a,
                                            l_rewards=l_rewards,
                                            cons_pos=self.cons_pos,
                                            cons_theta=self.theta_threshold_radians,
                                            target=self.target_pos,
                                            violation_of_constraint=violation_of_constraint,
                                            reference=0,
                                            state_of_interest=theta,
                                            )

    def reset(self, seed=None, options=None):
        # Seed the environment if a seed is provided
        self.np_random, seed = seeding.np_random(seed)
        
        # Reset the environment state
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[0] = self.np_random.uniform(low=-5, high=5)
        self.steps_beyond_done = None
        
        return np.array(self.state), {}  # Return state and an empty dictionary for compatibility


    def render(self, mode="human"):
        self.screen.fill((255, 255, 255))  # Clear the screen (white background)

        # Draw the cart
        cart_x = self.state[0] * 100 + self.screen_width / 2  # Mapping cart position
        cart_y = self.screen_height // 2

        pygame.draw.rect(self.screen, self.cart_color,
                         (cart_x - self.cart_width / 2, cart_y - self.cart_height / 2,
                          self.cart_width, self.cart_height))

        # Draw the pole
        pole_angle = self.state[2]  # Angle of the pole
        pole_x = cart_x
        pole_y = cart_y - self.cart_height / 2
        pole_end_x = pole_x + np.sin(pole_angle) * self.pole_length
        pole_end_y = pole_y - np.cos(pole_angle) * self.pole_length

        pygame.draw.line(self.screen, self.pole_color, (pole_x, pole_y),
                         (pole_end_x, pole_end_y), self.pole_width)

        pygame.display.flip()  # Update the display

    def close(self):
        pygame.quit()

def COST_1000(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - 4 * abs(x) ** 2
    return cost

def COST_V3(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = np.sign(r2) * ((10 * r2) ** 2) - abs(x) ** 4
    return cost

def COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 20*np.sign(r2) * ((r2) ** 2)+ 1* np.sign(r1) * (( r1) ** 2)
    return cost


def COST_V2(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 5 * max(r2, 0) + 1* max(r1,0) + 1
    return cost
