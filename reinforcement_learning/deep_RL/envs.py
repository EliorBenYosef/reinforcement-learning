"""
https://github.com/openai/gym/wiki/Table-of-environments

https://github.com/openai/gym/tree/master/gym/envs -
    check gym/gym/envs/__init__.py for solved properties (max_episode_steps, reward_threshold, optimum).
    Solved: avg_score >= reward_threshold, over 100 consecutive trials.
    "Unsolved environment" - doesn't have a specified reward_threshold at which it's considered solved.

Environments:
    Classic Control - CartPole, Pendulum, MountainCarContinuous.  # TODO: Acrobot.
    Box2D - LunarLander, LunarLanderContinuous, BipedalWalker.
    Atari - Breakout, SpaceInvaders.

####################################

Atari environments:
Atari environments must be trained on a GPU (will basically take thousands of years on CPU).

observation's shape: (210, 160, 3)    # (H, W, C)
screen_size = (210, 160)              # (H, W)
image_channels = 3                    # RGB

observation pre-process # reshaping (usually)
  1. the atari screen should be truncated (cropped) - since there's no need for the score, etc...
  2. remove color by getting the mean of the 3 channels (axis=2 means along the RGB values)

input_type = INPUT_TYPE_STACKED_FRAMES
"""

import numpy as np
import gym

from reinforcement_learning.deep_RL.const import INPUT_TYPE_OBSERVATION_VECTOR, INPUT_TYPE_STACKED_FRAMES, \
    ATARI_FRAMES_STACK_SIZE, ATARI_IMAGE_CHANNELS_GRAYSCALE
from reinforcement_learning.utils.utils import normalize


class BaseEnv:
    name: str
    file_name: str
    env: gym.wrappers.time_limit.TimeLimit
    input_type: int
    input_dims: tuple
    is_discrete_action_space: bool
    n_actions: int
    action_space: list
    GAMMA: float
    # EPS_MIN: float

    @staticmethod
    def get_state(observation, prev_s):
        return observation

    @staticmethod
    def update_reward(reward, done, info):
        return reward


# ClassicControl:

class CartPole(BaseEnv):
    """
    AKA "Inverted Pendulum".
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
        The system is controlled by applying a force of +1 or -1 to the cart.

    Goal: to prevent the pendulum from falling over.

    Starting State: the pendulum starts upright.

    Episode Termination (besides reaching the goal):
        the pole is more than 15 degrees from vertical.
        the cart moves more than 2.4 units from the center.

    Rewards: +1 for every time-step that the pole remains upright.

    Solved:
    gym/gym/envs/__init__.py :
        CartPole-v0: max_episode_steps = 200, reward_threshold = 195.0
        CartPole-v1: max_episode_steps = 500, reward_threshold = 475.0

    Continuous observation space (4D).
        O = ndarray[x, x_dot, theta, theta_dot]
            x - Cart Position [-2.4, 2.4]
            x_dot - Cart Velocity [-Inf, Inf]
            theta - Pole Angle [~-41.8°, ~41.8°]
            theta_dot - Pole Velocity [-Inf, Inf]

    Discrete action space (1D).
    Actions (2): left (0), right (1)
    cart_pole_policy = lambda theta_state: 0 if theta_state < (pole_theta_bin_num // 2) else 1
    """

    def __init__(self):
        self.name = 'Cart Pole'
        self.file_name = 'cart-pole-v1'
        self.env = gym.make('CartPole-v1')

        self.input_type = INPUT_TYPE_OBSERVATION_VECTOR
        self.input_dims = (4,)   # Box(4,)

        self.is_discrete_action_space = True
        self.n_actions = 2      # Discrete(2)
        self.action_space = [i for i in range(self.n_actions)]

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 1000000
        self.memory_batch_size = 64


class Pendulum(BaseEnv):
    """
    Solved:
    gym/gym/envs/__init__.py :
        Pendulum-v0: max_episode_steps = 200
        InvertedPendulum-v2: max_episode_steps = 1000, reward_threshold = 950.0
        InvertedDoublePendulum-v2: max_episode_steps = 1000, reward_threshold = 9100.0

    Continuous observation space (3D).

    Continuous action space (1D).
    """

    def __init__(self):
        self.name = 'Pendulum'
        self.file_name = 'pendulum-v0'
        self.env = gym.make('Pendulum-v0')

        self.input_type = INPUT_TYPE_OBSERVATION_VECTOR
        self.input_dims = (3,)   # Box(3,)

        self.is_discrete_action_space = False
        self.n_actions = 1      # Box(1,)
        self.action_boundary = 2

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 1000000
        self.memory_batch_size = 64


class MountainCarContinuous(BaseEnv):
    """
    Solved:
    gym/gym/envs/__init__.py :
        MountainCarContinuous-v0: max_episode_steps = 999, reward_threshold = 90.0

    Continuous observation space (2D).

    Continuous action space (1D).
    """

    def __init__(self):
        self.name = 'Mountain Car Continuous'
        self.file_name = 'mountain-car-continuous-v0'
        self.env = gym.make('MountainCarContinuous-v0')

        self.input_type = INPUT_TYPE_OBSERVATION_VECTOR
        self.input_dims = (2,)   # Box(2,)

        self.is_discrete_action_space = False
        self.n_actions = 1      # Box(1,)
        self.action_boundary = 1

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 1000000
        self.memory_batch_size = 64


# Box2D:

class LunarLander(BaseEnv):
    """
    Solved:
    gym/gym/envs/__init__.py :
        LunarLander-v2: max_episode_steps = 1000, reward_threshold = 200

    Continuous observation space (8D).

    Discrete action space (1D).
    Actions (4)
    """

    def __init__(self):
        self.name = 'Lunar Lander'
        self.file_name = 'lunar-lander-v2'
        self.env = gym.make('LunarLander-v2')

        self.input_type = INPUT_TYPE_OBSERVATION_VECTOR
        self.input_dims = (8,)   # Box(8,)

        self.is_discrete_action_space = True
        self.n_actions = 4      # Discrete(4)
        self.action_space = [i for i in range(self.n_actions)]

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 1000000
        self.memory_batch_size = 64


class LunarLanderContinuous(BaseEnv):
    """
    Solved:
    gym/gym/envs/__init__.py :
        LunarLanderContinuous-v2: max_episode_steps = 1000, reward_threshold = 200

    Continuous observation space (8D).

    Continuous action space (2D).
    """

    def __init__(self):
        self.name = 'Lunar Lander Continuous'
        self.file_name = 'lunar-lander-continuous-v2'
        self.env = gym.make('LunarLanderContinuous-v2')

        self.input_type = INPUT_TYPE_OBSERVATION_VECTOR
        self.input_dims = (8,)   # Box(8,)

        self.is_discrete_action_space = False
        self.n_actions = 2      # Box(2,)
        self.action_boundary = [1, 1]

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 1000000
        self.memory_batch_size = 64


class BipedalWalker(BaseEnv):
    """
    Solved:
    gym/gym/envs/__init__.py :
        BipedalWalker-v3: max_episode_steps = 1600, reward_threshold = 300
        BipedalWalkerHardcore-v3: max_episode_steps = 2000, reward_threshold = 300

    Continuous observation space (24D).
    State vector consists of:
        hull angle speed
        angular velocity
        horizontal speed
        vertical speed
        position of joints
        joints angular speed
        legs contact with ground
        10 lidar rangefinder measurements
    There are no coordinates in the state vector.

    Continuous action space (4D).

    DDPG - 5K episodes take around 12h, and it takes around 15-20K to get score 255
    """

    def __init__(self):
        self.name = 'Bipedal Walker'
        self.file_name = 'bipedal-walker-v3'
        self.env = gym.make('BipedalWalker-v3')

        self.input_type = INPUT_TYPE_OBSERVATION_VECTOR
        self.input_dims = (24,)      # Box(24,)

        self.is_discrete_action_space = False
        self.n_actions = 4          # Box(4,)
        self.action_boundary = [1, 1, 1, 1]

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 1000000
        self.memory_batch_size = 64


# Atari:

def stack_frames(stacked_frames, frame):  # to get a sense of motion. observation == frame, prev s == stacked_frames
    if stacked_frames is None:  # start of the episode: duplicate frame
        return np.repeat(frame, repeats=ATARI_FRAMES_STACK_SIZE, axis=2)
    else:  # remove first frame, and add frame to the end
        return np.concatenate((stacked_frames[:, :, 1:], frame), axis=2)


class Breakout(BaseEnv):
    """
    Continuous observation space (210,160,3 D).

    Discrete action space (1D).
    Actions (4)
    """

    def __init__(self):
        self.name = 'Breakout'
        self.file_name = 'breakout-v0'
        self.env = gym.make('Breakout-v0')

        self.input_type = INPUT_TYPE_STACKED_FRAMES
        self.image_channels = ATARI_IMAGE_CHANNELS_GRAYSCALE
        self.relevant_screen_size = (180, 160)
        self.input_dims = (*self.relevant_screen_size, ATARI_FRAMES_STACK_SIZE)     # Box(210,160,3)

        self.is_discrete_action_space = True
        self.n_actions = 3                                                          # Discrete(4)
        self.action_space = [1, 2, 3]

        self.GAMMA = 0.99
        self.EPS_MIN = None

        self.memory_size = 6000  # saving transitions (stacked frames): 6-7K --> ~16Gb RAM, 25K --> ~48Gb RAM
        self.memory_batch_size = 32

    def get_state(self, observation, prev_s):
        pre_processed_o = self.preprocess_image(observation)
        s = stack_frames(prev_s, pre_processed_o)
        return s

    def preprocess_image(self, o):
        o = o[30:, :]  # crop
        if self.image_channels == ATARI_IMAGE_CHANNELS_GRAYSCALE:  # adjust channels from RGB to Grayscale
            o = np.mean(o, axis=2)[:, :, np.newaxis]  # .reshape((*self.relevant_screen_size, 1))
        o = normalize(o)
        return o


class SpaceInvaders(BaseEnv):
    """
    Continuous observation space (210,160,3 D).

    Discrete action space (1D).
    Actions (6): none (0), fire (1), right (2), left (3), right & fire (4), left & fire (5)
    """

    def __init__(self):
        self.name = 'Space Invaders'
        self.file_name = 'space-invaders-v0'
        self.env = gym.make('SpaceInvaders-v0')

        self.input_type = INPUT_TYPE_STACKED_FRAMES
        self.image_channels = ATARI_IMAGE_CHANNELS_GRAYSCALE
        self.relevant_screen_size = (185, 95)
        self.input_dims = (*self.relevant_screen_size, ATARI_FRAMES_STACK_SIZE)     # Box(210,160,3)

        self.is_discrete_action_space = True
        self.n_actions = 6                                                          # Discrete(6)
        self.action_space = [i for i in range(self.n_actions)]

        self.GAMMA = 0.95  # 0.9 in PG tf.
        self.EPS_MIN = None

        self.memory_size = 5000
        self.memory_batch_size = 32

    def get_state(self, observation, prev_s):
        pre_processed_o = self.preprocess_image(observation)
        s = stack_frames(prev_s, pre_processed_o)
        return s

    def preprocess_image(self, o):
        o = o[15:200, 30:125]  # crop
        if self.image_channels == ATARI_IMAGE_CHANNELS_GRAYSCALE:  # adjust channels from RGB to Grayscale
            o = np.mean(o, axis=2)[:, :, np.newaxis]  # .reshape((*self.relevant_screen_size, 1))
        o = normalize(o)
        return o

    @staticmethod
    def update_reward(reward, done, info):
        """
        Penalize the agent for losing (0 number of lives).
        ALE is the emulator on which the open ai gym's Atari library is built.
        """
        if done and info['ale.lives'] == 0:
            return reward - 100

        return reward
