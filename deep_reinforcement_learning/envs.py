import numpy as np
import gym


# https://github.com/openai/gym/wiki/Table-of-environments


class BaseEnv:

    @staticmethod
    def get_state(observation, prev_s):
        return observation

    @staticmethod
    def update_reward(reward, done, info):
        return reward


class Envs:

    # input_dims == env.observation_space  # how many elements is the observation vector comprised of.

    # env.action_space -
    # is_discrete_action_space = isinstance(self.env.action_space, gym.spaces.discrete.Discrete)
    # n_actions == env.action_space.n
    # action_boundary == env.action_space.high  # needs to be positive so that we won't flip the actions.

    INPUT_TYPE_OBSERVATION_VECTOR = 0
    INPUT_TYPE_STACKED_FRAMES = 1

    WINDOW_UNKNOWN = 100

    # Observation types:
    #   Discrete
    #   Box - Continuous Range
    #   Pixels (which is usually a Box(0, 255, [height, width, 3]) for RGB pixels).

    # n_actions == action_space_size

    class ClassicControl:

        class CartPole(BaseEnv):

            # Solved: avg score >= 195.0 over 100 consecutive trials.

            def __init__(self):
                self.name = 'Cart Pole'
                self.file_name = 'cart-pole-v1'
                self.env = gym.make('CartPole-v1')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [4]   # Box(4,)
                self.is_discrete_action_space = True
                self.n_actions = 2      # Discrete(2)
                self.action_space = [i for i in range(self.n_actions)]
                self.memory_size = 1000000
                self.memory_batch_size = 64
                self.window = 100

        class Pendulum(BaseEnv):

            def __init__(self):
                self.name = 'Pendulum'
                self.file_name = 'pendulum-v0'
                self.env = gym.make('Pendulum-v0')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [3]   # Box(3,)
                self.is_discrete_action_space = False
                self.n_actions = 1      # Box(1,)
                self.action_boundary = 2
                self.memory_size = 1000000
                self.memory_batch_size = 64
                self.window = Envs.WINDOW_UNKNOWN

        class MountainCarContinuous(BaseEnv):

            def __init__(self):
                self.name = 'Mountain Car Continuous'
                self.file_name = 'mountain-car-continuous-v0'
                self.env = gym.make('MountainCarContinuous-v0')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [2]   # Box(2,)
                self.is_discrete_action_space = False
                self.n_actions = 1      # Box(1,)
                self.action_boundary = 1
                self.memory_size = 1000000
                self.memory_batch_size = 64
                self.window = Envs.WINDOW_UNKNOWN

    class Box2D:

        class LunarLander(BaseEnv):

            # Solved: avg score >= 200 over 100 consecutive trials.

            def __init__(self):
                self.name = 'Lunar Lander'
                self.file_name = 'lunar-lander-v2'
                self.env = gym.make('LunarLander-v2')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [8]   # Box(8,)
                self.is_discrete_action_space = True
                self.n_actions = 4      # Discrete(4)
                self.action_space = [i for i in range(self.n_actions)]
                self.memory_size = 1000000
                self.memory_batch_size = 64
                self.window = 100

        class LunarLanderContinuous(BaseEnv):

            # Solved: avg score >= 200 over 100 consecutive trials.

            def __init__(self):
                self.name = 'Lunar Lander Continuous'
                self.file_name = 'lunar-lander-continuous-v2'
                self.env = gym.make('LunarLanderContinuous-v2')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [8]   # Box(8,)
                self.is_discrete_action_space = False
                self.n_actions = 2      # Box(2,)
                self.action_boundary = [1, 1]
                self.memory_size = 1000000
                self.memory_batch_size = 64
                self.window = 100

        class BipedalWalker(BaseEnv):

            # Solved: avg score >= 300 over 100 consecutive trials.
            #   To solve the game you need to get 300 points in 1600 time steps.
            #   To solve hardcore version you need 300 points in 2000 time steps.

            # DDPG - 5K episodes take around 12h, and it takes around 15-20K to get score 255

            # State consists of:
            #   hull angle speed
            #   angular velocity
            #   horizontal speed
            #   vertical speed
            #   position of joints
            #   joints angular speed
            #   legs contact with ground
            #   10 lidar rangefinder measurements
            # There's no coordinates in the state vector.

            def __init__(self):
                self.name = 'Bipedal Walker'
                self.file_name = 'bipedal-walker-v2'
                self.env = gym.make('BipedalWalker-v2')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [24]      # Box(24,)
                self.is_discrete_action_space = False
                self.n_actions = 4          # Box(4,)
                self.action_boundary = [1, 1, 1, 1]
                self.memory_size = 1000000
                self.memory_batch_size = 64
                self.window = 100

    class Atari:

        # must be trained on a GPU. will basically take thousands of years on CPU.

        frames_stack_size = 4  # to give the agent a sense of motion, buffer_size

        IMAGE_CHANNELS_GRAYSCALE = 1
        IMAGE_CHANNELS_RGB = 3

        # observation's shape: (210, 160, 3)    # (H, W, C)
        # screen_size = (210, 160)              # (H, W)
        # image_channels = 3                    # RGB

        # observation pre-process # reshaping (usually)
        #   1. the atari screen should be truncated (cropped) - since there's no need for the score, etc...
        #   2. remove color by getting the mean of the 3 channels (axis=2 means along the RGB values)

        # input_type = Envs.INPUT_TYPE_STACKED_FRAMES

        @staticmethod
        def stack_frames(stacked_frames, frame):  # to get a sense of motion. observation == frame, prev s == stacked_frames
            buffer_size = Envs.Atari.frames_stack_size

            if stacked_frames is None:  # at the start of the episode
                stacked_frames = np.zeros((buffer_size, *frame.shape))
                for i, _ in enumerate(stacked_frames):
                    stacked_frames[i, :] = frame

            else:  # not the beginning of the episode
                stacked_frames = stacked_frames.reshape((buffer_size, *frame.shape))  # for being able to edit it
                stacked_frames[0:buffer_size - 1, :] = stacked_frames[1:, :]  # shift the set of frames down (-1 index), the bottom frame gets overwriten
                stacked_frames[buffer_size - 1, :] = frame  # append the current frame to the top (end of the array \ stack)

            stacked_frames = stacked_frames.reshape((*frame.shape[0:2], buffer_size))  # for being able to feed it to the NN

            return stacked_frames

        class Breakout(BaseEnv):

            def __init__(self):
                self.name = 'Breakout'
                self.file_name = 'breakout-v0'
                self.env = gym.make('Breakout-v0')
                self.GAMMA = 0.99
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_STACKED_FRAMES
                self.image_channels = Envs.Atari.IMAGE_CHANNELS_GRAYSCALE
                self.relevant_screen_size = (180, 160)
                self.input_dims = (*self.relevant_screen_size, Envs.Atari.frames_stack_size)    # Box(210,160,3)
                self.is_discrete_action_space = True
                self.n_actions = 3                                                              # Discrete(4)
                self.action_space = [1, 2, 3]
                self.memory_size = 6000  # saving transitions (stacked frames): 6-7K --> ~16Gb RAM, 25K --> ~48Gb RAM
                self.memory_batch_size = 32
                self.window = 10

            def get_state(self, observation, prev_s):
                observation_pre_processed = self.preprocess(observation)
                s = Envs.Atari.stack_frames(prev_s, observation_pre_processed)
                return s

            def preprocess(self, observation):
                cropped_observation = observation[30:, :]
                if self.image_channels == Envs.Atari.IMAGE_CHANNELS_RGB:
                    return cropped_observation
                else:
                    return np.mean(cropped_observation, axis=2)  # .reshape((*self.relevant_screen_size, 1))

        class SpaceInvaders(BaseEnv):

            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire

            def __init__(self):
                self.name = 'Space Invaders'
                self.file_name = 'space-invaders-v0'
                self.env = gym.make('SpaceInvaders-v0')
                self.GAMMA = 0.95  # 0.9 in PG tf.
                self.EPS_MIN = None
                self.input_type = Envs.INPUT_TYPE_STACKED_FRAMES
                self.image_channels = Envs.Atari.IMAGE_CHANNELS_GRAYSCALE
                self.relevant_screen_size = (185, 95)
                self.input_dims = (*self.relevant_screen_size, Envs.Atari.frames_stack_size)    # Box(210,160,3)
                self.is_discrete_action_space = True
                self.n_actions = 6                                                              # Discrete(6)
                self.action_space = [i for i in range(self.n_actions)]
                self.memory_size = 5000
                self.memory_batch_size = 32
                self.window = 10

            def get_state(self, observation, prev_s):
                observation_pre_processed = self.preprocess(observation)
                s = Envs.Atari.stack_frames(prev_s, observation_pre_processed)  # here it should be 3 frames?
                return s

            def preprocess(self, observation):
                cropped_observation = observation[15:200, 30:125]
                if self.image_channels == Envs.Atari.IMAGE_CHANNELS_RGB:
                    return cropped_observation
                else:
                    return np.mean(cropped_observation, axis=2)  # .reshape((*self.relevant_screen_size, 1))

            @staticmethod
            def update_reward(reward, done, info):
                # penalize the agent for losing (0 number of lives):
                #   ALE is the emulator on which the open ai gym's Atari library is built.
                if done and info['ale.lives'] == 0:
                    return reward - 100

                return reward
