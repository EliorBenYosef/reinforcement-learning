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

    INPUT_TYPE_OBSERVATION_VECTOR = 0
    INPUT_TYPE_STACKED_FRAMES = 1

    WINDOW_UNKNOWN = 100

    class ClassicControl:

        class CartPole(BaseEnv):

            # if you check gym/gym/envs/__init__.py :
            # CartPole-v0 (rows 53-58): max_episode_steps = 200, reward_threshold = 195.0
            # CartPole-v1 (rows 60-65): max_episode_steps = 500, reward_threshold = 475.0

            def __init__(self):
                self.name = 'Cart Pole'
                self.file_name = 'cart-pole-v1'
                self.env = gym.make('CartPole-v1')

                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [4]   # Box(4,)

                self.is_discrete_action_space = True
                self.n_actions = 2      # Discrete(2)
                self.action_space = [i for i in range(self.n_actions)]

                self.window = 100

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 1000000
                self.memory_batch_size = 64

        class Pendulum(BaseEnv):

            # Continuous observation space.
            # Continuous action space (1 dimension).

            def __init__(self):
                self.name = 'Pendulum'
                self.file_name = 'pendulum-v0'
                self.env = gym.make('Pendulum-v0')

                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [3]   # Box(3,)

                self.is_discrete_action_space = False
                self.n_actions = 1      # Box(1,)
                self.action_boundary = 2

                self.window = Envs.WINDOW_UNKNOWN

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 1000000
                self.memory_batch_size = 64

        class MountainCarContinuous(BaseEnv):

            def __init__(self):
                self.name = 'Mountain Car Continuous'
                self.file_name = 'mountain-car-continuous-v0'
                self.env = gym.make('MountainCarContinuous-v0')

                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [2]   # Box(2,)

                self.is_discrete_action_space = False
                self.n_actions = 1      # Box(1,)
                self.action_boundary = 1

                self.window = Envs.WINDOW_UNKNOWN

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 1000000
                self.memory_batch_size = 64

    class Box2D:

        class LunarLander(BaseEnv):

            # Solved: avg score >= 200 over 100 consecutive trials.

            def __init__(self):
                self.name = 'Lunar Lander'
                self.file_name = 'lunar-lander-v2'
                self.env = gym.make('LunarLander-v2')

                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [8]   # Box(8,)

                self.is_discrete_action_space = True
                self.n_actions = 4      # Discrete(4)
                self.action_space = [i for i in range(self.n_actions)]

                self.window = 100

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 1000000
                self.memory_batch_size = 64

        class LunarLanderContinuous(BaseEnv):

            # Solved: avg score >= 200 over 100 consecutive trials.

            def __init__(self):
                self.name = 'Lunar Lander Continuous'
                self.file_name = 'lunar-lander-continuous-v2'
                self.env = gym.make('LunarLanderContinuous-v2')

                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [8]   # Box(8,)

                self.is_discrete_action_space = False
                self.n_actions = 2      # Box(2,)
                self.action_boundary = [1, 1]

                self.window = 100

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 1000000
                self.memory_batch_size = 64

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

                self.input_type = Envs.INPUT_TYPE_OBSERVATION_VECTOR
                self.input_dims = [24]      # Box(24,)

                self.is_discrete_action_space = False
                self.n_actions = 4          # Box(4,)
                self.action_boundary = [1, 1, 1, 1]

                self.window = 100

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 1000000
                self.memory_batch_size = 64

    class Atari:

        # Atari environments must be trained on a GPU (will basically take thousands of years on CPU).

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
            if stacked_frames is None:  # start of the episode: duplicate frame
                return np.repeat(frame, repeats=Envs.Atari.frames_stack_size, axis=2)
            else:  # remove first frame, and add frame to the end
                return np.concatenate((stacked_frames[:, :, 1:], frame), axis=2)

        class Breakout(BaseEnv):

            def __init__(self):
                self.name = 'Breakout'
                self.file_name = 'breakout-v0'
                self.env = gym.make('Breakout-v0')

                self.input_type = Envs.INPUT_TYPE_STACKED_FRAMES
                self.image_channels = Envs.Atari.IMAGE_CHANNELS_GRAYSCALE
                self.relevant_screen_size = (180, 160)
                self.input_dims = (*self.relevant_screen_size, Envs.Atari.frames_stack_size)    # Box(210,160,3)

                self.is_discrete_action_space = True
                self.n_actions = 3                                                              # Discrete(4)
                self.action_space = [1, 2, 3]

                self.window = 10

                self.GAMMA = 0.99
                self.EPS_MIN = None

                self.memory_size = 6000  # saving transitions (stacked frames): 6-7K --> ~16Gb RAM, 25K --> ~48Gb RAM
                self.memory_batch_size = 32

            def get_state(self, observation, prev_s):
                observation_pre_processed = self.preprocess(observation)
                s = Envs.Atari.stack_frames(prev_s, observation_pre_processed)
                return s

            def preprocess(self, observation):
                cropped_observation = observation[30:, :]
                if self.image_channels == Envs.Atari.IMAGE_CHANNELS_RGB:
                    return cropped_observation
                else:
                    return np.mean(cropped_observation, axis=2)[:, :, np.newaxis]  # .reshape((*self.relevant_screen_size, 1))

        class SpaceInvaders(BaseEnv):

            # Actions (Discrete 6): no action (0), fire (1),
            #                       move right (2), move left (3),
            #                       move right fire (4), move left fire (5)

            def __init__(self):
                self.name = 'Space Invaders'
                self.file_name = 'space-invaders-v0'
                self.env = gym.make('SpaceInvaders-v0')

                self.input_type = Envs.INPUT_TYPE_STACKED_FRAMES
                self.image_channels = Envs.Atari.IMAGE_CHANNELS_GRAYSCALE
                self.relevant_screen_size = (185, 95)
                self.input_dims = (*self.relevant_screen_size, Envs.Atari.frames_stack_size)    # Box(210,160,3)

                self.is_discrete_action_space = True
                self.n_actions = 6                                                              # Discrete(6)
                self.action_space = [i for i in range(self.n_actions)]

                self.window = 10

                self.GAMMA = 0.95  # 0.9 in PG tf.
                self.EPS_MIN = None

                self.memory_size = 5000
                self.memory_batch_size = 32

            def get_state(self, observation, prev_s):
                observation_pre_processed = self.preprocess(observation)
                s = Envs.Atari.stack_frames(prev_s, observation_pre_processed)  # here it should be 3 frames?
                return s

            def preprocess(self, observation):
                cropped_observation = observation[15:200, 30:125]
                if self.image_channels == Envs.Atari.IMAGE_CHANNELS_RGB:
                    return cropped_observation
                else:
                    return np.mean(cropped_observation, axis=2)[:, :, np.newaxis]  # .reshape((*self.relevant_screen_size, 1))

            @staticmethod
            def update_reward(reward, done, info):
                # penalize the agent for losing (0 number of lives):
                #   ALE is the emulator on which the open ai gym's Atari library is built.
                if done and info['ale.lives'] == 0:
                    return reward - 100

                return reward
