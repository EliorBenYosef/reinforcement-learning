import numpy as np

from utils import LIBRARY_TORCH


class ReplayBuffer(object):  # aka 'ReplayMemory'

    def __init__(self, custom_env, memory_size, lib_type, is_discrete_action_space=None):

        self.is_discrete_action_space = is_discrete_action_space if is_discrete_action_space is not None \
            else custom_env.is_discrete_action_space

        self.n_actions = custom_env.n_actions
        if self.is_discrete_action_space:
            self.action_space = custom_env.action_space
            # self.action_boundary = None
        else:
            self.action_space = None
            # self.action_boundary = custom_env.action_boundary

        self.memory_size = memory_size
        self.memory_counter = 0

        # init memory:
        # to handle the agent's memory, we can also use Deque
        #   (python data structure that emulates the behavior of a queue, where we can pop things in and out)
        # we're using a numpy array because we already know the max memory size,
        #   and we'll overwrite memories as we go along.

        input_dims = custom_env.input_dims

        if not self.is_discrete_action_space:
            self.dtype = np.float32  # pytorch
        else:
            # int8 means an 8 bit integer (up to 2^8 = 256). important for bool arrays (0\1)
            if lib_type == LIBRARY_TORCH:
                self.dtype = np.uint8  # pytorch only accepts unsigned (u) dtypes
            else:
                self.dtype = np.int8

        self.memory_s = np.zeros((self.memory_size, *input_dims))
        self.memory_s_ = np.zeros((self.memory_size, *input_dims))
        self.memory_r = np.zeros(self.memory_size)
        self.memory_terminal = np.zeros(self.memory_size, dtype=self.dtype)

        if self.is_discrete_action_space:
            self.memory_a_indices_one_hot = np.zeros((self.memory_size, self.n_actions), dtype=self.dtype)
        else:
            self.memory_a = np.zeros((self.memory_size, self.n_actions), dtype=self.dtype)

    def store_transition(self, s, a, r, s_, is_terminal):
        i = self.memory_counter % self.memory_size  # calculate the first available memory location\position

        self.memory_s[i] = s
        self.memory_s_[i] = s_
        self.memory_r[i] = r
        self.memory_terminal[i] = 1 - int(is_terminal)

        if self.is_discrete_action_space:
            # integer action --> one hot encoding of actions:
            #   necessary for multiply the tensor [memory_batch_size, n_actions] with a vector [n_actions]
            a_indices_one_hot = np.zeros(self.n_actions, dtype=self.dtype)
            a_index = self.action_space.index(a)
            a_indices_one_hot[a_index] = 1
            self.memory_a_indices_one_hot[i] = a_indices_one_hot
        else:
            self.memory_a[i] = a

        self.memory_counter += 1

    def sample_batch(self, memory_batch_size, sequential=False):
        # Batch Learning - sample a batch of the memories.

        max_mem = min(self.memory_counter, self.memory_size)

        if not sequential:
            # non-sequential random subset (important for robust learning):
            batch = np.random.choice(max_mem, memory_batch_size)
        else:
            # sequential subset:
            memory_start_index = int(np.random.choice(range(
                self.memory_counter if self.memory_counter + memory_batch_size < self.memory_size
                else self.memory_size - memory_batch_size - 1
            )))
            batch = np.array(memory_start_index, memory_start_index + memory_batch_size)

        batch_s = self.memory_s[batch]
        batch_s_ = self.memory_s_[batch]
        batch_r = self.memory_r[batch]
        batch_terminal = self.memory_terminal[batch]

        if self.is_discrete_action_space:
            batch_a_indices_one_hot = self.memory_a_indices_one_hot[batch]

            # simplest way to convert: multiplying the vector with the matrix \ vector.
            # a_values = np.array(self.action_space, dtype=self.dtype)
            # batch_a_values = np.dot(batch_a_indices_one_hot, a_values)  # one hot encoding of actions --> integer action
            a_indices = np.array([i for i in range(self.n_actions)], dtype=self.dtype)
            batch_a_indices = np.dot(batch_a_indices_one_hot, a_indices)  # one hot encoding of actions --> action's index

            return batch_s, batch_s_, batch_r, batch_terminal, batch_a_indices_one_hot, batch_a_indices,
        else:
            batch_a = self.memory_a[batch]

            return batch_s, batch_s_, batch_r, batch_terminal, batch_a
