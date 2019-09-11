from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import os
from gym import wrappers
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import random_uniform_initializer as random_uniform

import torch as T
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.rmsprop as optim_rmsprop

import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers

import utils
from deep_reinforcement_learning.envs import Envs


class DNN(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, alpha):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.n_actions = custom_env.n_actions

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.chkpt_dir = 'tmp/' + custom_env.file_name + '/PG/NNs'

    def create_dnn_tensorflow(self, name):
        return DNN.DNN_TensorFlow(self, name)

    def create_dnn_torch(self, relevant_screen_size, image_channels):
        return DNN.DNN_Torch(self, relevant_screen_size, image_channels)

    class DNN_TensorFlow(object):

        def __init__(self, dnn, name, device_type=None):
            self.dnn = dnn

            self.name = name

            self.sess = utils.get_tf_session_according_to_device_type(device_type)
            self.build_network()
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.checkpoint_file = os.path.join(dnn.chkpt_dir, 'dnn_tf.ckpt')

            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        def build_network(self):
            with tf.variable_scope(self.name):
                self.s = tf.placeholder(tf.float32, shape=[None, *self.dnn.input_dims], name='s')
                self.a_index = tf.placeholder(tf.int32, shape=[None], name='a_index')
                self.G = tf.placeholder(tf.float32, shape=[None], name='G')

                if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                    fc1_activated = tf.layers.dense(inputs=self.s, units=self.dnn.fc_layers_dims[0],
                                                    activation='relu',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                    fc2_activated = tf.layers.dense(inputs=fc1_activated, units=self.dnn.fc_layers_dims[1],
                                                    activation='relu',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                    fc_last = tf.layers.dense(inputs=fc2_activated, units=self.dnn.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())

                else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                    conv1 = tf.layers.conv2d(inputs=self.s, filters=32,
                                             kernel_size=(8, 8), strides=4, name='conv1',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                    conv1_bn = tf.layers.batch_normalization(inputs=conv1, epsilon=1e-5, name='conv1_bn')
                    conv1_bn_activated = tf.nn.relu(conv1_bn)
                    conv2 = tf.layers.conv2d(inputs=conv1_bn_activated, filters=64,
                                             kernel_size=(4, 4), strides=2, name='conv2',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                    conv2_bn = tf.layers.batch_normalization(inputs=conv2, epsilon=1e-5, name='conv2_bn')
                    conv2_bn_activated = tf.nn.relu(conv2_bn)
                    conv3 = tf.layers.conv2d(inputs=conv2_bn_activated, filters=128,
                                             kernel_size=(3, 3), strides=1, name='conv3',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                    conv3_bn = tf.layers.batch_normalization(inputs=conv3, epsilon=1e-5, name='conv3_bn')
                    conv3_bn_activated = tf.nn.relu(conv3_bn)

                    flat = tf.layers.flatten(conv3_bn_activated)
                    fc1_activated = tf.layers.dense(inputs=flat, units=self.dnn.fc_layers_dims[0],
                                                    activation='relu',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
                    fc_last = tf.layers.dense(inputs=fc1_activated, units=self.dnn.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())

                self.actions_probabilities = tf.nn.softmax(fc_last, name='actions_probabilities')

                negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=fc_last, labels=self.a_index
                )
                loss = negative_log_probability * self.G
                if self.dnn.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                    loss = tf.reduce_mean(loss)

                if self.dnn.optimizer_type == utils.OPTIMIZER_SGD:
                    optimizer = tf.train.MomentumOptimizer(self.dnn.ALPHA, 0.9)  # SGD + momentum
                    # optimizer = tf.train.GradientDescentOptimizer(self.dqn.ALPHA)  # SGD?
                elif self.dnn.optimizer_type == utils.OPTIMIZER_RMSprop:
                    optimizer = tf.train.RMSPropOptimizer(self.dnn.ALPHA, decay=0.99, momentum=0.0, epsilon=1e-6)
                else:  # self.dqn.optimizer_type == utils.OPTIMIZER_Adam
                    optimizer = tf.train.AdamOptimizer(self.dnn.ALPHA)

                self.optimize = optimizer.minimize(loss)  # train_op

        def get_actions_probabilities(self, batch_s):
            return self.sess.run(self.actions_probabilities, feed_dict={self.s: batch_s})[0]

        def learn_entire_batch(self, memory, GAMMA):
            memory_s = np.array(memory.memory_s)
            memory_a_index = np.array(memory.memory_a_index)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.int8)

            memory_G = utils.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)

            print('Training Started')
            _ = self.sess.run(self.optimize,
                              feed_dict={self.s: memory_s,
                                         self.a_index: memory_a_index,
                                         self.G: memory_G})
            print('Training Finished')

        def load_model_file(self):
            print("...Loading tf checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving tf checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class DNN_Torch(nn.Module):

        def __init__(self, dnn, relevant_screen_size, image_channels, device_type='cuda'):

            super(DNN.DNN_Torch, self).__init__()

            self.dnn = dnn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(dnn.chkpt_dir, 'dnn_torch')

            self.build_network()

            if self.dnn.optimizer_type == utils.OPTIMIZER_SGD:
                self.optimizer = optim.SGD(self.parameters(), lr=self.dnn.ALPHA, momentum=0.9)
            elif self.dnn.optimizer_type == utils.OPTIMIZER_RMSprop:
                self.optimizer = optim_rmsprop.RMSprop(self.parameters(), lr=self.dnn.ALPHA)
            else:  # self.dnn.optimizer_type == utils.OPTIMIZER_Adam
                self.optimizer = optim.Adam(self.parameters(), lr=self.dnn.ALPHA)

            self.device = utils.get_torch_device_according_to_device_type(device_type)
            self.to(self.device)

        def build_network(self):
            if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                self.fc1 = nn.Linear(*self.dnn.input_dims, self.dnn.fc_layers_dims[0])
                self.fc2 = nn.Linear(self.dnn.fc_layers_dims[0], self.dnn.fc_layers_dims[1])
                self.fc3 = nn.Linear(self.dnn.fc_layers_dims[1], self.dnn.n_actions)

            else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                frames_stack_size = Envs.Atari.frames_stack_size
                self.in_channels = frames_stack_size * self.image_channels

                conv1_filters, conv2_filters, conv3_filters = 32, 64, 128
                conv1_fps = 8, 1, 4
                conv2_fps = 4, 0, 2
                conv3_fps = 3, 0, 1

                self.conv1 = nn.Conv2d(self.in_channels, conv1_filters, conv1_fps[0],
                                       padding=conv1_fps[1], stride=conv1_fps[2])
                self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, conv2_fps[0],
                                       padding=conv2_fps[1], stride=conv2_fps[2])
                self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, conv3_fps[0],
                                       padding=conv3_fps[1], stride=conv3_fps[2])

                i_H, i_W = self.dnn.input_dims[0], self.dnn.input_dims[1]
                conv1_o_H, conv1_o_W = utils.calc_conv_layer_output_dims(i_H, i_W, *conv1_fps)
                conv2_o_H, conv2_o_W = utils.calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fps)
                conv3_o_H, conv3_o_W = utils.calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fps)
                self.flat_dims = conv3_filters * conv3_o_H * conv3_o_W

                self.fc1 = nn.Linear(self.flat_dims, self.dnn.fc_layers_dims[0])
                self.fc2 = nn.Linear(self.dnn.fc_layers_dims[0], self.dnn.n_actions)

        def forward(self, s):
            input = T.tensor(s, dtype=T.float).to(self.device)

            if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:

                fc1_activated = F.relu(self.fc1(input))
                fc2_activated = F.relu(self.fc2(fc1_activated))
                fc_last = self.fc3(fc2_activated)

            else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES

                input = input.view(-1, self.in_channels, *self.relevant_screen_size)
                conv1_activated = F.relu(self.conv1(input))
                conv2_activated = F.relu(self.conv2(conv1_activated))
                conv3_activated = F.relu(self.conv3(conv2_activated))
                flat = conv3_activated.view(-1, self.flat_dims).to(self.device)
                fc1_activated = F.relu(self.fc1(flat))
                fc_last = self.fc2(fc1_activated)

            actions_probabilities = F.softmax(fc_last).to(self.device)

            return actions_probabilities

        def learn_entire_batch(self, memory, GAMMA):
            memory_a_log_probs = np.array(memory.memory_a_log_probs)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.uint8)

            memory_G = utils.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)
            memory_G = T.tensor(memory_G, dtype=T.float).to(self.device)

            self.optimizer.zero_grad()
            loss = 0
            for g, logprob in zip(memory_G, memory_a_log_probs):
                loss += -g * logprob
            loss.backward()
            print('Training Started')
            self.optimizer.step()
            print('Training Finished')

        def load_model_file(self):
            print("...Loading torch model...")
            self.load_state_dict(T.load(self.model_file))

        def save_model_file(self):
            print("...Saving torch model...")
            T.save(self.state_dict(), self.model_file)


class Memory(object):

    def __init__(self, custom_env, lib_type):

        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space

        self.lib_type = lib_type

        if self.lib_type == utils.LIBRARY_TF:
            self.memory_s = []
            self.memory_a_index = []
        else:  # self.lib_type == LIBRARY_TORCH
            self.memory_a_log_probs = []

        self.memory_r = []
        self.memory_terminal = []

    def store_transition(self, s, a, r, is_terminal):
        if self.lib_type == utils.LIBRARY_TF:
            self.memory_s.append(s)
            self.memory_a_index.append(self.action_space.index(a))

        self.memory_r.append(r)
        self.memory_terminal.append(int(is_terminal))

    def store_a_log_probs(self, a_log_probs):
        if self.lib_type == utils.LIBRARY_TORCH:
            self.memory_a_log_probs.append(a_log_probs)

    def reset_memory(self):
        if self.lib_type == utils.LIBRARY_TF:
            self.memory_s = []
            self.memory_a_index = []
        else:
            self.memory_a_log_probs = []

        self.memory_r = []
        self.memory_terminal = []


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims, alpha, optimizer_type=utils.OPTIMIZER_Adam, lib_type=utils.LIBRARY_TF):

        self.GAMMA = custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.action_space = custom_env.action_space

        self.lib_type = lib_type

        self.memory = Memory(custom_env, lib_type)

        self.policy_dnn = self.init_network(custom_env)

    def init_network(self, custom_env):
        dnn_base = DNN(custom_env, self.fc_layers_dims, self.optimizer_type, self.ALPHA)

        if self.lib_type == utils.LIBRARY_TF:
            dnn = dnn_base.create_dnn_tensorflow(name='q_policy')

        else:  # self.lib_type == LIBRARY_TORCH
            if custom_env.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                relevant_screen_size = custom_env.relevant_screen_size
                image_channels = custom_env.image_channels
            else:
                relevant_screen_size = None
                image_channels = None

            dnn = dnn_base.create_dnn_torch(relevant_screen_size, image_channels)

        return dnn

    def store_transition(self, s, a, r, is_terminal):
        self.memory.store_transition(s, a, r, is_terminal)

    def choose_action(self, s):
        s = s[np.newaxis, :]

        if self.lib_type == utils.LIBRARY_TF:
            probabilities = self.policy_dnn.get_actions_probabilities(s)
            a = np.random.choice(self.action_space, p=probabilities)

        else:  # self.lib_type == LIBRARY_TORCH
            probabilities = self.policy_dnn.forward(s)
            actions_probs = distributions.Categorical(probabilities)
            action_tensor = actions_probs.sample()
            a_log_probs = actions_probs.log_prob(action_tensor)
            self.memory.store_a_log_probs(a_log_probs)
            a_index = action_tensor.item()
            a = self.action_space[a_index]

        return a

    def learn(self):
        print('Learning Session')

        self.policy_dnn.learn_entire_batch(self.memory, self.GAMMA)
        self.memory.reset_memory()

    def save_models(self):
        self.policy_dnn.save_model_file()

    def load_models(self):
        self.policy_dnn.load_model_file()


def train(custom_env, agent, n_episodes, enable_models_saving, ep_batch_num=1):
    env = custom_env.env

    # uncomment the line below to record every episode.
    # env = wrappers.Monitor(env, 'tmp/' + custom_env.file_name + '/PG/recordings',
    #                        video_callable=lambda episode_id: True, force=True)

    print('\n', 'Game Started', '\n')

    scores_history = []

    for i in range(n_episodes):
        done = False
        ep_score = 0

        observation = env.reset()
        s = custom_env.get_state(observation, None)
        while not done:
            a = agent.choose_action(s)
            observation_, r, done, info = env.step(a)
            r = custom_env.update_reward(r, done, info)
            s_ = custom_env.get_state(observation_, s.copy())
            ep_score += r
            agent.store_transition(s, a, r, done)
            observation, s = observation_, s_
        scores_history.append(ep_score)

        avg_num = custom_env.window
        if ep_batch_num > avg_num:
            avg_num = ep_batch_num
        utils.print_training_progress(i, ep_score, scores_history, avg_num)

        if (i + 1) % ep_batch_num == 0:
            agent.learn()
            if enable_models_saving:
                agent.save_models()

    print('\n', 'Game Ended', '\n')

    utils.plot_running_average(
        custom_env.name, scores_history, window=custom_env.window, show=False,
        file_name=utils.get_plot_file_name(custom_env.file_name, agent, memory=True)
    )


def play(env_type, lib_type=utils.LIBRARY_TF, enable_models_saving=False, load_checkpoint=False):
    if lib_type == utils.LIBRARY_KERAS:
        print('\n', "Algorithm currently doesn't work with Keras", '\n')
        return

    if env_type == 0:
        # custom_env = Envs.Box2D.LunarLander()
        custom_env = Envs.ClassicControl.CartPole()
        optimizer_type = utils.OPTIMIZER_Adam
        alpha = 0.0005 if lib_type == utils.LIBRARY_TF else 0.001
        fc_layers_dims = [64, 64] if lib_type == utils.LIBRARY_TF else [128, 128]
        ep_batch_num = 1  # REINFORCE algorithm (MC PG)
        n_episodes = 2500  # supposed to be enough for good results in PG
    elif env_type == 1:
        custom_env = Envs.Atari.Breakout()
        optimizer_type = utils.OPTIMIZER_RMSprop  # utils.OPTIMIZER_SGD
        alpha = 0.00025
        fc_layers_dims = [256]
        ep_batch_num = 1  # REINFORCE algorithm (MC PG)
        n_episodes = 200  # start with 200, then 5000 ?
    else:
        custom_env = Envs.Atari.SpaceInvaders()
        optimizer_type = utils.OPTIMIZER_RMSprop  # utils.OPTIMIZER_SGD
        alpha = 0.001  # 0.003
        fc_layers_dims = [256]
        ep_batch_num = 10
        n_episodes = 1000

    if not custom_env.is_discrete_action_space:
        print('\n', "Environment's Action Space should be discrete!", '\n')
        return

    custom_env.env.seed(28)

    agent = Agent(custom_env, fc_layers_dims, alpha, optimizer_type=optimizer_type, lib_type=lib_type)

    if enable_models_saving and load_checkpoint:
        agent.load_models()

    train(custom_env, agent, n_episodes, enable_models_saving, ep_batch_num)


if __name__ == '__main__':
    play(0, lib_type=utils.LIBRARY_TF)          # CartPole (0), Breakout (1), SpaceInvaders (2)
