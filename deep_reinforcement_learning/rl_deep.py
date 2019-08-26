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

from utils import Utils
from utils import LIBRARY_KERAS, LIBRARY_TF, LIBRARY_TORCH
from deep_reinforcement_learning.envs import Envs
from deep_reinforcement_learning.replay_buffer import ReplayBuffer


class DQL:

    class DeepQNetwork(object):

        def __init__(self, alpha, fc_layers_dims, custom_env):
            self.ALPHA = alpha

            self.input_type = custom_env.input_type

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            self.chkpt_dir = 'tmp/' + custom_env.file_name + '/DQL/NNs'

        def create_dqn_tensorflow(self, name):
            return DQL.DeepQNetwork.DQN_TensorFlow(self, name)

        def create_dqn_torch(self, relevant_screen_size, image_channels):
            return DQL.DeepQNetwork.DQN_Torch(self, relevant_screen_size, image_channels)

        def create_dqn_keras(self):
            return DQL.DeepQNetwork.DQN_Keras(self)

        class DQN_TensorFlow(object):

            def __init__(self, dqn, name, device_type=None):
                self.dqn = dqn

                self.name = name

                self.sess = Utils.get_tf_session_according_to_device_type(device_type)
                self.build_network()
                self.sess.run(tf.global_variables_initializer())

                self.saver = tf.train.Saver()
                self.checkpoint_file = os.path.join(dqn.chkpt_dir, 'dqn_tf.ckpt')

                self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            def build_network(self):
                with tf.variable_scope(self.name):
                    self.s = tf.placeholder(tf.float32, shape=[None, *self.dqn.input_dims], name='s')
                    self.a_indices_one_hot = tf.placeholder(tf.float32, shape=[None, self.dqn.n_actions], name='a_indices_one_hot')
                    self.q_target = tf.placeholder(tf.float32, shape=[None, self.dqn.n_actions], name='q_target')

                    if self.dqn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                        fc1 = tf.layers.dense(inputs=self.s, units=self.dqn.fc_layers_dims[0],
                                              activation='relu')
                        fc2 = tf.layers.dense(inputs=fc1, units=self.dqn.fc_layers_dims[1],
                                              activation='relu')
                        self.q = tf.layers.dense(inputs=fc2, units=self.dqn.n_actions)

                    else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                        conv1 = tf.layers.conv2d(inputs=self.s, filters=32,
                                                 kernel_size=(8, 8), strides=4, name='conv1',
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
                        conv1_activated = tf.nn.relu(conv1)
                        conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                                 kernel_size=(4, 4), strides=2, name='conv2',
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
                        conv2_activated = tf.nn.relu(conv2)
                        conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128,
                                                 kernel_size=(3, 3), strides=1, name='conv3',
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
                        conv3_activated = tf.nn.relu(conv3)
                        flat = tf.layers.flatten(conv3_activated)
                        fc1 = tf.layers.dense(inputs=flat, units=self.dqn.fc_layers_dims[0],
                                              activation='relu',
                                              kernel_initializer=tf.variance_scaling_initializer(scale=2))
                        self.q = tf.layers.dense(inputs=fc1, units=self.dqn.n_actions,
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
                        # self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))  # the actual Q value for each action

                    self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))  # self.q - self.q_target
                    self.optimize = tf.train.AdamOptimizer(self.dqn.ALPHA).minimize(self.loss)  # train_op

            def forward(self, batch_s):
                q_eval_s = self.sess.run(self.q, feed_dict={self.s: batch_s})
                return q_eval_s

            def learn_batch(self, batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                            input_type, GAMMA, memory_batch_size, q_eval_s, q_eval_s_):

                q_target = q_eval_s.copy()
                batch_index = np.arange(memory_batch_size, dtype=np.int32)
                q_target[batch_index, batch_a_indices] = \
                    batch_r + GAMMA * np.max(q_eval_s_, axis=1) * batch_terminal

                # print('Training Started')
                _ = self.sess.run(self.optimize,
                                  feed_dict={self.s: batch_s,
                                             self.a_indices_one_hot: batch_a_indices_one_hot,
                                             self.q_target: q_target})
                # print('Training Finished')

            def load_model_file(self):
                print("...Loading tf checkpoint...")
                self.saver.restore(self.sess, self.checkpoint_file)

            def save_model_file(self):
                print("...Saving tf checkpoint...")
                self.saver.save(self.sess, self.checkpoint_file)

        class DQN_Torch(nn.Module):

            def __init__(self, dqn, relevant_screen_size, image_channels, device_type='cuda'):

                super(DQL.DeepQNetwork.DQN_Torch, self).__init__()

                self.dqn = dqn
                self.relevant_screen_size = relevant_screen_size
                self.image_channels = image_channels

                self.model_file = os.path.join(dqn.chkpt_dir, 'dqn_torch')

                self.build_network()

                if self.dqn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                    self.optimizer = optim.Adam(self.parameters(), lr=self.dqn.ALPHA)

                else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                    # self.optimizer = optim.SGD(self.parameters(), lr=self.dqn.ALPHA, momentum=0.9)
                    self.optimizer = optim_rmsprop.RMSprop(self.parameters(), lr=self.dqn.ALPHA)

                self.loss = nn.MSELoss()

                self.device = Utils.get_torch_device_according_to_device_type(device_type)
                self.to(self.device)

            def build_network(self):
                if self.dqn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                    self.fc1 = nn.Linear(*self.dqn.input_dims, self.dqn.fc_layers_dims[0])
                    self.fc2 = nn.Linear(self.dqn.fc_layers_dims[0], self.dqn.fc_layers_dims[1])
                    self.fc3 = nn.Linear(self.dqn.fc_layers_dims[1], self.dqn.n_actions)

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

                    i_H, i_W = self.dqn.input_dims[0], self.dqn.input_dims[1]
                    conv1_o_H, conv1_o_W = Utils.calc_conv_layer_output_dims(i_H, i_W, *conv1_fps)
                    conv2_o_H, conv2_o_W = Utils.calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fps)
                    conv3_o_H, conv3_o_W = Utils.calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fps)
                    self.flat_dims = conv3_filters * conv3_o_H * conv3_o_W

                    self.fc1 = nn.Linear(self.flat_dims, self.dqn.fc_layers_dims[0])
                    self.fc2 = nn.Linear(self.dqn.fc_layers_dims[0], self.dqn.n_actions)

            def forward(self, s):
                input = T.tensor(s, dtype=T.float).to(self.device)

                if self.dqn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:

                    fc1_activated = F.relu(self.fc1(input))
                    fc2_activated = F.relu(self.fc2(fc1_activated))
                    actions_q_values = self.fc3(fc2_activated)

                else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES

                    input = input.view(-1, self.in_channels, *self.relevant_screen_size)
                    conv1_activated = F.relu(self.conv1(input))
                    conv2_activated = F.relu(self.conv2(conv1_activated))
                    conv3_activated = F.relu(self.conv3(conv2_activated))
                    flat = conv3_activated.view(-1, self.flat_dims).to(self.device)
                    fc1_activated = F.relu(self.fc1(flat))
                    actions_q_values = self.fc2(fc1_activated)

                actions_q_values = actions_q_values.to(self.device)

                return actions_q_values

            def learn_batch(self, batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                            input_type, GAMMA, memory_batch_size, q_eval_s, q_eval_s_):
                batch_index = T.tensor(np.arange(memory_batch_size), dtype=T.long).to(self.device)
                batch_a_indices = T.tensor(batch_a_indices, dtype=T.long).to(self.device)
                batch_r = T.tensor(batch_r, dtype=T.float).to(self.device)
                batch_terminal = T.tensor(batch_terminal, dtype=T.float).to(self.device)

                q_target = self.forward(batch_s)  # there's no copy() in torch... need to feed-forward again.
                q_target[batch_index, batch_a_indices] = \
                    batch_r + GAMMA * T.max(q_eval_s_, dim=1)[0] * batch_terminal

                # print('Training Started')
                self.optimizer.zero_grad()
                loss = self.loss(q_target, q_eval_s).to(self.device)
                loss.backward()
                self.optimizer.step()
                # print('Training Finished')

            def load_model_file(self):
                print("...Loading torch model...")
                self.load_state_dict(T.load(self.model_file))

            def save_model_file(self):
                print("...Saving torch model...")
                T.save(self.state_dict(), self.model_file)

        class DQN_Keras(object):

            def __init__(self, dqn):
                self.dqn = dqn

                self.h5_file = os.path.join(dqn.chkpt_dir, 'dqn_keras.h5')

                if self.dqn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:

                    self.model = models.Sequential([
                        layers.Dense(self.dqn.fc_layers_dims[0], activation='relu', input_shape=self.dqn.input_dims),
                        layers.Dense(self.dqn.fc_layers_dims[1], activation='relu'),
                        layers.Dense(self.dqn.n_actions)])

                else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES

                    self.model = models.Sequential([
                        layers.Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu', input_shape=self.dqn.input_dims),
                        layers.Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu'),
                        layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
                        layers.Flatten(),
                        layers.Dense(self.dqn.fc_layers_dims[0], activation='relu'),
                        layers.Dense(self.dqn.n_actions)])

                self.model.compile(optimizer=optimizers.Adam(lr=self.dqn.ALPHA), loss='mse')

            def forward(self, batch_s):
                q_eval_s = self.model.predict(batch_s)
                return q_eval_s

            def learn_batch(self, batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                            input_type, GAMMA, memory_batch_size, q_eval_s, q_eval_s_):

                q_target = q_eval_s.copy()
                batch_index = np.arange(memory_batch_size, dtype=np.int32)
                q_target[batch_index, batch_a_indices] = \
                    batch_r + GAMMA * np.max(q_eval_s_, axis=1) * batch_terminal

                # print('Training Started')
                _ = self.model.fit(batch_s, q_target, verbose=0)
                # print('Training Finished')

            def load_model_file(self):
                print("...Loading keras h5...")
                self.model = models.load_model(self.h5_file)

            def save_model_file(self):
                print("...Saving keras h5...")
                self.model.save(self.h5_file)

    class Agent(object):

        def __init__(self, custom_env, alpha, fc_layers_dims,
                     eps_max=1.0, eps_min=0.01, eps_dec=0.996,
                     eps_dec_type=Utils.EPS_DEC_EXPONENTIAL, pure_exploration_phase=0,
                     double_dql=True, tau=10000,
                     lib_type=LIBRARY_TF):

            self.input_type = custom_env.input_type
            self.GAMMA = custom_env.GAMMA

            self.fc_layers_dims = fc_layers_dims
            self.ALPHA = alpha

            self.action_space = custom_env.action_space

            self.EPS = eps_max
            self.eps_max = eps_max
            self.eps_min = eps_min
            self.eps_dec = eps_dec
            self.eps_dec_type = eps_dec_type
            self.pure_exploration_phase = pure_exploration_phase

            self.lib_type = lib_type

            if self.lib_type == LIBRARY_TORCH:
                self.dtype = np.uint8
            else:  # TF \ Keras
                self.dtype = np.int8

            self.memory_batch_size = custom_env.memory_batch_size
            self.memory = ReplayBuffer(custom_env, lib_type, is_discrete_action_space=True)

            self.learn_step_counter = 0

            self.policy_dqn = self.init_network(custom_env, 'policy')

            if double_dql:
                self.target_dqn = self.init_network(custom_env, 'target')
                self.tau = tau
            else:
                self.target_dqn = None
                self.tau = None

        def init_network(self, custom_env, name):
            if self.lib_type == LIBRARY_TF:
                dqn = DQL.DeepQNetwork(
                    self.ALPHA, self.fc_layers_dims, custom_env
                ).create_dqn_tensorflow(name='q_' + name)

            elif self.lib_type == LIBRARY_TORCH:
                if custom_env.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                    relevant_screen_size = custom_env.relevant_screen_size
                    image_channels = custom_env.image_channels
                else:
                    relevant_screen_size = None
                    image_channels = None

                dqn = DQL.DeepQNetwork(
                    self.ALPHA, self.fc_layers_dims, custom_env
                ).create_dqn_torch(relevant_screen_size, image_channels)

            else:  # self.lib_type == LIBRARY_KERAS
                dqn = DQL.DeepQNetwork(
                    self.ALPHA, self.fc_layers_dims, custom_env
                ).create_dqn_keras()
            return dqn

        def store_transition(self, s, a, r, s_, done):
            self.memory.store_transition(s, a, r, s_, done)

        def choose_action(self, s):
            s = s[np.newaxis, :]

            actions_q_values = self.policy_dqn.forward(s)
            if self.lib_type == LIBRARY_TORCH:
                action_tensor = T.argmax(actions_q_values)
                a_index = action_tensor.item()
            else:  # TF \ Keras
                a_index = np.argmax(actions_q_values)
            a = self.action_space[a_index]

            rand = np.random.random()
            if rand < self.EPS:
                # pure exploration, no chance to get the greedy action
                modified_action_space = self.action_space.copy()
                modified_action_space.remove(a)
                a = np.random.choice(modified_action_space)

            return a

        # def choose_action(self, s):
        #     s = s[np.newaxis, :]
        #
        #     rand = np.random.random()
        #     if rand < self.EPS:
        #         # here there's also a chance to get the greedy action
        #         a = np.random.choice(self.action_space)
        #     else:
        #         actions_q_values = self.policy_dqn.forward(s)
        #         if self.lib_type == LIBRARY_TORCH:
        #             action_tensor = T.argmax(actions_q_values)
        #             a_index = action_tensor.item()
        #         else:  # TF \ Keras
        #             a_index = np.argmax(actions_q_values)
        #         a = self.action_space[a_index]
        #
        #     return a

        def learn_wrapper(self):
            if self.target_dqn is not None \
                    and self.tau \
                    and self.learn_step_counter % self.tau == 0:
                self.update_target_network()

            if self.memory.memory_counter >= self.memory_batch_size:
                self.learn()

        def update_target_network(self):
            if self.lib_type == LIBRARY_TF:
                target_network_params = self.target_dqn.params
                policy_network_params = self.policy_dqn.params
                for t_n_param, p_n_param in zip(target_network_params, policy_network_params):
                    self.policy_dqn.sess.run(tf.assign(t_n_param, p_n_param))

            elif self.lib_type == LIBRARY_TORCH:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            else:  # self.lib_type == LIBRARY_KERAS
                self.target_dqn.model.set_weights(self.policy_dqn.model.get_weights())

        def learn(self):
            # print('Learning Session')

            batch_s, batch_s_, batch_r, batch_terminal, batch_a_indices_one_hot, batch_a_indices = \
                self.memory.sample_batch(self.memory_batch_size)

            q_eval_s = self.policy_dqn.forward(batch_s)
            if self.target_dqn is not None:
                q_eval_s_ = self.target_dqn.forward(batch_s_)
            else:
                q_eval_s_ = self.policy_dqn.forward(batch_s_)

            self.policy_dqn.learn_batch(
                batch_s, batch_a_indices, batch_r, batch_terminal, batch_a_indices_one_hot,
                self.input_type, self.GAMMA, self.memory_batch_size, q_eval_s, q_eval_s_
            )

            self.learn_step_counter += 1

            if self.learn_step_counter > self.pure_exploration_phase:
                self.EPS = Utils.decrement_eps(self.EPS, self.eps_min, self.eps_dec, self.eps_dec_type)

        def save_models(self):
            self.policy_dqn.save_model_file()
            if self.target_dqn is not None:
                self.target_dqn.save_model_file()

        def load_models(self):
            self.policy_dqn.load_model_file()
            if self.target_dqn is not None:
                self.target_dqn.load_model_file()

    @staticmethod
    def load_up_agent_memory_with_random_gameplay(custom_env, agent, n_episodes=None):
        if n_episodes is None or n_episodes > custom_env.memory_size:
            n_episodes = custom_env.memory_size

        print('\n', "Loading up the agent's memory with random gameplay.", '\n')

        while agent.memory.memory_counter < n_episodes:
            done = False
            observation = custom_env.env.reset()
            s = custom_env.get_state(observation, None)
            while not done:
                a = np.random.choice(custom_env.action_space)
                observation_, r, done, info = custom_env.env.step(a)
                r = custom_env.update_reward(r, done, info)
                s_ = custom_env.get_state(observation_, s.copy())
                agent.store_transition(s, a, r, s_, done)
                observation, s = observation_, s_

        print('\n', "Done with random gameplay. Game on.", '\n')

    @staticmethod
    def train(custom_env, agent, n_episodes, enable_models_saving, save_checkpoint=10):
        env = custom_env.env

        # uncomment the line below to record every episode.
        # env = wrappers.Monitor(env, 'tmp/' + custom_env.file_name + '/DQL/recordings',
        #                        video_callable=lambda episode_id: True, force=True)

        print('\n', 'Game Started', '\n')

        scores_history = []
        eps_history = []

        for i in range(n_episodes):
            done = False
            ep_score = 0
            eps_history.append(agent.EPS)

            observation = env.reset()
            s = custom_env.get_state(observation, None)
            while not done:
                a = agent.choose_action(s)
                observation_, r, done, info = env.step(a)
                r = custom_env.update_reward(r, done, info)
                s_ = custom_env.get_state(observation_, s.copy())
                ep_score += r
                agent.store_transition(s, a, r, s_, done)
                agent.learn_wrapper()
                observation, s = observation_, s_

            scores_history.append(ep_score)

            Utils.print_training_progress(i, ep_score, scores_history, custom_env.window, agent.EPS)

            if enable_models_saving and (i + 1) % save_checkpoint == 0:
                agent.save_models()

        print('\n', 'Game Ended', '\n')

        Utils.plot_eps_history_and_running_avg(
            custom_env.name, scores_history, eps_history, window=custom_env.window, show=False,
            file_name=Utils.get_plot_file_name(custom_env, agent.fc_layers_dims, agent.ALPHA)
        )

    @staticmethod
    def play(env_type, lib_type=LIBRARY_TF, load_checkpoint=False, perform_random_gameplay=True):

        enable_models_saving = False

        if env_type == 0:
            # custom_env = Envs.Box2D.LunarLander()
            custom_env = Envs.ClassicControl.CartPole()
            alpha = 0.0005  # 0.003 ?
            fc_layers_dims = [256, 256]
            eps_min = 0.01
            # very aggressive epsilon decrement strategy:
            eps_dec = 0.996
            eps_dec_type = Utils.EPS_DEC_EXPONENTIAL
            pure_exploration_phase = 0
            double_dql = False
            tau = None
            n_episodes = 500

        elif env_type == 1:
            custom_env = Envs.Atari.Breakout()
            alpha = 0.00025
            fc_layers_dims = [1024]
            eps_min = 0.05  # 0.01
            eps_dec = 4e-7  # 0.9999999
            eps_dec_type = Utils.EPS_DEC_LINEAR  # EPS_DEC_EXPONENTIAL
            pure_exploration_phase = 6000  # 25000, 100000, 200000
            double_dql = True
            tau = 10000
            n_episodes = 200  # start with 200, then 5000 ?

        else:
            custom_env = Envs.Atari.SpaceInvaders()
            alpha = 0.003
            fc_layers_dims = [1024]
            eps_min = 0.05
            eps_dec = 1e-4
            eps_dec_type = Utils.EPS_DEC_LINEAR
            pure_exploration_phase = 500
            double_dql = True
            tau = None
            n_episodes = 50

        if not custom_env.is_discrete_action_space:
            print('\n', "Environment's Action Space should be discrete!", '\n')
            return

        agent = DQL.Agent(
            custom_env, alpha, fc_layers_dims,
            eps_min=eps_min, eps_dec=eps_dec, eps_dec_type=eps_dec_type,
            pure_exploration_phase=pure_exploration_phase,
            double_dql=double_dql, tau=tau, lib_type=lib_type
        )

        if enable_models_saving and load_checkpoint:
            agent.load_models()

        if perform_random_gameplay:
            # the agent's memory is originally initialized with zeros (which is perfectly acceptable).
            # however, we can overwrite these zeros with actual gameplay sampled from the environment.
            DQL.load_up_agent_memory_with_random_gameplay(custom_env, agent)

        DQL.train(custom_env, agent, n_episodes, enable_models_saving)


class PG:

    class DeepNeuralNetwork(object):

        def __init__(self, alpha, fc_layers_dims, custom_env):
            self.ALPHA = alpha

            self.input_type = custom_env.input_type

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            self.chkpt_dir = 'tmp/' + custom_env.file_name + '/PG/NNs'

        def create_dnn_tensorflow(self, name):
            return PG.DeepNeuralNetwork.DNN_TensorFlow(self, name)

        def create_dnn_torch(self, relevant_screen_size, image_channels):
            return PG.DeepNeuralNetwork.DNN_Torch(self, relevant_screen_size, image_channels)

        class DNN_TensorFlow(object):

            def __init__(self, dnn, name, device_type=None):
                self.dnn = dnn

                self.name = name

                self.sess = Utils.get_tf_session_according_to_device_type(device_type)
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

                    # train_op:
                    if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                        self.optimize = tf.train.AdamOptimizer(
                            self.dnn.ALPHA
                        ).minimize(loss)
                    else:
                        self.optimize = tf.train.RMSPropOptimizer(
                            self.dnn.ALPHA, decay=0.99, momentum=0.0, epsilon=1e-6
                        ).minimize(loss)

            def get_actions_probabilities(self, batch_s):
                return self.sess.run(self.actions_probabilities, feed_dict={self.s: batch_s})[0]

            def learn_entire_batch(self, memory, GAMMA):
                memory_s = np.array(memory.memory_s)
                memory_a_index = np.array(memory.memory_a_index)
                memory_r = np.array(memory.memory_r)
                memory_terminal = np.array(memory.memory_terminal, dtype=np.int8)

                memory_G = Utils.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)

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

                super(PG.DeepNeuralNetwork.DNN_Torch, self).__init__()

                self.dnn = dnn
                self.relevant_screen_size = relevant_screen_size
                self.image_channels = image_channels

                self.model_file = os.path.join(dnn.chkpt_dir, 'dnn_torch')

                self.build_network()

                if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                    self.optimizer = optim.Adam(self.parameters(), lr=self.dnn.ALPHA)
                else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                    # self.optimizer = optim.SGD(self.parameters(), lr=self.dnn.ALPHA, momentum=0.9)
                    self.optimizer = optim_rmsprop.RMSprop(self.parameters(), lr=self.dnn.ALPHA)

                self.device = Utils.get_torch_device_according_to_device_type(device_type)
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
                    conv1_o_H, conv1_o_W = Utils.calc_conv_layer_output_dims(i_H, i_W, *conv1_fps)
                    conv2_o_H, conv2_o_W = Utils.calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fps)
                    conv3_o_H, conv3_o_W = Utils.calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fps)
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

                memory_G = Utils.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)
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

            if self.lib_type == LIBRARY_TF:
                self.memory_s = []
                self.memory_a_index = []
            else:  # self.lib_type == LIBRARY_TORCH
                self.memory_a_log_probs = []

            self.memory_r = []
            self.memory_terminal = []

        def store_transition(self, s, a, r, is_terminal):
            if self.lib_type == LIBRARY_TF:
                self.memory_s.append(s)
                self.memory_a_index.append(self.action_space.index(a))

            self.memory_r.append(r)
            self.memory_terminal.append(int(is_terminal))

        def store_a_log_probs(self, a_log_probs):
            if self.lib_type == LIBRARY_TORCH:
                self.memory_a_log_probs.append(a_log_probs)

        def reset_memory(self):
            if self.lib_type == LIBRARY_TF:
                self.memory_s = []
                self.memory_a_index = []
            else:
                self.memory_a_log_probs = []

            self.memory_r = []
            self.memory_terminal = []

    class Agent(object):

        def __init__(self, custom_env, alpha, fc_layers_dims, lib_type=LIBRARY_TF):

            self.fc_layers_dims = fc_layers_dims
            self.ALPHA = alpha
            self.GAMMA = custom_env.GAMMA

            self.action_space = custom_env.action_space

            self.lib_type = lib_type

            self.memory = PG.Memory(custom_env, lib_type)

            self.policy_dnn = self.init_network(custom_env)

        def init_network(self, custom_env):
            if self.lib_type == LIBRARY_TF:
                dnn = PG.DeepNeuralNetwork(
                    self.ALPHA, self.fc_layers_dims, custom_env
                ).create_dnn_tensorflow(name='q_policy')

            else:  # self.lib_type == LIBRARY_TORCH
                if custom_env.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                    relevant_screen_size = custom_env.relevant_screen_size
                    image_channels = custom_env.image_channels
                else:
                    relevant_screen_size = None
                    image_channels = None

                dnn = PG.DeepNeuralNetwork(
                    self.ALPHA, self.fc_layers_dims, custom_env
                ).create_dnn_torch(relevant_screen_size, image_channels)

            return dnn

        def store_transition(self, s, a, r, is_terminal):
            self.memory.store_transition(s, a, r, is_terminal)

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.lib_type == LIBRARY_TF:
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

    @staticmethod
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
            Utils.print_training_progress(i, ep_score, scores_history, avg_num)

            if (i + 1) % ep_batch_num == 0:
                agent.learn()
                if enable_models_saving:
                    agent.save_models()

        print('\n', 'Game Ended', '\n')

        Utils.plot_running_average(
            custom_env.name, scores_history, window=custom_env.window, show=False,
            file_name=Utils.get_plot_file_name(custom_env, agent.fc_layers_dims, agent.ALPHA)
        )

    @staticmethod
    def play(env_type, lib_type=LIBRARY_TF, load_checkpoint=False):

        if lib_type == LIBRARY_KERAS:
            print('\n', "Algorithm currently doesn't work with Keras", '\n')
            return

        enable_models_saving = False

        if env_type == 0:
            # custom_env = Envs.Box2D.LunarLander()
            custom_env = Envs.ClassicControl.CartPole()
            ep_batch_num = 1  # REINFORCE algorithm (MC PG)
            n_episodes = 2500  # supposed to be enough for good results in PG
            if lib_type==LIBRARY_TF:
                alpha = 0.0005
                fc_layers_dims = [64, 64]
            else:
                alpha = 0.001
                fc_layers_dims = [128, 128]
        elif env_type == 1:
            custom_env = Envs.Atari.Breakout()
            ep_batch_num = 1  # REINFORCE algorithm (MC PG)
            alpha = 0.00025
            fc_layers_dims = [256]
            n_episodes = 200  # start with 200, then 5000 ?
        else:
            custom_env = Envs.Atari.SpaceInvaders()
            ep_batch_num = 10
            alpha = 0.001  # 0.003
            fc_layers_dims = [256]
            n_episodes = 1000

        if not custom_env.is_discrete_action_space:
            print('\n', "Environment's Action Space should be discrete!", '\n')
            return

        agent = PG.Agent(custom_env, alpha, fc_layers_dims, lib_type=lib_type)

        if enable_models_saving and load_checkpoint:
            agent.load_models()

        PG.train(custom_env, agent, n_episodes, enable_models_saving, ep_batch_num)


class AC:

    NETWORK_TYPE_SEPARATE = 0
    NETWORK_TYPE_SHARED = 1

    class AC(object):

        class AC_TF(object):

            class AC_DNN_TF(object):

                def __init__(self, custom_env, fc_layers_dims, sess, lr, name, network_type, is_actor=False):

                    self.name = name
                    self.lr = lr
                    self.is_actor = is_actor

                    self.network_type = network_type

                    self.input_dims = custom_env.input_dims
                    self.fc_layers_dims = fc_layers_dims
                    self.n_outputs = custom_env.n_actions if custom_env.is_discrete_action_space else 2

                    self.sess = sess

                    self.build_network()

                    self.params = tf.trainable_variables(scope=self.name)

                def build_network(self):
                    with tf.variable_scope(self.name):
                        self.s = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='s')
                        self.td_error = tf.placeholder(tf.float32, shape=[None, 1], name='td_error')
                        self.a_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='a_log_probs')
                        self.a = tf.placeholder(tf.int32, shape=[None, 1], name='a')

                        fc1_activated = tf.layers.dense(inputs=self.s, units=self.fc_layers_dims[0],
                                                        activation='relu')
                        fc2_activated = tf.layers.dense(inputs=fc1_activated, units=self.fc_layers_dims[1],
                                                        activation='relu')

                        if self.network_type == AC.NETWORK_TYPE_SEPARATE:  # build_A_or_C_network
                            self.fc3 = tf.layers.dense(inputs=fc2_activated, units=self.n_outputs if self.is_actor else 1)
                            loss = self.get_actor_loss() if self.is_actor else self.get_critic_loss()

                        else:  # self.network_type == AC.NETWORK_TYPE_SHARED  # build_A_and_C_network
                            self.fc3 = tf.layers.dense(inputs=fc2_activated, units=self.n_outputs)  # Actor layer
                            self.v = tf.layers.dense(inputs=fc2_activated, units=1)                 # Critic layer
                            loss = self.get_actor_loss() + self.get_critic_loss()

                        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(loss)  # train_op

                def get_actor_loss(self):
                    negative_a_log_probs = tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.fc3, labels=self.a  # discrete AS: a_index. labels are the actions indices
                    )
                    actor_loss = negative_a_log_probs * self.td_error
                    # if self.dnn.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                    #     loss = tf.reduce_mean(loss)
                    return actor_loss

                # def get_actor_loss(self):
                #     if True:
                #         probabilities = tf.nn.softmax(self.fc3)[0]
                #         actions_probs = tfp.distributions.Categorical(probs=probabilities)
                #         # action_tensor = actions_probs.sample()
                #     else:
                #         mu, sigma_unactivated = self.fc3  # Mean (μ), STD (σ)
                #         sigma = tf.exp(sigma_unactivated)
                #         actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
                #         # action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
                #
                #     a_log_probs = actions_probs.log_prob(self.a)
                #     # a_log_probs = tf.log(self.fc3[0, self.a])
                #     # # # advantage (TD_error) guided loss:
                #     # # actor_loss = a_log_probs * self.td_error
                #     # # self.exp_v = tf.reduce_mean(actor_loss)
                #
                #     actor_loss = -a_log_probs * self.td_error
                #
                #     return actor_loss

                def get_critic_loss(self):
                    critic_loss = self.td_error ** 2
                    return critic_loss

                def predict(self, s):
                    if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                        state_value = self.sess.run(self.fc3, feed_dict={self.s: s})
                        return state_value

                    else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                        state_value = self.sess.run(self.fc3, feed_dict={self.s: s})   # Actor value
                        v = self.sess.run(self.v, feed_dict={self.s: s})               # Critic value
                        return state_value, v

                def train(self, s, td_error, a=None):
                    print('Training Started')
                    self.sess.run(self.optimize,
                                  feed_dict={self.s: s,
                                             self.td_error: td_error,
                                             self.a: a})
                    print('Training Finished')

            def __init__(self, custom_env, lr_actor, lr_critic, fc_layers_dims, chkpt_dir, network_type, device_type):

                self.GAMMA = custom_env.GAMMA

                # self.a_log_probs = None

                self.is_discrete_action_space = custom_env.is_discrete_action_space
                self.n_actions = custom_env.n_actions
                self.action_space = custom_env.action_space if self.is_discrete_action_space else None
                self.action_boundary = custom_env.action_boundary if not self.is_discrete_action_space else None

                self.sess = Utils.get_tf_session_according_to_device_type(device_type)

                self.network_type = network_type
                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    self.actor = AC.AC.AC_TF.AC_DNN_TF(
                        custom_env, fc_layers_dims, self.sess, lr_actor, 'Actor', network_type, is_actor=True)
                    self.critic = AC.AC.AC_TF.AC_DNN_TF(
                        custom_env, fc_layers_dims, self.sess, lr_critic, 'Critic', network_type, is_actor=False)

                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    self.actor_critic = AC.AC.AC_TF.AC_DNN_TF(
                        custom_env, fc_layers_dims, self.sess, lr_actor, 'ActorCritic', network_type)

                self.saver = tf.train.Saver()
                self.checkpoint_file = os.path.join(chkpt_dir, 'ac_tf.ckpt')

                self.sess.run(tf.global_variables_initializer())

            def choose_action(self, s):
                s = s[np.newaxis, :]

                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    actor_value = self.actor.predict(s)
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    actor_value, _ = self.actor_critic.predict(s)

                if self.is_discrete_action_space:
                    return self.choose_action_discrete(actor_value)
                else:
                    return self.choose_action_continuous(actor_value)

            def choose_action_discrete(self, pi):
                probabilities = tf.nn.softmax(pi)[0]
                a_index = np.random.choice(self.action_space, p=probabilities)
                a = self.action_space[a_index]
                return a

            def choose_action_continuous(self, actor_value):
                mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
                sigma = tf.exp(sigma_unactivated)
                actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
                action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
                action_tensor = tf.nn.tanh(action_tensor)
                action_tensor = tf.multiply(action_tensor, self.action_boundary)
                a = action_tensor.item()
                a = np.array(a).reshape((1,))
                return a

            # def choose_action_discrete(self, pi):
            #     probabilities = tf.nn.softmax(pi)[0]
            #     actions_probs = tfp.distributions.Categorical(probs=probabilities)
            #     action_tensor = actions_probs.sample()
            #     self.a_log_probs = actions_probs.log_prob(action_tensor)
            #     a_index = action_tensor.item()
            #     a = self.action_space[a_index]
            #     return a
            #
            # def choose_action_continuous(self, actor_value):
            #     mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
            #     sigma = tf.exp(sigma_unactivated)
            #     actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
            #     action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
            #     self.a_log_probs = actions_probs.log_prob(action_tensor)
            #     action_tensor = tf.nn.tanh(action_tensor)
            #     action_tensor = tf.multiply(action_tensor, self.action_boundary)
            #     a = action_tensor.item()
            #     a = np.array(a).reshape((1,))
            #     return a

            def learn(self, s, a, r, s_, is_terminal):
                # print('Learning Session')

                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    v = self.critic.predict(s)
                    v_ = self.critic.predict(s_)
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    _, v = self.actor_critic.predict(s)
                    _, v_ = self.actor_critic.predict(s_)

                td_error = r + self.GAMMA * v_ * (1 - int(is_terminal)) - v

                if self.is_discrete_action_space:
                    a = self.action_space.index(a)

                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    self.actor.train(s, td_error, a)
                    self.critic.train(s, td_error)
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    self.actor_critic.train(s, td_error, a)

            def load_model_file(self):
                print("...Loading tf checkpoint...")
                self.saver.restore(self.sess, self.checkpoint_file)

            def save_model_file(self):
                print("...Saving tf checkpoint...")
                self.saver.save(self.sess, self.checkpoint_file)

        class AC_Torch(object):

            class AC_DNN_Torch(nn.Module):

                def __init__(self, custom_env, fc_layers_dims, lr, name, chkpt_dir, network_type, is_actor=False,
                             device_type='cuda'):
                    super(AC.AC.AC_Torch.AC_DNN_Torch, self).__init__()

                    self.name = name
                    self.lr = lr
                    self.is_actor = is_actor

                    self.network_type = network_type

                    self.model_file = os.path.join(chkpt_dir, 'ac_torch_' + name)

                    self.input_dims = custom_env.input_dims
                    self.fc_layers_dims = fc_layers_dims
                    self.n_outputs = custom_env.n_actions if custom_env.is_discrete_action_space else 2

                    self.build_network()

                    self.optimizer = optim.Adam(self.parameters(), lr=lr)

                    self.device = Utils.get_torch_device_according_to_device_type(device_type)
                    self.to(self.device)

                def load_model_file(self):
                    self.load_state_dict(T.load(self.model_file))

                def save_model_file(self):
                    T.save(self.state_dict(), self.model_file)

                def build_network(self):
                    self.fc1 = nn.Linear(*self.input_dims, self.fc_layers_dims[0])
                    self.fc2 = nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])

                    if self.network_type == AC.NETWORK_TYPE_SEPARATE:       # build_A_or_C_network
                        self.fc3 = nn.Linear(self.fc_layers_dims[1], self.n_outputs if self.is_actor else 1)

                    else:  # self.network_type == AC.NETWORK_TYPE_SHARED    # build_A_and_C_network
                        self.fc3 = nn.Linear(self.fc_layers_dims[1], self.n_outputs)    # Actor layer
                        self.v = nn.Linear(self.fc_layers_dims[1], 1)                   # Critic layer

                def forward(self, s):
                    state_value = T.tensor(s, dtype=T.float).to(self.device)

                    state_value = self.fc1(state_value)
                    state_value = F.relu(state_value)

                    state_value = self.fc2(state_value)
                    state_value = F.relu(state_value)

                    if self.network_type == AC.NETWORK_TYPE_SEPARATE:       # forward_A_or_C_network
                        state_value = self.fc3(state_value)
                        return state_value

                    else:  # self.network_type == AC.NETWORK_TYPE_SHARED    # forward_A_and_C_network
                        actor_value = self.fc3(state_value)     # Actor value
                        v = self.v(state_value)                 # Critic value
                        return actor_value, v

            def __init__(self, custom_env, lr_actor, lr_critic, fc_layers_dims, chkpt_dir, network_type):

                self.GAMMA = custom_env.GAMMA

                self.a_log_probs = None

                self.is_discrete_action_space = custom_env.is_discrete_action_space
                self.n_actions = custom_env.n_actions
                self.action_space = custom_env.action_space if self.is_discrete_action_space else None
                self.action_boundary = custom_env.action_boundary if not self.is_discrete_action_space else None

                self.network_type = network_type
                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    self.actor = AC.AC.AC_Torch.AC_DNN_Torch(
                        custom_env, fc_layers_dims, lr_actor, 'Actor', chkpt_dir, network_type, is_actor=True)
                    self.critic = AC.AC.AC_Torch.AC_DNN_Torch(
                        custom_env, fc_layers_dims, lr_critic, 'Critic', chkpt_dir, network_type, is_actor=False)

                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    self.actor_critic = AC.AC.AC_Torch.AC_DNN_Torch(
                        custom_env, fc_layers_dims, lr_actor, 'ActorCritic', chkpt_dir, network_type)

            def choose_action(self, s):
                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    actor_value = self.actor.forward(s)
                    device = self.actor.device
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    actor_value, _ = self.actor_critic.forward(s)
                    device = self.actor_critic.device

                if self.is_discrete_action_space:
                    return self.choose_action_discrete(actor_value, device)
                else:
                    return self.choose_action_continuous(actor_value, device)

            def choose_action_discrete(self, pi, device):
                probabilities = F.softmax(pi)
                actions_probs = distributions.Categorical(probabilities)
                action_tensor = actions_probs.sample()
                self.a_log_probs = actions_probs.log_prob(action_tensor).to(device)
                a_index = action_tensor.item()
                a = self.action_space[a_index]
                return a

            def choose_action_continuous(self, actor_value, device):
                mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
                sigma = T.exp(sigma_unactivated)
                actions_probs = distributions.Normal(mu, sigma)
                action_tensor = actions_probs.sample(sample_shape=T.Size([self.n_actions]))
                self.a_log_probs = actions_probs.log_prob(action_tensor).to(device)
                action_tensor = T.tanh(action_tensor)
                action_tensor = T.mul(action_tensor, self.action_boundary)
                a = action_tensor.item()
                a = np.array(a).reshape((1,))
                return a

            def learn(self, s, a, r, s_, is_terminal):
                # print('Learning Session')

                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    r = T.tensor(r, dtype=T.float).to(self.critic.device)
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    v = self.critic.forward(s)
                    v_ = self.critic.forward(s_)
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    r = T.tensor(r, dtype=T.float).to(self.actor_critic.device)
                    self.actor_critic.optimizer.zero_grad()
                    _, v = self.actor_critic.forward(s)
                    _, v_ = self.actor_critic.forward(s_)

                td_error = r + self.GAMMA * v_ * (1 - int(is_terminal)) - v

                actor_loss = -self.a_log_probs * td_error
                critic_loss = td_error ** 2
                (actor_loss + critic_loss).backward()

                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    self.actor.optimizer.step()
                    self.critic.optimizer.step()
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    self.actor_critic.optimizer.step()

            def load_model_file(self):
                print("...Loading torch models...")
                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    self.actor.load_model_file()
                    self.critic.load_model_file()
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    self.actor_critic.load_model_file()

            def save_model_file(self):
                print("...Saving torch models...")
                if self.network_type == AC.NETWORK_TYPE_SEPARATE:
                    self.actor.save_model_file()
                    self.critic.save_model_file()
                else:  # self.network_type == AC.NETWORK_TYPE_SHARED
                    self.actor_critic.save_model_file()

    class Agent(object):

        def __init__(self, custom_env, lr_actor=0.0001, lr_critic=None,
                     fc_layers_dims=(400, 300),
                     device_type=None, lib_type=LIBRARY_TF):

            # necessary for filename when saving:
            self.fc_layers_dims = fc_layers_dims
            self.ALPHA = lr_actor
            self.BETA = lr_critic if lr_critic is not None else lr_actor
            self.GAMMA = custom_env.GAMMA

            chkpt_dir = 'tmp/' + custom_env.file_name + '/AC/NNs/'
            network_type = AC.NETWORK_TYPE_SEPARATE if lr_critic is not None else AC.NETWORK_TYPE_SHARED
            if lib_type == LIBRARY_TF:
                self.ac = AC.AC.AC_TF(custom_env, lr_actor, lr_critic, fc_layers_dims, chkpt_dir, network_type,
                                      device_type)
            else:
                self.ac = AC.AC.AC_Torch(custom_env, lr_actor, lr_critic, fc_layers_dims, chkpt_dir, network_type)

        def choose_action(self, s):
            return self.ac.choose_action(s)

        def learn(self, s, a, r, s_, is_terminal):
            self.ac.learn(s, a, r, s_, is_terminal)

        def save_models(self):
            self.ac.save_model_file()

        def load_models(self):
            self.ac.load_model_file()

    @staticmethod
    def train(custom_env, agent, n_episodes, enable_models_saving, save_checkpoint=25):
        env = custom_env.env

        # uncomment the line below to record every episode.
        # env = wrappers.Monitor(env, 'tmp/' + custom_env.file_name + '/AC/recordings',
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
                agent.learn(s, a, r, s_, done)
                observation, s = observation_, s_
            scores_history.append(ep_score)

            Utils.print_training_progress(i, ep_score, scores_history, custom_env.window)

            if enable_models_saving and (i + 1) % save_checkpoint == 0:
                agent.save_models()

        print('\n', 'Game Ended', '\n')

        Utils.plot_running_average(
            custom_env.name, scores_history, window=custom_env.window, show=False,
            file_name=Utils.get_plot_file_name(custom_env, agent.fc_layers_dims, agent.ALPHA, agent.BETA)
        )

    @staticmethod
    def play(env_type, lib_type=LIBRARY_TF, load_checkpoint=False):

        if lib_type == LIBRARY_KERAS or lib_type == LIBRARY_TF:
            print('\n', "Algorithm currently doesn't work with Keras or TensorFlow", '\n')
            return

        enable_models_saving = False

        if env_type == 0:
            custom_env = Envs.ClassicControl.CartPole()
            fc_layers_dims = (32, 32)
            alpha = 0.0001  # 0.00001
            beta = 0.0005
            n_episodes = 2500

        elif env_type == 1:
            # custom_env = Envs.Box2D.LunarLander()
            custom_env = Envs.ClassicControl.Pendulum()
            fc_layers_dims = (2048, 512)
            alpha = 0.00001
            beta = None
            n_episodes = 2000

        else:
            custom_env = Envs.ClassicControl.MountainCarContinuous()
            fc_layers_dims = (256, 256)
            alpha = 0.000005
            beta = 0.00001
            n_episodes = 100  # longer than 100 --> instability (because the value function estimation is unstable)

        if custom_env.input_type != Envs.INPUT_TYPE_OBSERVATION_VECTOR:
            print('\n', 'Algorithm currently works only with INPUT_TYPE_OBSERVATION_VECTOR!', '\n')
            return

        Utils.init_seeds(lib_type, custom_env.env, env_seed=0, np_seed=0, tf_seed=0)

        agent = AC.Agent(custom_env, alpha, beta, fc_layers_dims, lib_type=lib_type)

        if enable_models_saving and load_checkpoint:
            agent.load_models()

        AC.train(custom_env, agent, n_episodes, enable_models_saving)


class DDPG:

    # https://arxiv.org/pdf/1509.02971.pdf

    class AC(object):

        class AC_TF(object):

            class AC_DNN_TF(object):

                def __init__(self, custom_env, fc_layers_dims, sess, lr, name):
                    self.name = name
                    self.lr = lr

                    self.input_dims = custom_env.input_dims
                    self.fc_layers_dims = fc_layers_dims
                    self.n_actions = custom_env.n_actions

                    # relevant for the Actor only:
                    self.memory_batch_size = custom_env.memory_batch_size
                    self.action_boundary = custom_env.action_boundary if not custom_env.is_discrete_action_space \
                        else None

                    self.sess = sess

                def create_actor(self):
                    return DDPG.AC.AC_TF.AC_DNN_TF.Actor(self)

                def create_critic(self):
                    return DDPG.AC.AC_TF.AC_DNN_TF.Critic(self)

                class Actor(object):

                    def __init__(self, ac):
                        self.ac = ac

                        self.build_network()

                        self.params = tf.trainable_variables(scope=self.ac.name)

                        self.mu_gradients = tf.gradients(self.mu, self.params, -self.a_grad)
                        self.normalized_mu_gradients = list(
                            map(lambda x: tf.div(x, self.ac.memory_batch_size), self.mu_gradients))
                        self.optimize = tf.train.AdamOptimizer(self.ac.lr).apply_gradients(
                            zip(self.normalized_mu_gradients, self.params))

                    def build_network(self):
                        with tf.variable_scope(self.ac.name):
                            self.s = tf.placeholder(tf.float32, shape=[None, *self.ac.input_dims], name='s')
                            self.a_grad = tf.placeholder(tf.float32, shape=[None, self.ac.n_actions], name='a_grad')

                            f1 = 1. / np.sqrt(self.ac.fc_layers_dims[0])
                            fc1 = tf.layers.dense(inputs=self.s, units=self.ac.fc_layers_dims[0],
                                                  kernel_initializer=random_uniform(-f1, f1),
                                                  bias_initializer=random_uniform(-f1, f1))
                            fc1_bn = tf.layers.batch_normalization(fc1)
                            fc1_bn_activated = tf.nn.relu(fc1_bn)

                            f2 = 1. / np.sqrt(self.ac.fc_layers_dims[1])
                            fc2 = tf.layers.dense(inputs=fc1_bn_activated, units=self.ac.fc_layers_dims[1],
                                                  kernel_initializer=random_uniform(-f2, f2),
                                                  bias_initializer=random_uniform(-f2, f2))
                            fc2_bn = tf.layers.batch_normalization(fc2)
                            fc2_bn_activated = tf.nn.relu(fc2_bn)

                            f3 = 0.003
                            mu = tf.layers.dense(inputs=fc2_bn_activated, units=self.ac.n_actions,
                                                 activation='tanh',
                                                 kernel_initializer=random_uniform(-f3, f3),
                                                 bias_initializer=random_uniform(-f3, f3))
                            self.mu = tf.multiply(mu, self.ac.action_boundary)  # an ndarray of ndarrays

                    def train(self, s, a_grad):
                        # print('Training Started')
                        self.ac.sess.run(self.optimize,
                                         feed_dict={self.s: s,
                                                    self.a_grad: a_grad})
                        # print('Training Finished')

                    def predict(self, s):
                        return self.ac.sess.run(self.mu,
                                                feed_dict={self.s: s})

                class Critic(object):

                    def __init__(self, ac):
                        self.ac = ac

                        self.build_network()

                        self.params = tf.trainable_variables(scope=self.ac.name)

                        self.optimize = tf.train.AdamOptimizer(self.ac.lr).minimize(self.loss)  # train_op

                        self.action_gradients = tf.gradients(self.q, self.a)  # a list containing an ndarray of ndarrays

                    def build_network(self):
                        with tf.variable_scope(self.ac.name):
                            self.s = tf.placeholder(tf.float32, shape=[None, *self.ac.input_dims], name='s')
                            self.a = tf.placeholder(tf.float32, shape=[None, self.ac.n_actions], name='a')
                            self.q_target = tf.placeholder(tf.float32, shape=[None, 1], name='q_target')

                            f1 = 1. / np.sqrt(self.ac.fc_layers_dims[0])
                            fc1 = tf.layers.dense(inputs=self.s, units=self.ac.fc_layers_dims[0],
                                                  kernel_initializer=random_uniform(-f1, f1),
                                                  bias_initializer=random_uniform(-f1, f1))
                            fc1_bn = tf.layers.batch_normalization(fc1)
                            fc1_bn_activated = tf.nn.relu(fc1_bn)

                            f2 = 1. / np.sqrt(self.ac.fc_layers_dims[1])
                            fc2 = tf.layers.dense(inputs=fc1_bn_activated, units=self.ac.fc_layers_dims[1],
                                                  kernel_initializer=random_uniform(-f2, f2),
                                                  bias_initializer=random_uniform(-f2, f2))
                            fc2_bn = tf.layers.batch_normalization(fc2)

                            action_in_activated = tf.layers.dense(inputs=self.a, units=self.ac.fc_layers_dims[1],
                                                                  activation='relu')

                            state_actions = tf.add(fc2_bn, action_in_activated)
                            state_actions_activated = tf.nn.relu(state_actions)

                            f3 = 0.003
                            self.q = tf.layers.dense(inputs=state_actions_activated, units=1,
                                                     kernel_initializer=random_uniform(-f3, f3),
                                                     bias_initializer=random_uniform(-f3, f3),
                                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))

                            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

                    def train(self, s, a, q_target):
                        # print('Training Started')
                        self.ac.sess.run(self.optimize,
                                         feed_dict={self.s: s,
                                                    self.a: a,
                                                    self.q_target: q_target})
                        # print('Training Finished')

                    def predict(self, s, a):
                        return self.ac.sess.run(self.q,
                                                feed_dict={self.s: s,
                                                           self.a: a})

                    def get_action_gradients(self, inputs, actions):
                        return self.ac.sess.run(self.action_gradients,
                                                feed_dict={self.s: inputs,
                                                           self.a: actions})

            def __init__(self, custom_env, lr_actor, lr_critic, tau, fc_layers_dims, chkpt_dir, device_type):

                self.GAMMA = 0.99
                self.TAU = tau

                self.sess = Utils.get_tf_session_according_to_device_type(device_type)

                #############################

                # Networks:

                self.actor = DDPG.AC.AC_TF.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, lr_actor, 'Actor'
                ).create_actor()
                self.target_actor = DDPG.AC.AC_TF.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, lr_actor, 'ActorTarget'
                ).create_actor()

                self.critic = DDPG.AC.AC_TF.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, lr_critic, 'Critic'
                ).create_critic()
                self.target_critic = DDPG.AC.AC_TF.AC_DNN_TF(
                    custom_env, fc_layers_dims, self.sess, lr_critic, 'CriticTarget'
                ).create_critic()

                #############################

                self.saver = tf.train.Saver()
                self.checkpoint_file = os.path.join(chkpt_dir, 'ddpg_tf.ckpt')

                #############################

                # Soft Update Operations:

                self.update_target_actor = [self.target_actor.params[i].assign(
                    tf.multiply(self.actor.params[i], self.TAU)
                    + tf.multiply(self.target_actor.params[i], 1. - self.TAU)
                ) for i in range(len(self.target_actor.params))]

                self.update_target_critic = [self.target_critic.params[i].assign(
                    tf.multiply(self.critic.params[i], self.TAU)
                    + tf.multiply(self.target_critic.params[i], 1. - self.TAU)
                ) for i in range(len(self.target_critic.params))]

                #############################

                self.sess.run(tf.global_variables_initializer())

                self.update_target_networks_params(first=True)

            def update_target_networks_params(self, first=False):
                if first:
                    original_tau = self.TAU
                    self.TAU = 1.0
                    self.target_actor.ac.sess.run(self.update_target_actor)
                    self.target_critic.ac.sess.run(self.update_target_critic)
                    self.TAU = original_tau
                else:
                    self.target_actor.ac.sess.run(self.update_target_actor)
                    self.target_critic.ac.sess.run(self.update_target_critic)

            def choose_action(self, s, noise):
                s = s[np.newaxis, :]

                mu = self.actor.predict(s)[0]
                a = mu + noise  # mu_prime (mu')
                return a

            def learn(self, memory_batch_size, batch_s, batch_a, batch_r, batch_s_, batch_terminal):
                # print('Learning Session')

                batch_target_mu_ = self.target_actor.predict(batch_s_)
                batch_target_q_ = self.target_critic.predict(batch_s_, batch_target_mu_)

                batch_target_q_ = np.reshape(batch_target_q_, memory_batch_size)
                batch_q_target = batch_r + self.GAMMA * batch_target_q_ * batch_terminal
                batch_q_target = np.reshape(batch_q_target, (memory_batch_size, 1))
                self.critic.train(batch_s, batch_a, batch_q_target)

                batch_mu = self.actor.predict(batch_s)
                batch_a_grads = self.critic.get_action_gradients(batch_s, batch_mu)[0]
                self.actor.train(batch_s, batch_a_grads)

                self.update_target_networks_params()

            def load_model_file(self):
                print("...Loading tf checkpoint...")
                self.saver.restore(self.sess, self.checkpoint_file)

            def save_model_file(self):
                print("...Saving tf checkpoint...")
                self.saver.save(self.sess, self.checkpoint_file)

        class AC_Torch(object):

            class AC_DNN_Torch(nn.Module):

                def __init__(self, custom_env, fc_layers_dims, lr, name, chkpt_dir, is_actor, device_type='cuda'):
                    super(DDPG.AC.AC_Torch.AC_DNN_Torch, self).__init__()

                    self.name = name
                    self.lr = lr
                    self.is_actor = is_actor

                    self.model_file = os.path.join(chkpt_dir, 'ddpg_torch_' + name)

                    self.input_dims = custom_env.input_dims
                    self.fc_layers_dims = fc_layers_dims
                    self.n_actions = custom_env.n_actions

                    # relevant for the Actor only:
                    # self.memory_batch_size = custom_env.memory_batch_size
                    self.action_boundary = custom_env.action_boundary if not custom_env.is_discrete_action_space \
                        else None

                    self.build_network()

                    self.optimizer = optim.Adam(self.parameters(), lr=lr)

                    self.device = Utils.get_torch_device_according_to_device_type(device_type)
                    self.to(self.device)

                def load_model_file(self):
                    self.load_state_dict(T.load(self.model_file))

                def save_model_file(self):
                    T.save(self.state_dict(), self.model_file)

                def build_network(self):
                    if self.is_actor:
                        self.build_network_actor()
                    else:
                        self.build_network_critic()

                def forward(self, s, a=None):
                    if self.is_actor:
                        return self.forward_actor(s)
                    else:
                        return self.forward_critic(s, a)

                def build_network_actor(self):
                    f1 = 1. / np.sqrt(self.fc_layers_dims[0])
                    self.fc1 = nn.Linear(*self.input_dims, self.fc_layers_dims[0])
                    nn.init.uniform_(self.fc1.weight.data, -f1, f1)
                    nn.init.uniform_(self.fc1.bias.data, -f1, f1)
                    self.fc1_bn = nn.LayerNorm(self.fc_layers_dims[0])

                    f2 = 1. / np.sqrt(self.fc_layers_dims[1])
                    self.fc2 = nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])
                    nn.init.uniform_(self.fc2.weight.data, -f2, f2)
                    nn.init.uniform_(self.fc2.bias.data, -f2, f2)
                    self.fc2_bn = nn.LayerNorm(self.fc_layers_dims[1])

                    f3 = 0.003
                    self.mu = nn.Linear(self.fc_layers_dims[1], self.n_actions)
                    nn.init.uniform_(self.mu.weight.data, -f3, f3)
                    nn.init.uniform_(self.mu.bias.data, -f3, f3)

                def forward_actor(self, s):
                    state_value = T.tensor(s, dtype=T.float).to(self.device)

                    state_value = self.fc1(state_value)
                    state_value = self.fc1_bn(state_value)
                    state_value = F.relu(state_value)

                    state_value = self.fc2(state_value)
                    state_value = self.fc2_bn(state_value)
                    state_value = F.relu(state_value)

                    mu_value = self.mu(state_value)
                    mu_value = T.tanh(mu_value)
                    mu_value = T.mul(mu_value, self.action_boundary)
                    return mu_value.to(self.device)

                def build_network_critic(self):
                    f1 = 1. / np.sqrt(self.fc_layers_dims[0])
                    self.fc1 = nn.Linear(*self.input_dims, self.fc_layers_dims[0])
                    nn.init.uniform_(self.fc1.weight.data, -f1, f1)
                    nn.init.uniform_(self.fc1.bias.data, -f1, f1)
                    self.fc1_bn = nn.LayerNorm(self.fc_layers_dims[0])

                    f2 = 1. / np.sqrt(self.fc_layers_dims[1])
                    self.fc2 = nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])
                    nn.init.uniform_(self.fc2.weight.data, -f2, f2)
                    nn.init.uniform_(self.fc2.bias.data, -f2, f2)
                    self.fc2_bn = nn.LayerNorm(self.fc_layers_dims[1])

                    self.action_in = nn.Linear(self.n_actions, self.fc_layers_dims[1])

                    f3 = 0.003
                    self.q = nn.Linear(self.fc_layers_dims[1], 1)
                    nn.init.uniform_(self.q.weight.data, -f3, f3)
                    nn.init.uniform_(self.q.bias.data, -f3, f3)

                    # TODO: add l2 kernel_regularizer of 0.01

                def forward_critic(self, s, a):
                    state_value = T.tensor(s, dtype=T.float).to(self.device)

                    state_value = self.fc1(state_value)
                    state_value = self.fc1_bn(state_value)
                    state_value = F.relu(state_value)

                    state_value = self.fc2(state_value)
                    state_value = self.fc2_bn(state_value)

                    action_value = T.tensor(a, dtype=T.float).to(self.device)

                    action_value = self.action_in(action_value)
                    action_value = F.relu(action_value)

                    state_action_value = T.add(state_value, action_value)
                    state_action_value = F.relu(state_action_value)

                    q_value = self.q(state_action_value)
                    # TODO: apply l2 kernel_regularizer of 0.01
                    return q_value.to(self.device)

            def __init__(self, custom_env, lr_actor, lr_critic, tau, fc_layers_dims, chkpt_dir):

                self.GAMMA = 0.99
                self.TAU = tau

                #############################

                # Networks:

                self.actor = DDPG.AC.AC_Torch.AC_DNN_Torch(
                    custom_env, fc_layers_dims, lr_actor, 'Actor', chkpt_dir, is_actor=True
                )
                self.target_actor = DDPG.AC.AC_Torch.AC_DNN_Torch(
                    custom_env, fc_layers_dims, lr_actor, 'ActorTarget', chkpt_dir, is_actor=True
                )

                self.critic = DDPG.AC.AC_Torch.AC_DNN_Torch(
                    custom_env, fc_layers_dims, lr_critic, 'Critic', chkpt_dir, is_actor=False
                )
                self.target_critic = DDPG.AC.AC_Torch.AC_DNN_Torch(
                    custom_env, fc_layers_dims, lr_critic, 'CriticTarget', chkpt_dir, is_actor=False
                )

                #############################

                self.update_target_networks_params(first=True)

            def update_target_networks_params(self, first=False):
                tau = 1.0 if first else self.TAU

                # update_target_actor:
                actor_state_dict = dict(self.actor.named_parameters())
                target_actor_dict = dict(self.target_actor.named_parameters())
                for name in actor_state_dict:
                    actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                             (1 - tau) * target_actor_dict[name].clone()
                self.target_actor.load_state_dict(actor_state_dict)

                # update_target_critic:
                critic_state_dict = dict(self.critic.named_parameters())
                target_critic_dict = dict(self.target_critic.named_parameters())
                for name in critic_state_dict:
                    critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                              (1 - tau) * target_critic_dict[name].clone()
                self.target_critic.load_state_dict(critic_state_dict)

                # if tau == 1.0
                #     self.verify_copy(tau)

            def verify_copy(self):
                actor_state_dict = dict(self.target_actor.named_parameters())
                print('Verifying Target Actor params have been copied')
                for name, param in self.actor.named_parameters():
                    print(name, T.equal(param, actor_state_dict[name]))

                critic_state_dict = dict(self.target_critic.named_parameters())
                print('Verifying Target Critic params have been copied')
                for name, param in self.critic.named_parameters():
                    print(name, T.equal(param, critic_state_dict[name]))

                input()

            def choose_action(self, s, noise):
                noise = T.tensor(noise, dtype=T.float).to(self.actor.device)

                self.actor.eval()
                mu = self.actor.forward(s)
                mu_prime = mu + noise

                return mu_prime.cpu().detach().numpy()

            def learn(self, memory_batch_size, batch_s, batch_a, batch_r, batch_s_, batch_terminal):
                # print('Learning Session')

                batch_r = T.tensor(batch_r, dtype=T.float).to(self.critic.device)
                batch_terminal = T.tensor(batch_terminal).to(self.critic.device)

                self.target_actor.eval()
                batch_target_mu_ = self.target_actor.forward(batch_s_)
                self.target_critic.eval()
                batch_target_q_ = self.target_critic.forward(batch_s_, batch_target_mu_)

                batch_target_q_ = batch_target_q_.view(memory_batch_size)
                batch_q_target = batch_r + self.GAMMA * batch_target_q_ * batch_terminal
                batch_q_target = batch_q_target.view(memory_batch_size, 1).to(self.critic.device)

                self.critic.eval()
                batch_q = self.critic.forward(batch_s, batch_a)

                self.critic.train()
                self.critic.optimizer.zero_grad()
                critic_loss = F.mse_loss(batch_q_target, batch_q)
                critic_loss.backward()
                self.critic.optimizer.step()

                self.actor.eval()
                batch_mu = self.actor.forward(batch_s)
                self.critic.eval()
                batch_q_to_mu = self.critic.forward(batch_s, batch_mu)

                self.actor.train()
                self.actor.optimizer.zero_grad()
                actor_loss = T.mean(-batch_q_to_mu)
                actor_loss.backward()
                self.actor.optimizer.step()

                self.update_target_networks_params()

            def load_model_file(self):
                print("...Loading models...")
                self.actor.load_model_file()
                self.target_actor.load_model_file()
                self.critic.load_model_file()
                self.target_critic.load_model_file()

            def save_model_file(self):
                print("...Saving models...")
                self.actor.save_model_file()
                self.target_actor.save_model_file()
                self.critic.save_model_file()
                self.target_critic.save_model_file()

    class OUActionNoise(object):

        # https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

        def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
            self.mu = mu
            self.sigma = sigma
            self.theta = theta
            self.dt = dt  # differential with respect to time - this is a temporally correlated noise
            self.x0 = x0  # initial value
            self.reset()  # resets the temporal correlation

        def reset(self):
            # sets the previous value for the noise
            self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        def __call__(self):
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
            self.x_prev = x
            return x

        def __repr__(self):
            return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    class Agent(object):

        def __init__(self, custom_env, lr_actor=0.0001, lr_critic=0.001, tau=0.001,
                     fc_layers_dims=(400, 300), memory_batch_size=None, device_type=None, lib_type=LIBRARY_TF):

            # necessary for filename when saving:
            self.fc_layers_dims = fc_layers_dims
            self.ALPHA = lr_actor
            self.BETA = lr_critic

            self.GAMMA = 0.99
            self.TAU = tau

            self.noise = DDPG.OUActionNoise(mu=np.zeros(custom_env.n_actions), sigma=0.2)

            if memory_batch_size is not None:
                self.memory_batch_size = memory_batch_size
            else:
                self.memory_batch_size = 64 if custom_env.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR else 16
            self.memory = ReplayBuffer(custom_env, lib_type, is_discrete_action_space=False)

            chkpt_dir = 'tmp/' + custom_env.file_name + '/DDPG/NNs/'
            if lib_type == LIBRARY_TF:
                self.ac = DDPG.AC.AC_TF(custom_env, lr_actor, lr_critic, tau, fc_layers_dims, chkpt_dir,
                                        device_type)
            else:
                self.ac = DDPG.AC.AC_Torch(custom_env, lr_actor, lr_critic, tau, fc_layers_dims, chkpt_dir)

        def store_transition(self, s, a, r, s_, is_terminal):
            self.memory.store_transition(s, a, r, s_, is_terminal)

        def choose_action(self, s, training_mode=False):
            noise = self.noise() if training_mode else 0
            a = self.ac.choose_action(s, noise)
            return a

        def learn_wrapper(self):
            if self.memory.memory_counter >= self.memory_batch_size:
                self.learn()

        def learn(self):
            # print('Learning Session')

            batch_s, batch_s_, batch_r, batch_terminal, batch_a = \
                self.memory.sample_batch(self.memory_batch_size)

            self.ac.learn(self.memory_batch_size,
                          batch_s, batch_a, batch_r, batch_s_,batch_terminal)

        def save_models(self):
            self.ac.save_model_file()

        def load_models(self):
            self.ac.load_model_file()

    @staticmethod
    def train(custom_env, agent, n_episodes, enable_models_saving, save_checkpoint=25):
        env = custom_env.env

        # uncomment the line below to record every episode.
        # env = wrappers.Monitor(env, 'tmp/' + custom_env.file_name + '/DDPG/recordings',
        #                        video_callable=lambda episode_id: True, force=True)

        print('\n', 'Game Started', '\n')

        scores_history = []

        for i in range(n_episodes):
            done = False
            ep_score = 0

            agent.noise.reset()

            observation = env.reset()
            s = custom_env.get_state(observation, None)
            while not done:
                a = agent.choose_action(s, training_mode=True)
                observation_, r, done, info = env.step(a)
                r = custom_env.update_reward(r, done, info)
                s_ = custom_env.get_state(observation_, s.copy())
                ep_score += r
                agent.store_transition(s, a, r, s_, done)
                agent.learn()
                observation, s = observation_, s_
            scores_history.append(ep_score)

            Utils.print_training_progress(i, ep_score, scores_history, custom_env.window)

            if enable_models_saving and (i + 1) % save_checkpoint == 0:
                agent.save_models()

        print('\n', 'Game Ended', '\n')

        Utils.plot_running_average(
            custom_env.name, scores_history, window=custom_env.window, show=False,
            file_name=Utils.get_plot_file_name(custom_env, agent.fc_layers_dims, agent.ALPHA, agent.BETA)
        )

    @staticmethod
    def play(env_type, lib_type=LIBRARY_TF, load_checkpoint=False):

        if lib_type == LIBRARY_KERAS:
            print('\n', "Algorithm currently doesn't work with Keras", '\n')
            return

        enable_models_saving = False

        if env_type == 0:
            custom_env = Envs.ClassicControl.Pendulum()
            fc_layers_dims = (800, 600)
            alpha = 0.00005
            beta = 0.0005
            n_episodes = 1000

        # elif env_type == 1:
            # custom_env = Envs.Box2D.BipedalWalker()
            # fc_layers_dims = (400, 300)
            # alpha = 0.00005
            # beta = 0.0005
            # n_episodes = 5000

        else:
            # custom_env = Envs.Box2D.LunarLanderContinuous()
            custom_env = Envs.ClassicControl.MountainCarContinuous()
            fc_layers_dims = (400, 300)
            alpha = 0.000025
            beta = 0.00025
            n_episodes = 1000

        tau = 0.001

        if custom_env.is_discrete_action_space:
            print('\n', "Environment's Action Space should be continuous!", '\n')
            return

        if custom_env.input_type != Envs.INPUT_TYPE_OBSERVATION_VECTOR:
            print('\n', 'Algorithm currently works only with INPUT_TYPE_OBSERVATION_VECTOR!', '\n')
            return

        Utils.init_seeds(lib_type, custom_env.env, env_seed=0, np_seed=0, tf_seed=0)

        agent = DDPG.Agent(
            custom_env, alpha, beta, tau, fc_layers_dims,
            memory_batch_size=custom_env.memory_batch_size, lib_type=lib_type
        )

        if enable_models_saving and load_checkpoint:
            agent.load_models()

        DDPG.train(custom_env, agent, n_episodes, enable_models_saving)

