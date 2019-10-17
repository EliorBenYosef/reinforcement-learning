from numpy.random import seed
seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import os
from gym import wrappers
import numpy as np
import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import random_uniform_initializer as random_uniform

import torch as T
import torch.distributions as distributions
import torch.nn.functional as F

import keras.models as models
import keras.layers as layers
import keras.initializers as initializers
import keras.backend as K

import utils
from deep_reinforcement_learning.envs import Envs


class DNN(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, alpha, chkpt_dir):
        self.input_type = custom_env.input_type

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.n_actions = custom_env.n_actions

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.chkpt_dir = chkpt_dir

    def create_dnn_tensorflow(self, name):
        return DNN.DNN_TensorFlow(self, name)

    def create_dnn_keras(self):
        return DNN.DNN_Keras(self)

    def create_dnn_torch(self, relevant_screen_size, image_channels):
        return DNN.DNN_Torch(self, relevant_screen_size, image_channels)

    class DNN_TensorFlow(object):

        def __init__(self, dnn, name, device_map=None):
            self.dnn = dnn

            self.name = name

            self.sess = utils.DeviceSetUtils.tf_get_session_according_to_device(device_map)
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
                    fc1_ac = tf.layers.dense(inputs=self.s, units=self.dnn.fc_layers_dims[0],
                                             activation='relu',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    fc2_ac = tf.layers.dense(inputs=fc1_ac, units=self.dnn.fc_layers_dims[1],
                                             activation='relu',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    fc_last = tf.layers.dense(inputs=fc2_ac, units=self.dnn.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

                else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                    conv1 = tf.layers.conv2d(inputs=self.s, filters=32,
                                             kernel_size=(8, 8), strides=4, name='conv1',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    conv1_bn = tf.layers.batch_normalization(inputs=conv1, epsilon=1e-5, name='conv1_bn')
                    conv1_bn_ac = tf.nn.relu(conv1_bn)
                    conv2 = tf.layers.conv2d(inputs=conv1_bn_ac, filters=64,
                                             kernel_size=(4, 4), strides=2, name='conv2',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    conv2_bn = tf.layers.batch_normalization(inputs=conv2, epsilon=1e-5, name='conv2_bn')
                    conv2_bn_ac = tf.nn.relu(conv2_bn)
                    conv3 = tf.layers.conv2d(inputs=conv2_bn_ac, filters=128,
                                             kernel_size=(3, 3), strides=1, name='conv3',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    conv3_bn = tf.layers.batch_normalization(inputs=conv3, epsilon=1e-5, name='conv3_bn')
                    conv3_bn_ac = tf.nn.relu(conv3_bn)

                    flat = tf.layers.flatten(conv3_bn_ac)
                    fc1_ac = tf.layers.dense(inputs=flat, units=self.dnn.fc_layers_dims[0],
                                             activation='relu',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                    fc_last = tf.layers.dense(inputs=fc1_ac, units=self.dnn.n_actions,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

                self.actions_probabilities = tf.nn.softmax(fc_last, name='actions_probabilities')

                negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.a_index, logits=fc_last)
                loss = negative_log_probability * self.G
                if self.dnn.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                    loss = tf.reduce_mean(loss)

                optimizer = utils.Optimizers.tf_get_optimizer(self.dnn.optimizer_type, self.dnn.ALPHA)
                self.optimize = optimizer.minimize(loss)  # train_op

        def get_actions_probabilities(self, batch_s):
            return self.sess.run(self.actions_probabilities, feed_dict={self.s: batch_s})[0]

        def learn_entire_batch(self, memory, GAMMA):
            memory_s = np.array(memory.memory_s)
            memory_a_indices = np.array(memory.memory_a_indices)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.int8)

            memory_G = utils.Calculator.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)

            print('Training Started')
            _ = self.sess.run(self.optimize,
                              feed_dict={self.s: memory_s,
                                         self.a_index: memory_a_indices,
                                         self.G: memory_G})
            print('Training Finished')

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class DNN_Keras(object):

        def __init__(self, dnn):
            self.dnn = dnn

            self.h5_file = os.path.join(dnn.chkpt_dir, 'dnn_keras.h5')

            self.optimizer = utils.Optimizers.keras_get_optimizer(self.dnn.optimizer_type, self.dnn.ALPHA)

            self.model, self.policy = self.build_networks()

        def build_networks(self):

            s = layers.Input(shape=self.dnn.input_dims, dtype='float32', name='s')

            if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                x = layers.Dense(self.dnn.fc_layers_dims[0], activation='relu',
                                 kernel_initializer=initializers.glorot_uniform(seed=None))(s)
                x = layers.Dense(self.dnn.fc_layers_dims[1], activation='relu',
                                 kernel_initializer=initializers.glorot_uniform(seed=None))(x)

            else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES

                x = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, name='conv1',
                                  kernel_initializer=initializers.glorot_uniform(seed=None))(s)
                x = layers.BatchNormalization(epsilon=1e-5, name='conv1_bn')(x)
                x = layers.Activation(activation='relu', name='conv1_bn_ac')(x)
                x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, name='conv2',
                                  kernel_initializer=initializers.glorot_uniform(seed=None))(x)
                x = layers.BatchNormalization(epsilon=1e-5, name='conv2_bn')(x)
                x = layers.Activation(activation='relu', name='conv2_bn_ac')(x)
                x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, name='conv3',
                                  kernel_initializer=initializers.glorot_uniform(seed=None))(x)
                x = layers.BatchNormalization(epsilon=1e-5, name='conv3_bn')(x)
                x = layers.Activation(activation='relu', name='conv3_bn_ac')(x)
                x = layers.Flatten()(x)
                x = layers.Dense(self.dnn.fc_layers_dims[0], activation='relu',
                                 kernel_initializer=initializers.glorot_uniform(seed=None))(x)

            actions_probabilities = layers.Dense(self.dnn.n_actions, activation='softmax', name='actions_probabilities',
                                                 kernel_initializer=initializers.glorot_uniform(seed=None))(x)

            policy = models.Model(inputs=s, outputs=actions_probabilities)

            #############################

            G = layers.Input(shape=(1,), dtype='float32', name='G')  # advantages. batch_shape=[None]

            def custom_loss(y_true, y_pred):  # (a_indices_one_hot, intermediate_model.output)
                y_pred_clipped = K.clip(y_pred, 1e-8, 1 - 1e-8)  # we set boundaries so we won't take log of 0\1
                log_lik = y_true * K.log(y_pred_clipped)  # log_probability
                loss = K.sum(-log_lik * G)  # K.mean ?
                return loss

            model = models.Model(inputs=[s, G], outputs=actions_probabilities)  # policy_model
            model.compile(optimizer=self.optimizer, loss=custom_loss)

            return model, policy

        def get_actions_probabilities(self, batch_s):
            actions_probabilities = self.policy.predict(batch_s)[0]
            return actions_probabilities

        def learn_entire_batch(self, memory, GAMMA):
            memory_s = np.array(memory.memory_s)
            memory_a_indices = np.array(memory.memory_a_indices)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.int8)

            memory_G = utils.Calculator.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)

            memory_size = len(memory_a_indices)
            memory_a_indices_one_hot = np.zeros((memory_size, self.dnn.n_actions), dtype=np.int8)
            memory_a_indices_one_hot[np.arange(memory_size), memory_a_indices] = 1

            print('Training Started')
            _ = self.model.fit([memory_s, memory_G], memory_a_indices_one_hot, verbose=0)
            print('Training Finished')

        def load_model_file(self):
            print("...Loading Keras h5...")
            self.model = models.load_model(self.h5_file)

        def save_model_file(self):
            print("...Saving Keras h5...")
            self.model.save(self.h5_file)

    class DNN_Torch(T.nn.Module):

        def __init__(self, dnn, relevant_screen_size, image_channels, device_str='cuda'):

            super(DNN.DNN_Torch, self).__init__()

            self.dnn = dnn
            self.relevant_screen_size = relevant_screen_size
            self.image_channels = image_channels

            self.model_file = os.path.join(dnn.chkpt_dir, 'dnn_torch')

            self.build_network()

            self.optimizer = utils.Optimizers.torch_get_optimizer(
                self.dnn.optimizer_type, self.parameters(), self.dnn.ALPHA)

            self.device = utils.DeviceSetUtils.torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def build_network(self):
            if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                self.fc1 = T.nn.Linear(*self.dnn.input_dims, self.dnn.fc_layers_dims[0])
                self.fc2 = T.nn.Linear(self.dnn.fc_layers_dims[0], self.dnn.fc_layers_dims[1])
                self.fc3 = T.nn.Linear(self.dnn.fc_layers_dims[1], self.dnn.n_actions)

            else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES
                frames_stack_size = Envs.Atari.frames_stack_size
                self.in_channels = frames_stack_size * self.image_channels

                conv1_filters, conv2_filters, conv3_filters = 32, 64, 128
                conv1_fps = 8, 1, 4
                conv2_fps = 4, 0, 2
                conv3_fps = 3, 0, 1

                self.conv1 = T.nn.Conv2d(self.in_channels, conv1_filters, conv1_fps[0],
                                       padding=conv1_fps[1], stride=conv1_fps[2])
                self.conv2 = T.nn.Conv2d(conv1_filters, conv2_filters, conv2_fps[0],
                                       padding=conv2_fps[1], stride=conv2_fps[2])
                self.conv3 = T.nn.Conv2d(conv2_filters, conv3_filters, conv3_fps[0],
                                       padding=conv3_fps[1], stride=conv3_fps[2])

                i_H, i_W = self.dnn.input_dims[0], self.dnn.input_dims[1]
                conv1_o_H, conv1_o_W = utils.Calculator.calc_conv_layer_output_dims(i_H, i_W, *conv1_fps)
                conv2_o_H, conv2_o_W = utils.Calculator.calc_conv_layer_output_dims(conv1_o_H, conv1_o_W, *conv2_fps)
                conv3_o_H, conv3_o_W = utils.Calculator.calc_conv_layer_output_dims(conv2_o_H, conv2_o_W, *conv3_fps)
                self.flat_dims = conv3_filters * conv3_o_H * conv3_o_W

                self.fc1 = T.nn.Linear(self.flat_dims, self.dnn.fc_layers_dims[0])
                self.fc2 = T.nn.Linear(self.dnn.fc_layers_dims[0], self.dnn.n_actions)

        def forward(self, s):
            input = T.tensor(s, dtype=T.float).to(self.device)

            if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:

                fc1_ac = F.relu(self.fc1(input))
                fc2_ac = F.relu(self.fc2(fc1_ac))
                fc_last = self.fc3(fc2_ac)

            else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES

                input = input.view(-1, self.in_channels, *self.relevant_screen_size)
                conv1_ac = F.relu(self.conv1(input))
                conv2_ac = F.relu(self.conv2(conv1_ac))
                conv3_ac = F.relu(self.conv3(conv2_ac))
                flat = conv3_ac.view(-1, self.flat_dims).to(self.device)
                fc1_ac = F.relu(self.fc1(flat))
                fc_last = self.fc2(fc1_ac)

            actions_probabilities = F.softmax(fc_last).to(self.device)

            return actions_probabilities

        def learn_entire_batch(self, memory, GAMMA):
            memory_a_log_probs = np.array(memory.memory_a_log_probs)
            memory_r = np.array(memory.memory_r)
            memory_terminal = np.array(memory.memory_terminal, dtype=np.uint8)

            memory_G = utils.Calculator.calculate_returns_of_consecutive_episodes(memory_r, memory_terminal, GAMMA)
            memory_G = T.tensor(memory_G, dtype=T.float).to(self.device)

            self.optimizer.zero_grad()
            loss = 0
            for G, a_log_prob in zip(memory_G, memory_a_log_probs):
                loss += -a_log_prob * G
            loss.backward()
            print('Training Started')
            self.optimizer.step()
            print('Training Finished')

        def load_model_file(self):
            print("...Loading Torch file...")
            self.load_state_dict(T.load(self.model_file))

        def save_model_file(self):
            print("...Saving Torch file...")
            T.save(self.state_dict(), self.model_file)


class Memory(object):

    def __init__(self, custom_env, lib_type):
        self.n_actions = custom_env.n_actions
        self.action_space = custom_env.action_space

        self.lib_type = lib_type

        if self.lib_type == utils.LIBRARY_TORCH:
            self.memory_a_log_probs = []

        else:  # utils.LIBRARY_TF \ utils.LIBRARY_KERAS
            self.memory_s = []
            self.memory_a_indices = []

        self.memory_r = []
        self.memory_terminal = []

    def store_transition(self, s, a, r, is_terminal):
        if self.lib_type != utils.LIBRARY_TORCH:  # utils.LIBRARY_TF \ utils.LIBRARY_KERAS
            self.memory_s.append(s)
            self.memory_a_indices.append(self.action_space.index(a))

        self.memory_r.append(r)
        self.memory_terminal.append(int(is_terminal))

    def store_a_log_probs(self, a_log_probs):
        if self.lib_type == utils.LIBRARY_TORCH:
            self.memory_a_log_probs.append(a_log_probs)

    def reset_memory(self):
        if self.lib_type == utils.LIBRARY_TORCH:
            self.memory_a_log_probs = []

        else:  # utils.LIBRARY_TF \ utils.LIBRARY_KERAS
            self.memory_s = []
            self.memory_a_indices = []

        self.memory_r = []
        self.memory_terminal = []


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims, ep_batch_num,
                 alpha, optimizer_type=utils.Optimizers.OPTIMIZER_Adam,
                 lib_type=utils.LIBRARY_TF,
                 base_dir=''):

        self.GAMMA = custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.ep_batch_num = ep_batch_num

        self.optimizer_type = optimizer_type
        self.ALPHA = alpha

        self.action_space = custom_env.action_space

        self.lib_type = lib_type

        self.memory = Memory(custom_env, lib_type)

        # sub_dir = utils.General.get_file_name(None, self) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        utils.General.make_sure_dir_exists(self.chkpt_dir)

        self.policy_dnn = self.init_network(custom_env)

    def init_network(self, custom_env):
        dnn_base = DNN(custom_env, self.fc_layers_dims, self.optimizer_type, self.ALPHA, self.chkpt_dir)

        if self.lib_type == utils.LIBRARY_TF:
            dnn = dnn_base.create_dnn_tensorflow(name='q_policy')

        elif self.lib_type == utils.LIBRARY_KERAS:
            dnn = dnn_base.create_dnn_keras()

        else:  # self.lib_type == utils.LIBRARY_TORCH
            if custom_env.input_type == Envs.INPUT_TYPE_STACKED_FRAMES:
                relevant_screen_size = custom_env.relevant_screen_size
                image_channels = custom_env.image_channels
            else:
                relevant_screen_size = None
                image_channels = None

            dnn = dnn_base.create_dnn_torch(relevant_screen_size, image_channels)

        return dnn

    def choose_action(self, s):
        s = s[np.newaxis, :]

        if self.lib_type == utils.LIBRARY_TORCH:
            probabilities = self.policy_dnn.forward(s)
            actions_probs = distributions.Categorical(probabilities)
            action_tensor = actions_probs.sample()
            a_log_probs = actions_probs.log_prob(action_tensor)
            self.memory.store_a_log_probs(a_log_probs)
            a_index = action_tensor.item()
            a = self.action_space[a_index]

        else:  # utils.LIBRARY_TF \ utils.LIBRARY_KERAS
            probabilities = self.policy_dnn.get_actions_probabilities(s)
            a = np.random.choice(self.action_space, p=probabilities)

        return a

    def store_transition(self, s, a, r, is_terminal):
        self.memory.store_transition(s, a, r, is_terminal)

    def learn(self):
        print('Learning Session')

        self.policy_dnn.learn_entire_batch(self.memory, self.GAMMA)
        self.memory.reset_memory()

    def save_models(self):
        self.policy_dnn.save_model_file()

    def load_models(self):
        self.policy_dnn.load_model_file()


def train_agent(custom_env, agent, n_episodes,
                ep_batch_num,
                enable_models_saving, load_checkpoint,
                visualize=False, record=False):

    scores_history, learn_episode_index, max_avg = utils.SaverLoader.load_training_data(agent, load_checkpoint)
    save_episode_index = learn_episode_index

    env = custom_env.env

    if record:
        env = wrappers.Monitor(
            env, 'recordings/PG/', force=True,
            video_callable=lambda episode_id: episode_id == 0 or episode_id == (n_episodes - 1)
        )

    print('\n', 'Training Started', '\n')
    train_start_time = datetime.datetime.now()

    starting_ep = learn_episode_index + 1
    for i in range(starting_ep, n_episodes):
        ep_start_time = datetime.datetime.now()

        done = False
        ep_score = 0

        observation = env.reset()
        s = custom_env.get_state(observation, None)

        if visualize and i == n_episodes - 1:
            env.render()

        while not done:
            a = agent.choose_action(s)
            observation_, r, done, info = env.step(a)
            r = custom_env.update_reward(r, done, info)
            s_ = custom_env.get_state(observation_, s)
            ep_score += r
            agent.store_transition(s, a, r, done)
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        scores_history.append(ep_score)
        utils.SaverLoader.pickle_save(scores_history, 'scores_history_train_total', agent.chkpt_dir)

        avg_num = custom_env.window
        if ep_batch_num > avg_num:
            avg_num = ep_batch_num
        current_avg = utils.Printer.print_training_progress(i, ep_score, scores_history, avg_num, ep_start_time=ep_start_time)

        if enable_models_saving and current_avg is not None and learn_episode_index != -1 and \
                (max_avg is None or current_avg >= max_avg):
            max_avg = current_avg
            utils.SaverLoader.pickle_save(max_avg, 'max_avg', agent.chkpt_dir)
            if i - save_episode_index - 1 >= ep_batch_num:
                save_episode_index = learn_episode_index
                utils.SaverLoader.save_training_data(agent, learn_episode_index, scores_history)

        if (i + 1) % ep_batch_num == 0:
            learn_episode_index = i
            learn_start_time = datetime.datetime.now()
            agent.learn()
            print('Learn time: %s' % str(datetime.datetime.now() - learn_start_time).split('.')[0])

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Training Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
          (n_episodes - starting_ep, str(datetime.datetime.now() - train_start_time).split('.')[0]), '\n')

    return scores_history


def play(env_type, lib_type=utils.LIBRARY_TF, enable_models_saving=False, load_checkpoint=False):

    if env_type == 0:
        # custom_env = Envs.Box2D.LunarLander()
        custom_env = Envs.ClassicControl.CartPole()
        fc_layers_dims = [128, 128] if lib_type == utils.LIBRARY_TORCH else [64, 64]
        optimizer_type = utils.Optimizers.OPTIMIZER_Adam
        alpha = 0.001 if lib_type == utils.LIBRARY_TORCH else 0.0005
        ep_batch_num = 1  # REINFORCE algorithm (MC PG)
        n_episodes = 2000 if lib_type == utils.LIBRARY_KERAS else 2500  # supposed to be enough for good results in PG

    elif env_type == 1:
        custom_env = Envs.Atari.Breakout()
        fc_layers_dims = [256, 0]
        optimizer_type = utils.Optimizers.OPTIMIZER_RMSprop  # utils.Optimizers.OPTIMIZER_SGD
        alpha = 0.00025
        ep_batch_num = 1  # REINFORCE algorithm (MC PG)
        n_episodes = 200  # start with 200, then 5000 ?

    else:
        custom_env = Envs.Atari.SpaceInvaders()
        fc_layers_dims = [256, 0]
        optimizer_type = utils.Optimizers.OPTIMIZER_RMSprop  # utils.Optimizers.OPTIMIZER_SGD
        alpha = 0.001  # 0.003
        ep_batch_num = 10
        n_episodes = 1000

    if not custom_env.is_discrete_action_space:
        print('\n', "Environment's Action Space should be discrete!", '\n')
        return

    custom_env.env.seed(28)

    utils.DeviceSetUtils.set_device(lib_type, devices_dict=None)

    method_name = 'PG'
    base_dir = 'tmp/' + custom_env.file_name + '/' + method_name + '/'

    agent = Agent(custom_env, fc_layers_dims,
                  ep_batch_num,
                  alpha, optimizer_type=optimizer_type,
                  lib_type=lib_type, base_dir=base_dir)

    scores_history = train_agent(custom_env, agent, n_episodes,
                                 ep_batch_num,
                                 enable_models_saving, load_checkpoint)
    utils.Plotter.plot_running_average(
        custom_env.name, method_name, scores_history, window=custom_env.window, show=False,
        file_name=utils.General.get_file_name(custom_env.file_name, agent, n_episodes, method_name) + '_train',
        directory=agent.chkpt_dir if enable_models_saving else None
    )

    # scores_history_test = utils.Tester.test_trained_agent(custom_env, agent, enable_models_saving)
    # utils.Plotter.plot_running_average(
    #     custom_env.name, method_name, scores_history_test, window=custom_env.window, show=False,
    #     file_name=utils.General.get_file_name(custom_env.file_name, agent, n_episodes, method_name) + '_test',
    #     directory=agent.chkpt_dir if enable_models_saving else None
    # )


if __name__ == '__main__':
    play(0, lib_type=utils.LIBRARY_KERAS)          # CartPole (0), Breakout (1), SpaceInvaders (2)
