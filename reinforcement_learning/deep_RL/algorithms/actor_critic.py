from numpy.random import seed

import reinforcement_learning.utils.plotter

seed(28)
from tensorflow import set_random_seed
set_random_seed(28)

import os
import sys
from gym import wrappers
import numpy as np
import datetime

import tensorflow as tf
import tensorflow_probability as tfp

import torch as T
import torch.distributions as distributions
import torch.nn.functional as F

import keras.models as models
import keras.layers as layers
import keras.backend as K

from reinforcement_learning.utils import utils
import reinforcement_learning.deep_RL.envs as Envs


NETWORK_TYPE_SEPARATE = 0
NETWORK_TYPE_SHARED = 1


class DNN(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr, name, network_type, is_actor=False):

        self.name = name
        self.network_type = network_type
        self.is_actor = is_actor

        self.input_type = custom_env.input_type

        self.is_discrete_action_space = custom_env.is_discrete_action_space

        self.input_dims = custom_env.input_dims
        self.fc_layers_dims = fc_layers_dims
        self.n_outputs = custom_env.n_actions if custom_env.is_discrete_action_space else 2

        self.optimizer_type = optimizer_type
        self.lr = lr

    def create_dnn_tensorflow(self, sess):
        return DNN.AC_DNN_TF(self, sess)

    def create_dnn_keras(self, lr_actor, lr_critic, chkpt_dir):
        return DNN.AC_DNN_Keras(self, lr_actor, lr_critic, chkpt_dir)

    def create_dnn_torch(self, chkpt_dir, device_str='cuda'):
        return DNN.AC_DNN_Torch(self, chkpt_dir, device_str)

    class AC_DNN_TF(object):

        def __init__(self, dnn, sess):

            self.dnn = dnn

            self.sess = sess

            self.build_network()

            self.params = tf.trainable_variables(scope=self.dnn.name)

        def build_network(self):
            with tf.variable_scope(self.dnn.name):
                self.s = tf.placeholder(tf.float32, shape=[None, *self.dnn.input_dims], name='s')
                self.td_error = tf.placeholder(tf.float32, shape=[None, 1], name='td_error')
                self.a_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='a_log_probs')
                self.a = tf.placeholder(tf.int32, shape=[None, 1], name='a')

                fc1_ac = tf.layers.dense(inputs=self.s, units=self.dnn.fc_layers_dims[0],
                                         activation='relu')
                fc2_ac = tf.layers.dense(inputs=fc1_ac, units=self.dnn.fc_layers_dims[1],
                                         activation='relu')

                if self.dnn.network_type == NETWORK_TYPE_SEPARATE:  # build_A_or_C_network
                    self.fc3 = tf.layers.dense(inputs=fc2_ac, units=self.dnn.n_outputs if self.dnn.is_actor else 1)
                    loss = self.get_actor_loss() if self.dnn.is_actor else self.get_critic_loss()

                else:  # self.network_type == NETWORK_TYPE_SHARED  # build_A_and_C_network
                    self.fc3 = tf.layers.dense(inputs=fc2_ac, units=self.dnn.n_outputs)  # Actor layer
                    self.v = tf.layers.dense(inputs=fc2_ac, units=1)  # Critic layer
                    loss = self.get_actor_loss() + self.get_critic_loss()

                optimizer = utils.Optimizers.tf_get_optimizer(self.dnn.optimizer_type, self.dnn.lr)
                self.optimize = optimizer.minimize(loss)  # train_op

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
            if self.dnn.network_type == NETWORK_TYPE_SEPARATE:
                state_value = self.sess.run(self.fc3, feed_dict={self.s: s})
                return state_value

            else:  # self.network_type == NETWORK_TYPE_SHARED
                state_value = self.sess.run(self.fc3, feed_dict={self.s: s})  # Actor value
                v = self.sess.run(self.v, feed_dict={self.s: s})  # Critic value
                return state_value, v

        def train(self, s, td_error, a=None):
            print('Training Started')
            self.sess.run(self.optimize,
                          feed_dict={self.s: s,
                                     self.td_error: td_error,
                                     self.a: a})
            print('Training Finished')

    class AC_DNN_Keras(object):

        def __init__(self, dnn, lr_actor, lr_critic, chkpt_dir):

            self.dnn = dnn

            self.lr_actor = lr_actor
            self.lr_critic = lr_critic

            self.h5_file_actor = os.path.join(chkpt_dir, 'ac_keras_actor.h5')
            self.h5_file_critic = os.path.join(chkpt_dir, 'ac_keras_critic.h5')

            self.actor, self.critic, self.policy = self.build_networks()

        def build_networks(self):

            # s = layers.Input(shape=self.dnn.input_dims, dtype='float32', name='s')
            s = layers.Input(shape=self.dnn.input_dims)

            if self.dnn.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR:
                x = layers.Dense(self.dnn.fc_layers_dims[0], activation='relu')(s)
                x = layers.Dense(self.dnn.fc_layers_dims[1], activation='relu')(x)

            else:  # self.input_type == Envs.INPUT_TYPE_STACKED_FRAMES

                x = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, name='conv1')(s)
                x = layers.BatchNormalization(epsilon=1e-5, name='conv1_bn')(x)
                x = layers.Activation(activation='relu', name='conv1_bn_ac')(x)
                x = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, name='conv2')(x)
                x = layers.BatchNormalization(epsilon=1e-5, name='conv2_bn')(x)
                x = layers.Activation(activation='relu', name='conv2_bn_ac')(x)
                x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, name='conv3')(x)
                x = layers.BatchNormalization(epsilon=1e-5, name='conv3_bn')(x)
                x = layers.Activation(activation='relu', name='conv3_bn_ac')(x)
                x = layers.Flatten()(x)
                x = layers.Dense(self.dnn.fc_layers_dims[0], activation='relu')(x)

            #############################

            critic_value = layers.Dense(1, activation='linear', name='critic_value')(x)
            critic = models.Model(inputs=s, outputs=critic_value)
            optimizer_critic = utils.Optimizers.keras_get_optimizer(self.dnn.optimizer_type, self.lr_critic)
            critic.compile(optimizer_critic, loss='mean_squared_error')

            #############################

            activation = 'softmax' if self.dnn.is_discrete_action_space else None  # discrete - actions_probabilities
            actor_value = layers.Dense(self.dnn.n_outputs, activation=activation, name='actor_value')(x)

            #############

            policy = models.Model(inputs=s, outputs=actor_value)
            # we don't need to compile this model, because we won't perform backpropagation on it.

            #############

            td_error = layers.Input(shape=(1,), dtype='float32', name='td_error')

            def custom_loss(y_true, y_pred):  # (a_indices_one_hot, actor.output)
                y_pred_clipped = K.clip(y_pred, 1e-8, 1 - 1e-8)  # we set boundaries so we won't take log of 0\1
                log_lik = y_true * K.log(y_pred_clipped)  # log_probability
                loss = K.sum(-log_lik * td_error)  # K.mean ?
                return loss

            actor = models.Model(inputs=[s, td_error], outputs=actor_value)  # policy_model
            optimizer_actor = utils.Optimizers.keras_get_optimizer(self.dnn.optimizer_type, self.lr_actor)
            actor.compile(optimizer_actor, loss=custom_loss)

            return actor, critic, policy

        def load_model_file(self):
            print("...Loading Keras h5...")
            self.actor = models.load_model(self.h5_file_actor)
            self.critic = models.load_model(self.h5_file_critic)

        def save_model_file(self):
            print("...Saving Keras h5...")
            self.actor.save(self.h5_file_actor)
            self.critic.save(self.h5_file_critic)

    class AC_DNN_Torch(T.nn.Module):

        def __init__(self, dnn, chkpt_dir, device_str):
            super(DNN.AC_DNN_Torch, self).__init__()

            self.dnn = dnn

            self.model_file = os.path.join(chkpt_dir, 'ac_torch_' + self.dnn.name)

            self.build_network()

            self.optimizer = utils.Optimizers.torch_get_optimizer(
                self.dnn.optimizer_type, self.parameters(), self.dnn.lr)

            self.device = utils.DeviceSetUtils.torch_get_device_according_to_device_type(device_str)
            self.to(self.device)

        def load_model_file(self):
            print("...Loading Torch file...")
            self.load_state_dict(T.load(self.model_file))

        def save_model_file(self):
            print("...Saving Torch file...")
            T.save(self.state_dict(), self.model_file)

        def build_network(self):
            self.fc1 = T.nn.Linear(*self.input_dims, self.fc_layers_dims[0])
            self.fc2 = T.nn.Linear(self.fc_layers_dims[0], self.fc_layers_dims[1])

            if self.network_type == NETWORK_TYPE_SEPARATE:  # build_A_or_C_network
                self.fc3 = T.nn.Linear(self.fc_layers_dims[1], self.n_outputs if self.is_actor else 1)

            else:  # self.network_type == NETWORK_TYPE_SHARED    # build_A_and_C_network
                self.fc3 = T.nn.Linear(self.fc_layers_dims[1], self.n_outputs)  # Actor layer
                self.v = T.nn.Linear(self.fc_layers_dims[1], 1)  # Critic layer

        def forward(self, s):
            state_value = T.tensor(s, dtype=T.float).to(self.device)

            state_value = self.fc1(state_value)
            state_value = F.relu(state_value)

            state_value = self.fc2(state_value)
            state_value = F.relu(state_value)

            if self.network_type == NETWORK_TYPE_SEPARATE:  # forward_A_or_C_network
                state_value = self.fc3(state_value)
                return state_value

            else:  # self.network_type == NETWORK_TYPE_SHARED    # forward_A_and_C_network
                actor_value = self.fc3(state_value)  # Actor value
                v = self.v(state_value)  # Critic value
                return actor_value, v


class AC(object):

    def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, network_type, chkpt_dir):

        self.GAMMA = custom_env.GAMMA

        self.a_log_probs = None

        self.is_discrete_action_space = custom_env.is_discrete_action_space
        self.n_actions = custom_env.n_actions
        if self.is_discrete_action_space:
            self.action_space = custom_env.action_space
            self.action_boundary = None
        else:
            self.action_space = None
            self.action_boundary = custom_env.action_boundary

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic if lr_critic is not None else lr_actor

        self.network_type = network_type

        if self.network_type == NETWORK_TYPE_SEPARATE:
            self.actor_base = DNN(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'Actor', NETWORK_TYPE_SEPARATE, is_actor=True)
            self.critic_base = DNN(
                custom_env, fc_layers_dims, optimizer_type, lr_critic, 'Critic', NETWORK_TYPE_SEPARATE, is_actor=False)
        else:  # self.network_type == NETWORK_TYPE_SHARED
            self.actor_critic_base = DNN(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'ActorCritic', NETWORK_TYPE_SHARED)

        self.chkpt_dir = chkpt_dir

    def create_ac_tensorflow(self, device_map):
        return AC.AC_TF(self, device_map)

    def create_ac_keras(self):
        return AC.AC_Keras(self)

    def create_ac_torch(self):
        return AC.AC_Torch(self)

    class AC_TF(object):

        def __init__(self, ac, device_map):

            self.ac = ac

            self.sess = utils.DeviceSetUtils.tf_get_session_according_to_device(device_map)

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = self.ac.actor_base.create_dnn_tensorflow(self.sess)
                self.critic = self.ac.critic_base.create_dnn_tensorflow(self.sess)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = self.ac.actor_critic_base.create_dnn_tensorflow(self.sess)

            self.saver = tf.train.Saver()
            self.checkpoint_file = os.path.join(self.ac.chkpt_dir, 'ac_tf.ckpt')

            self.sess.run(tf.global_variables_initializer())

        def choose_action(self, s):
            s = s[np.newaxis, :]

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.predict(s)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value, _ = self.actor_critic.predict(s)

            if self.ac.is_discrete_action_space:
                return self.choose_action_discrete(actor_value)
            else:
                return self.choose_action_continuous(actor_value)

        def choose_action_discrete(self, pi):
            probabilities = tf.nn.softmax(pi)[0]
            a_index = np.random.choice(self.ac.action_space, p=probabilities)
            a = self.ac.action_space[a_index]
            return a

        def choose_action_continuous(self, actor_value):
            mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
            sigma = tf.exp(sigma_unactivated)
            actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
            action_tensor = actions_probs.sample(sample_shape=[self.ac.n_actions])
            action_tensor = tf.nn.tanh(action_tensor)
            action_tensor = tf.multiply(action_tensor, self.ac.action_boundary)
            a = action_tensor.item()
            a = np.array(a).reshape((1,))
            return a

        # def choose_action_discrete(self, pi):
        #     probabilities = tf.nn.softmax(pi)[0]
        #     actions_probs = tfp.distributions.Categorical(probs=probabilities)
        #     action_tensor = actions_probs.sample()
        #     self.ac.a_log_probs = actions_probs.log_prob(action_tensor)
        #     a_index = action_tensor.item()
        #     a = self.action_space[a_index]
        #     return a
        #
        # def choose_action_continuous(self, actor_value):
        #     mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
        #     sigma = tf.exp(sigma_unactivated)
        #     actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)
        #     action_tensor = actions_probs.sample(sample_shape=[self.n_actions])
        #     self.ac.a_log_probs = actions_probs.log_prob(action_tensor)
        #     action_tensor = tf.nn.tanh(action_tensor)
        #     action_tensor = tf.multiply(action_tensor, self.action_boundary)
        #     a = action_tensor.item()
        #     a = np.array(a).reshape((1,))
        #     return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                v = self.critic.predict(s)
                v_ = self.critic.predict(s_)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                _, v = self.actor_critic.predict(s)
                _, v_ = self.actor_critic.predict(s_)

            v_target = r + self.ac.GAMMA * v_ * (1 - int(is_terminal))
            td_error = v_target - v

            if self.ac.is_discrete_action_space:
                a = self.ac.action_space.index(a)

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.train(s, td_error, a)
                self.critic.train(s, td_error)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.train(s, td_error, a)

        def load_model_file(self):
            print("...Loading TF checkpoint...")
            self.saver.restore(self.sess, self.checkpoint_file)

        def save_model_file(self):
            print("...Saving TF checkpoint...")
            self.saver.save(self.sess, self.checkpoint_file)

    class AC_Keras(object):

        def __init__(self, ac):

            self.ac = ac

            self.dnn = self.ac.actor_critic_base.create_dnn_keras(self.ac.lr_actor, self.ac.lr_critic, self.ac.chkpt_dir)

        def choose_action(self, s):
            s = s[np.newaxis, :]
            actor_value = self.dnn.policy.predict(s)
            if self.ac.is_discrete_action_space:
                return self.choose_action_discrete(actor_value[0])
            else:
                return self.choose_action_continuous(actor_value)

        def choose_action_discrete(self, probabilities):
            a_index = np.random.choice(self.ac.action_space, p=probabilities)
            a = self.ac.action_space[a_index]
            return a

        def choose_action_continuous(self, actor_value):
            mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
            sigma = K.exp(sigma_unactivated)
            actions_probs = tfp.distributions.Normal(loc=mu, scale=sigma)  # K.random_normal(mean=mu, std=sigma)
            action_tensor = actions_probs.sample(sample_shape=[self.ac.n_actions])
            action_tensor = K.tanh(action_tensor)
            action_tensor = tf.multiply(action_tensor, self.ac.action_boundary)
            a = action_tensor.item()
            a = np.array(a).reshape((1,))
            return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            if not self.ac.is_discrete_action_space:
                sys.exit('Keras currently only works with Discrete action spaces')

            s = s[np.newaxis, :]
            s_ = s_[np.newaxis, :]

            v = self.dnn.critic.predict(s)
            v_ = self.dnn.critic.predict(s_)

            v_target = r + self.ac.GAMMA * v_ * (1 - int(is_terminal))
            td_error = v_target - v

            if self.ac.is_discrete_action_space:
                a_indices_one_hot = np.zeros(self.ac.n_actions, dtype=np.int8)
                a_index = self.ac.action_space.index(a)
                a_indices_one_hot[a_index] = 1
                a_indices_one_hot = a_indices_one_hot[np.newaxis, :]
                self.dnn.actor.fit([s, td_error], a_indices_one_hot, verbose=0)
            else:
                pass

            self.dnn.critic.fit(s, v_target, verbose=0)

        def load_model_file(self):
            print("...Loading Keras models...")
            self.dnn.load_model_file()

        def save_model_file(self):
            print("...Saving Keras models...")
            self.dnn.save_model_file()

    class AC_Torch(object):

        def __init__(self, ac):

            self.ac = ac

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor = self.ac.actor_base.create_dnn_torch(self.ac.chkpt_dir)
                self.critic = self.ac.critic_base.create_dnn_torch(self.ac.chkpt_dir)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic = self.ac.actor_critic_base.create_dnn_torch(self.ac.chkpt_dir)

        def choose_action(self, s):
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                actor_value = self.actor.forward(s)
                device = self.actor.device
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                actor_value, _ = self.actor_critic.forward(s)
                device = self.actor_critic.device

            if self.ac.is_discrete_action_space:
                return self.choose_action_discrete(actor_value, device)
            else:
                return self.choose_action_continuous(actor_value, device)

        def choose_action_discrete(self, pi, device):
            probabilities = F.softmax(pi)
            actions_probs = distributions.Categorical(probabilities)
            action_tensor = actions_probs.sample()
            self.ac.a_log_probs = actions_probs.log_prob(action_tensor).to(device)
            a_index = action_tensor.item()
            a = self.ac.action_space[a_index]
            return a

        def choose_action_continuous(self, actor_value, device):
            mu, sigma_unactivated = actor_value  # Mean (μ), STD (σ)
            sigma = T.exp(sigma_unactivated)
            actions_probs = distributions.Normal(mu, sigma)
            action_tensor = actions_probs.sample(sample_shape=T.Size([self.ac.n_actions]))
            self.ac.a_log_probs = actions_probs.log_prob(action_tensor).to(device)
            action_tensor = T.tanh(action_tensor)
            action_tensor = T.mul(action_tensor, self.ac.action_boundary)
            a = action_tensor.item()
            a = np.array(a).reshape((1,))
            return a

        def learn(self, s, a, r, s_, is_terminal):
            # print('Learning Session')

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                r = T.tensor(r, dtype=T.float).to(self.critic.device)
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                v = self.critic.forward(s)
                v_ = self.critic.forward(s_)
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                r = T.tensor(r, dtype=T.float).to(self.actor_critic.device)
                self.actor_critic.optimizer.zero_grad()
                _, v = self.actor_critic.forward(s)
                _, v_ = self.actor_critic.forward(s_)

            v_target = r + self.ac.GAMMA * v_ * (1 - int(is_terminal))
            td_error = v_target - v

            actor_loss = -self.ac.a_log_probs * td_error
            critic_loss = td_error ** 2
            (actor_loss + critic_loss).backward()

            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.optimizer.step()

        def load_model_file(self):
            print("...Loading Torch models...")
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.load_model_file()
                self.critic.load_model_file()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.load_model_file()

        def save_model_file(self):
            print("...Saving Torch models...")
            if self.ac.network_type == NETWORK_TYPE_SEPARATE:
                self.actor.save_model_file()
                self.critic.save_model_file()
            else:  # self.ac.network_type == NETWORK_TYPE_SHARED
                self.actor_critic.save_model_file()


class Agent(object):

    def __init__(self, custom_env, fc_layers_dims=(400, 300),
                 optimizer_type=utils.Optimizers.OPTIMIZER_Adam, lr_actor=0.0001, lr_critic=None,
                 device_type=None, lib_type=utils.LIBRARY_TF,
                 base_dir=''):

        self.GAMMA = custom_env.GAMMA
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = lr_actor
        self.BETA = lr_critic if lr_critic is not None else lr_actor

        # sub_dir = utils.General.get_file_name(None, self, self.BETA) + '/'
        sub_dir = ''
        self.chkpt_dir = base_dir + sub_dir
        utils.General.make_sure_dir_exists(self.chkpt_dir)

        network_type = NETWORK_TYPE_SHARED if (lr_critic is None or lib_type == utils.LIBRARY_KERAS) else NETWORK_TYPE_SEPARATE
        ac_base = AC(custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, network_type, self.chkpt_dir)
        if lib_type == utils.LIBRARY_TF:
            self.ac = ac_base.create_ac_tensorflow(device_type)
        elif lib_type == utils.LIBRARY_KERAS:
            self.ac = ac_base.create_ac_keras()
        else:
            self.ac = ac_base.create_ac_torch()

    def choose_action(self, s):
        return self.ac.choose_action(s)

    def learn(self, s, a, r, s_, is_terminal):
        self.ac.learn(s, a, r, s_, is_terminal)

    def save_models(self):
        self.ac.save_model_file()

    def load_models(self):
        self.ac.load_model_file()


def train_agent(custom_env, agent, n_episodes,
                enable_models_saving, load_checkpoint,
                visualize=False, record=False):

    scores_history, learn_episode_index, max_avg = utils.SaverLoader.load_training_data(agent, load_checkpoint)

    env = custom_env.envs

    if record:
        env = wrappers.Monitor(
            env, 'recordings/AC/', force=True,
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
            agent.learn(s, a, r, s_, done)
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        scores_history.append(ep_score)
        utils.SaverLoader.pickle_save(scores_history, 'scores_history_train_total', agent.chkpt_dir)

        current_avg = utils.Printer.print_training_progress(i, ep_score, scores_history, avg_num=custom_env.window, ep_start_time=ep_start_time)

        if enable_models_saving and current_avg is not None and \
                (max_avg is None or current_avg >= max_avg):
            max_avg = current_avg
            utils.SaverLoader.pickle_save(max_avg, 'max_avg', agent.chkpt_dir)
            utils.SaverLoader.save_training_data(agent, i, scores_history)

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Training Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
          (n_episodes - starting_ep, str(datetime.datetime.now() - train_start_time).split('.')[0]), '\n')

    return scores_history


def play(env_type, lib_type=utils.LIBRARY_TORCH, enable_models_saving=False, load_checkpoint=False):
    if lib_type == utils.LIBRARY_TF:
        print('\n', "Algorithm currently doesn't work with TensorFlow", '\n')
        return

    # SHARED vs SEPARATE explanation:
    #   SHARED is very helpful in more complex environments (like LunarLander)
    #   you can get away with SEPARATE in less complex environments (like MountainCar)

    if env_type == 0:
        custom_env = Envs.ClassicControl.CartPole()
        fc_layers_dims = [32, 32]
        optimizer_type = utils.Optimizers.OPTIMIZER_Adam
        alpha = 0.0001  # 0.00001
        beta = alpha * 5
        n_episodes = 2500

    elif env_type == 1:
        # custom_env = Envs.Box2D.LunarLander()
        custom_env = Envs.ClassicControl.Pendulum()
        fc_layers_dims = [2048, 512]  # Keras: [1024, 512]
        optimizer_type = utils.Optimizers.OPTIMIZER_Adam
        alpha = 0.00001
        beta = alpha * 5 if lib_type == utils.LIBRARY_KERAS else None
        n_episodes = 2000

    else:
        custom_env = Envs.ClassicControl.MountainCarContinuous()
        fc_layers_dims = [256, 256]
        optimizer_type = utils.Optimizers.OPTIMIZER_Adam
        alpha = 0.000005
        beta = alpha * 2
        n_episodes = 100  # longer than 100 --> instability (because the value function estimation is unstable)

    if lib_type == utils.LIBRARY_TORCH and custom_env.input_type != Envs.INPUT_TYPE_OBSERVATION_VECTOR:
        print('\n', 'the Torch implementation of the Algorithm currently works only with INPUT_TYPE_OBSERVATION_VECTOR!', '\n')
        return

    custom_env.env.seed(28)

    utils.DeviceSetUtils.set_device(lib_type, devices_dict=None)

    method_name = 'AC'
    base_dir = 'tmp/' + custom_env.file_name + '/' + method_name + '/'

    agent = Agent(custom_env, fc_layers_dims,
                  optimizer_type, lr_actor=alpha, lr_critic=beta,
                  lib_type=lib_type, base_dir=base_dir)

    scores_history = train_agent(custom_env, agent, n_episodes,
                                 enable_models_saving, load_checkpoint)
    reinforcement_learning.utils.plotter.Plotter.plot_running_average(
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
    play(0, lib_type=utils.LIBRARY_KERAS)          # CartPole (0), Pendulum (1), MountainCarContinuous (2)
