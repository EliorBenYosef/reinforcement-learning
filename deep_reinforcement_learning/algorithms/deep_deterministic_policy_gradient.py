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
import torch.optim.adagrad as optim_adagrad
import torch.optim.adadelta as optim_adadelta

import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers

import utils
from deep_reinforcement_learning.envs import Envs
from deep_reinforcement_learning.replay_buffer import ReplayBuffer


# https://arxiv.org/pdf/1509.02971.pdf


class DNN:

    class AC_DNN_TF(object):

        def __init__(self, custom_env, fc_layers_dims, sess, optimizer_type, lr, name):
            self.name = name

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            self.optimizer_type = optimizer_type
            self.lr = lr

            # relevant for the Actor only:
            self.memory_batch_size = custom_env.memory_batch_size
            self.action_boundary = custom_env.action_boundary if not custom_env.is_discrete_action_space \
                else None

            self.sess = sess

        def create_actor(self):
            return DNN.AC_DNN_TF.Actor(self)

        def create_critic(self):
            return DNN.AC_DNN_TF.Critic(self)

        class Actor(object):

            def __init__(self, ac):
                self.ac = ac

                self.build_network()

                self.params = tf.trainable_variables(scope=self.ac.name)

                self.mu_gradients = tf.gradients(self.mu, self.params, -self.a_grad)
                self.normalized_mu_gradients = list(
                    map(lambda x: tf.div(x, self.ac.memory_batch_size), self.mu_gradients))

                if self.ac.optimizer_type == utils.OPTIMIZER_SGD:
                    optimizer = tf.train.MomentumOptimizer(self.ac.lr, momentum=0.9)  # SGD + momentum
                    # optimizer = tf.train.GradientDescentOptimizer(self.dqn.ALPHA)  # SGD?
                elif self.ac.optimizer_type == utils.OPTIMIZER_Adagrad:
                    optimizer = tf.train.AdagradOptimizer(self.ac.lr)
                elif self.ac.optimizer_type == utils.OPTIMIZER_Adadelta:
                    optimizer = tf.train.AdadeltaOptimizer(self.ac.lr)
                elif self.ac.optimizer_type == utils.OPTIMIZER_RMSprop:
                    optimizer = tf.train.RMSPropOptimizer(self.ac.lr)
                else:  # self.ac.optimizer_type == utils.OPTIMIZER_Adam
                    optimizer = tf.train.AdamOptimizer(self.ac.lr)

                self.optimize = optimizer.apply_gradients(zip(self.normalized_mu_gradients, self.params))  # train_op

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

                if self.ac.optimizer_type == utils.OPTIMIZER_SGD:
                    optimizer = tf.train.MomentumOptimizer(self.ac.lr, momentum=0.9)  # SGD + momentum
                    # optimizer = tf.train.GradientDescentOptimizer(self.dqn.ALPHA)  # SGD?
                elif self.ac.optimizer_type == utils.OPTIMIZER_Adagrad:
                    optimizer = tf.train.AdagradOptimizer(self.ac.lr)
                elif self.ac.optimizer_type == utils.OPTIMIZER_Adadelta:
                    optimizer = tf.train.AdadeltaOptimizer(self.ac.lr)
                elif self.ac.optimizer_type == utils.OPTIMIZER_RMSprop:
                    optimizer = tf.train.RMSPropOptimizer(self.ac.lr)
                else:  # self.ac.optimizer_type == utils.OPTIMIZER_Adam
                    optimizer = tf.train.AdamOptimizer(self.ac.lr)

                self.optimize = optimizer.minimize(self.loss)  # train_op

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

    class AC_DNN_Torch(nn.Module):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr, name, chkpt_dir, is_actor, device_type='cuda'):
            super(DNN.AC_DNN_Torch, self).__init__()

            self.is_actor = is_actor
            self.name = name

            self.model_file = os.path.join(chkpt_dir, 'ddpg_torch_' + name)

            self.input_dims = custom_env.input_dims
            self.fc_layers_dims = fc_layers_dims
            self.n_actions = custom_env.n_actions

            # relevant for the Actor only:
            # self.memory_batch_size = custom_env.memory_batch_size
            self.action_boundary = custom_env.action_boundary if not custom_env.is_discrete_action_space \
                else None

            self.build_network()

            if optimizer_type == utils.OPTIMIZER_SGD:
                self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
            elif optimizer_type == utils.OPTIMIZER_Adagrad:
                self.optimizer = optim_adagrad.Adagrad(self.parameters(), lr=lr)
            elif optimizer_type == utils.OPTIMIZER_Adadelta:
                self.optimizer = optim_adadelta.Adadelta(self.parameters(), lr=lr)
            elif optimizer_type == utils.OPTIMIZER_RMSprop:
                self.optimizer = optim_rmsprop.RMSprop(self.parameters(), lr=lr)
            else:  # optimizer_type == utils.OPTIMIZER_Adam
                self.optimizer = optim.Adam(self.parameters(), lr=lr)

            self.device = utils.get_torch_device_according_to_device_type(device_type)
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


class AC(object):

    class AC_TF(object):

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir, device_type):

            self.GAMMA = 0.99
            self.TAU = tau

            self.sess = utils.get_tf_session_according_to_device_type(device_type)

            #############################

            # Networks:

            self.actor = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'Actor'
            ).create_actor()
            self.target_actor = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_actor, 'ActorTarget'
            ).create_actor()

            self.critic = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'Critic'
            ).create_critic()
            self.target_critic = DNN.AC_DNN_TF(
                custom_env, fc_layers_dims, self.sess, optimizer_type, lr_critic, 'CriticTarget'
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

        def __init__(self, custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir):

            self.GAMMA = 0.99
            self.TAU = tau

            #############################

            # Networks:

            self.actor = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'Actor', chkpt_dir, is_actor=True
            )
            self.target_actor = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_actor, 'ActorTarget', chkpt_dir, is_actor=True
            )

            self.critic = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_critic, 'Critic', chkpt_dir, is_actor=False
            )
            self.target_critic = DNN.AC_DNN_Torch(
                custom_env, fc_layers_dims, optimizer_type, lr_critic, 'CriticTarget', chkpt_dir, is_actor=False
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

    def __init__(self, custom_env, fc_layers_dims=(400, 300),  # paper
                 optimizer_type=utils.OPTIMIZER_Adam, lr_actor=0.0001, lr_critic=0.001, tau=0.001,  # paper
                 memory_size=None, memory_batch_size=None,
                 device_type=None, lib_type=utils.LIBRARY_TF):

        self.GAMMA = 0.99  # paper
        self.fc_layers_dims = fc_layers_dims

        self.optimizer_type = optimizer_type
        self.ALPHA = lr_actor
        self.BETA = lr_critic

        self.TAU = tau

        self.noise = OUActionNoise(mu=np.zeros(custom_env.n_actions), sigma=0.2)

        self.memory_size = memory_size if memory_size is not None else custom_env.memory_size
        self.memory_batch_size = memory_batch_size if memory_batch_size is not None else (
            64 if custom_env.input_type == Envs.INPUT_TYPE_OBSERVATION_VECTOR else 16  # paper
        )
        self.memory = ReplayBuffer(custom_env, self.memory_size, lib_type, is_discrete_action_space=False)

        chkpt_dir = 'tmp/' + custom_env.file_name + '/DDPG/NNs/'
        if lib_type == utils.LIBRARY_TF:
            self.ac = AC.AC_TF(custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir,
                                    device_type)
        else:
            self.ac = AC.AC_Torch(custom_env, fc_layers_dims, optimizer_type, lr_actor, lr_critic, tau, chkpt_dir)

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
                      batch_s, batch_a, batch_r, batch_s_, batch_terminal)

    def save_models(self):
        self.ac.save_model_file()

    def load_models(self):
        self.ac.load_model_file()


def train(custom_env, agent, n_episodes,
          enable_models_saving, load_checkpoint, save_checkpoint=25,
          visualize=False, record=False):

    best_score_episode_index = -1
    if load_checkpoint:
        try:
            agent.load_models()
            print('...Loading best_score_data...')
            best_score_data = utils.pickle_load('best_score_data')
            best_score = best_score_data[0]
            best_score_episode_index = best_score_data[1]
        except (ValueError, tf.OpError):
            print('...No models to load...')
        except FileNotFoundError:
            print('...No best_score_data to load...')

    env = custom_env.env

    if record:
        env = wrappers.Monitor(
            env, 'recordings/DDPG/', force=True,
            video_callable=lambda episode_id: episode_id == 0 or episode_id == (n_episodes - 1)
        )

    print('\n', 'Game Started', '\n')

    scores_history = []

    for i in range(best_score_episode_index + 1, n_episodes):
        done = False
        ep_score = 0

        agent.noise.reset()

        observation = env.reset()
        s = custom_env.get_state(observation, None)

        if visualize and i == n_episodes - 1:
            env.render()

        while not done:
            a = agent.choose_action(s, training_mode=True)
            observation_, r, done, info = env.step(a)
            r = custom_env.update_reward(r, done, info)
            s_ = custom_env.get_state(observation_, s.copy())
            ep_score += r
            agent.store_transition(s, a, r, s_, done)
            agent.learn()
            observation, s = observation_, s_

            if visualize and i == n_episodes - 1:
                env.render()

        # if enable_models_saving and (i + 1) % save_checkpoint == 0:
        if enable_models_saving and ('best_score' not in locals() or ep_score > best_score):
            best_score = ep_score
            best_score_episode_index = i
            utils.pickle_save([best_score, best_score_episode_index], 'best_score_data')
            agent.save_models()

        scores_history.append(ep_score)

        utils.print_training_progress(i, ep_score, scores_history, custom_env.window)
        print('Best - Episode: %d, Score: %d' % (best_score_episode_index + 1, best_score))

        if visualize and i == n_episodes - 1:
            env.close()

    print('\n', 'Game Ended', '\n')

    return scores_history


def play(env_type, lib_type=utils.LIBRARY_TF, enable_models_saving=False, load_checkpoint=False):
    if lib_type == utils.LIBRARY_KERAS:
        print('\n', "Algorithm currently doesn't work with Keras", '\n')
        return

    if env_type == 0:
        custom_env = Envs.ClassicControl.Pendulum()
        fc_layers_dims = (800, 600)
        optimizer_type = utils.OPTIMIZER_Adam
        alpha = 0.00005
        beta = 0.0005
        n_episodes = 1000

    # elif env_type == 1:
    # custom_env = Envs.Box2D.BipedalWalker()
    # fc_layers_dims = (400, 300)
    # optimizer_type = utils.OPTIMIZER_Adam
    # alpha = 0.00005
    # beta = 0.0005
    # n_episodes = 5000

    else:
        # custom_env = Envs.Box2D.LunarLanderContinuous()
        custom_env = Envs.ClassicControl.MountainCarContinuous()
        fc_layers_dims = (400, 300)
        optimizer_type = utils.OPTIMIZER_Adam
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

    custom_env.env.seed(28)

    if lib_type == utils.LIBRARY_TF:
        utils.tf_set_device()

    agent = Agent(
        custom_env, fc_layers_dims, optimizer_type, alpha, beta, tau,
        memory_batch_size=custom_env.memory_batch_size, lib_type=lib_type
    )

    scores_history = train(custom_env, agent, n_episodes, enable_models_saving, load_checkpoint)

    utils.plot_running_average(
        custom_env.name, scores_history, window=custom_env.window, show=False,
        file_name=utils.get_plot_file_name(custom_env.file_name, agent, memory=True, beta=agent.BETA)
    )


if __name__ == '__main__':
    play(0, lib_type=utils.LIBRARY_TF)        # Pendulum (0), MountainCarContinuous (1)
